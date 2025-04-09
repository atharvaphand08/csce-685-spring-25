import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from PIL import Image
import requests

import sys
sys.path.insert(0, "$SCRATCH/nltk_local")

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import timm
import math

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModel
import shutil

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import json

file_path = 'fixed_flickr_data.csv'

# Read the file, using the first line as header
flickr = pd.read_csv(file_path)

print(flickr.info())

# Define the image folder path
image_folder = '/scratch/user/ahphand/flickr30k_images/flickr30k_images'
flickr.columns = flickr.columns.str.strip()
print(flickr.columns)
# Add a new column with the full image path
flickr['path'] = image_folder + '/' + flickr['image_name']
print(flickr.head())

flickr=flickr[['image_name','comment','path']]

"""## III. Data Splitting"""

print("III. Data Splitting")

# Number of images in Flickr30k dataset (each image has 5 captions)
print(flickr['image_name'].nunique())

# Step 1: Get unique image names
image_names = flickr['image_name'].unique()

# Step 2: Split image names into training and temporary (validation + test) sets
train_image_names, temp_image_names = train_test_split(image_names, test_size=0.2, random_state=42)

# Step 3: Split the temporary image names into validation and test sets
val_image_names, test_image_names = train_test_split(temp_image_names, test_size=0.5, random_state=42)

# Step 4: Filter the original DataFrame based on the image names in each split
train_df = flickr[flickr['image_name'].isin(train_image_names)]
val_df = flickr[flickr['image_name'].isin(val_image_names)]
test_df = flickr[flickr['image_name'].isin(test_image_names)]

print("Length of traning set",len(train_df))
print("Length of validation set",len(val_df))
print("Length of testing set",len(test_df))

# Define a function to filter images based on DataFrame
def filter_images(df):
    filtered_images = [(row['path'], row['comment']) for index, row in df.iterrows()]
    return filtered_images

# Filter images for training and testing sets
train_images = filter_images(train_df)
test_images = filter_images(test_df)
val_images = filter_images(val_df)

"""## IV. Prepare data for training"""

print("IV. Prepare data for training")

# Initiate bert tokenizer
tokenizer = BertTokenizer.from_pretrained('/scratch/user/ahphand/bert_tokenizer')

class CaptionDataset(Dataset):
    def __init__(self, data, transform=None, tokenizer=tokenizer, max_length=50):
        """
        Args:
            data (list of tuples): A list where each tuple contains (image_path, caption).
            transform (callable, optional): Optional transform to be applied on a sample (image).
            tokenizer (BertTokenizer): Pretrained tokenizer for BERT (or any transformer model).
            max_length (int, optional): Maximum length of encoded captions (for padding/truncation).
        """
        self.data = data
        self.transform = transform
        self.max_length = max_length

        # Initialize the BERT tokenizer
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained('/scratch/user/ahphand/bert_tokenizer')

    def encode_caption(self, caption):
        """
        Converts a caption (string) into a list of token IDs using the BERT tokenizer.
        """
        # Tokenize and encode the caption with special tokens
        encoding = self.tokenizer.encode_plus(
            caption,
            add_special_tokens=True,  # Add [CLS] and [SEP]
            padding='max_length',     # Pad to max_length
            truncation=True,          # Truncate to max_length if needed
            max_length=self.max_length,  # Max length of caption
            return_tensors='pt',      # Return PyTorch tensors
        )

        # Return the encoded caption as a tensor (token IDs)
        return encoding['input_ids'].squeeze(0)  # Remove batch dimension

    def decode_caption(self, encoded_caption):
        """
        Decodes the tokenized caption back to a string using the BERT tokenizer.
        """
        decoded_caption = self.tokenizer.decode(encoded_caption, skip_special_tokens=True)
        return decoded_caption

    def convert_to_tokens(self, encoded_caption):
        tokens_converted = self.tokenizer.convert_ids_to_tokens(encoded_caption)
        return tokens_converted

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, encoded_caption)
                - image: The transformed image tensor.
                - encoded_caption: The encoded caption as a tensor of token IDs.
        """
        image_path, caption = self.data[idx]

        # Load the image
        image = Image.open(image_path).convert('RGB')

        # Apply the image transformation (if any)
        if self.transform:
            image = self.transform(image)

        # Encode the caption using the BERT tokenizer
        encoded_caption = self.encode_caption(caption)

        return image, encoded_caption

# Training set transformations: Include augmentations like random horizontal flip, rotation, etc.
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),               # Resize to a fixed size
    transforms.RandomHorizontalFlip(),           # Randomly flip the image
    transforms.RandomRotation(20),              # Random rotation
    transforms.ToTensor(),                      # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet statistics
])

# Validation and test set transformations: Only resizing and normalization
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),               # Resize to a fixed size
    transforms.ToTensor(),                       # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet statistics
])

# Create the CaptionDataset instance
train_dataset = CaptionDataset(data=train_images, transform=train_transform)
val_dataset = CaptionDataset(data=val_images, transform=val_test_transform)
test_dataset = CaptionDataset(data=test_images, transform=val_test_transform)

# Save Dataset (as an example, saving image paths and captions)
def save_dataset(dataset, file_path):
    data = []
    for image_path, caption in dataset.data:  # Access raw data instead of tensors
        data.append({"image_path": image_path, "caption": caption})  # Store only file paths and text captions
    with open(file_path, 'w') as f:
        json.dump(data, f)

# Save dataset
save_dataset(train_dataset, 'train_dataset.json')
save_dataset(val_dataset, 'val_dataset.json')
save_dataset(test_dataset, 'test_dataset.json')

# Save training settings (like transformations)
transform_settings = {
    'train_transform': str(train_transform),
    'val_test_transform': str(val_test_transform)
}
with open('transform_settings.json', 'w') as f:
    json.dump(transform_settings, f)

print("Created and Saved the CaptionDataset instance - train_dataset, val_dataset, test_dataset")

def collate_fn(batch):
    """
    Custom collate function to handle padding and batching for image-caption pairs.

    Args:
        batch (list of tuples): Each tuple contains (image, caption) where
                                 image is a tensor and caption is a string.

    Returns:
        dict: Contains 'images' (batch of images) and 'captions' (padded encoded captions).
    """
    # Initialize lists for images and captions
    images = []
    captions = []

    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('/scratch/user/ahphand/bert_tokenizer')

    for image, caption in batch:
        # Append the image tensor (already preprocessed and transformed)
        images.append(image)

        # Check if the caption is already tokenized (if it's a list of integers, skip tokenization)
        if isinstance(caption, str):  # Only tokenize if it's raw text
            encoding = tokenizer.encode_plus(
                caption,               # The raw caption text
                add_special_tokens=True,  # Add [CLS] and [SEP]
                padding='max_length',  # Pad to the max length
                truncation=True,       # Truncate if longer than max length
                max_length=50,         # Maximum length of caption (adjust as needed)
                return_tensors='pt',   # Return tensors
            )
            # Get the tokenized caption
            captions.append(encoding['input_ids'].squeeze(0))  # Remove the batch dimension
        else:  # If it's already tokenized (list of integers), just append
#             captions.append(torch.tensor(caption))  # Convert list of integers to tensor
            captions.append(caption.clone().detach())


    # Stack all the images into a batch
    images = torch.stack(images, dim=0)  # Shape: [batch_size, 3, H, W]

    # Pad the captions to make sure all captions in the batch have the same length
    captions = pad_sequence(captions, batch_first=True, padding_value=tokenizer.pad_token_id)

    return {
        'images': images,        # Batch of images
        'captions': captions     # Padded batch of tokenized captions
    }

batch_size = 32
# Create DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn,num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn,num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn,num_workers=4)

print("Created Loaders!")

# Example of iterating through the DataLoader
for batch in train_loader:
    images = batch['images']
    captions = batch['captions']
    print("Batch of images:", images.shape)  # Should be [batch_size, 3, 224, 224]
    print("Batch of captions:", captions.shape)  # Should be [batch_size, max_caption_length]
    print("Captions:", captions)
    break  # For illustration, we break after the first batch

"""## V. Encoder and decoder architecture"""

# ---- EfficientNetV2 Encoder ----
class EfficientNetEncoder(nn.Module):
    def __init__(self, embed_size):
        super(EfficientNetEncoder, self).__init__()
        
        # Load EfficientNetV2-S without pretrained weights
        self.efficientnet = timm.create_model("tf_efficientnetv2_s", pretrained=False, features_only=True)
        
        weights_path = "/scratch/user/ahphand/efficientnetv2_s_weights.pth"
        state_dict = torch.load(weights_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        self.efficientnet.load_state_dict(state_dict, strict=False)

        # Freeze the parameters
        for param in self.efficientnet.parameters():
            param.requires_grad = False  

        # Get the output feature dimension
        feature_dim = self.efficientnet.feature_info[-1]['num_chs']
        self.fc = nn.Linear(feature_dim, embed_size)   

    def forward(self, images):
        with torch.no_grad():
            features = self.efficientnet(images)[-1]  
        features = features.mean([2, 3])  
        return self.fc(features).unsqueeze(1)  # (batch_size, 1, embed_size)

# ---- Positional Encoding ----
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, max_len, embed_size)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

# ---- Transformer Decoder ----
class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_heads=8, num_layers=6):
        super(TransformerDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, 
            nhead=num_heads, 
            dim_feedforward=hidden_size,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, features, captions):
        """
        Args:
            features: (batch_size, 1, embed_size) -> Image features from encoder
            captions: (batch_size, seq_len) -> Tokenized input captions
        """
        embeddings = self.embed(captions)  # (batch_size, seq_len, embed_size)
        embeddings = self.pos_encoder(embeddings)  # Add positional encoding

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(captions.size(1)).to(captions.device)

        output = self.transformer_decoder(
            embeddings,  # tgt (captions)
            features,  # memory (image features)
            tgt_mask=tgt_mask
        )
        return self.fc_out(output)  # (batch_size, seq_len, vocab_size)

# ---- Combined Model ----
class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_heads=8, num_layers=6):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EfficientNetEncoder(embed_size)
        self.decoder = TransformerDecoder(embed_size, hidden_size, vocab_size, num_heads, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)  
        outputs = self.decoder(features, captions)  
        return outputs

# Model hyperparameters
embed_size = 512
hidden_size = 512
vocab_size=len(train_dataset.tokenizer.get_vocab())
print("Vocab size ", vocab_size)
num_heads = 8
num_layers = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageCaptioningModel(embed_size, hidden_size, vocab_size, num_heads, num_layers).to(device)
print(model)

"""## VI. Training"""

learning_rate = 1e-4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, threshold=0.01, min_lr=1e-6)

# Initialize TensorBoard writer
writer = SummaryWriter("runs/flickr")
step = 0
save_model = True

# Early stopping and learning rate scheduling
best_val_loss = float('inf')
counter = 0  # Early stopping patience counter
num_epochs = 5

# Lists to store metrics for analysis
train_losses = []
train_perplexities = []
val_losses = []
val_perplexities = []

# Main Training Loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_perplexity = 0.0

    # Training loop with progress bar
    for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Training", unit="batch")):
        # Extract images and captions
        images_tensor = batch['images'].to(device)
        captions_tensor = batch['captions'].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images_tensor, captions_tensor)

        # Compute loss
        loss = criterion(outputs.view(-1, vocab_size), captions_tensor.view(-1))

        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optimizer.step()

        # Track metrics
        running_loss += loss.item()
        running_perplexity += min(torch.exp(loss).item(), 1e6)  # Prevents overflow

    # Average metrics for the epoch
    avg_loss = running_loss / len(train_loader)
    avg_perplexity = running_perplexity / len(train_loader)

    # Log training metrics to TensorBoard
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Perplexity/train', avg_perplexity, epoch)

    # Append to lists for analysis
    train_losses.append(avg_loss)
    train_perplexities.append(avg_perplexity)

    # Validation loop
    model.eval()
    val_loss = 0.0
    val_perplexity = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validation", unit="batch"):
            # Extract validation images and captions
            images_tensor = batch['images'].to(device)
            captions_tensor = batch['captions'].to(device)

            # Forward pass
            outputs = model(images_tensor, captions_tensor)

            # Compute validation loss
            loss = criterion(outputs.view(-1, vocab_size), captions_tensor.view(-1))

            # Track validation metrics
            val_loss += loss.item()
            val_perplexity += min(torch.exp(loss).item(), 1e6)

    # Average validation metrics for the epoch
    avg_val_loss = val_loss / len(val_loader)
    avg_val_perplexity = val_perplexity / len(val_loader)

    # Log validation metrics to TensorBoard
    writer.add_scalar('Loss/val', avg_val_loss, epoch)
    writer.add_scalar('Perplexity/val', avg_val_perplexity, epoch)

    # Append validation metrics for analysis
    val_losses.append(avg_val_loss)
    val_perplexities.append(avg_val_perplexity)
    
    # Learning rate scheduler step
    scheduler.step(avg_val_loss)
    
    if best_val_loss - avg_val_loss > 1e-3:  # At least 0.001 improvement required
        best_val_loss = avg_val_loss
        counter = 0  # Reset patience counter
        if save_model:
            torch.save(model.state_dict(), 'best_model_encdectrans.pth')
    else:
        counter += 1

    # Print epoch summary
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {avg_loss:.4f}, Train Perplexity: {avg_perplexity:.4f} | "
          f"Val Loss: {avg_val_loss:.4f}, Val Perplexity: {avg_val_perplexity:.4f} | "
          f"Learning Rate: {current_lr:.6f}")

    # Early stopping
    if counter >= 3:
        print(f"Early stopping triggered after {epoch + 1} epochs.")
        break

# Close TensorBoard writer after training
writer.close()

plt.figure(figsize=(10, 4))

plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
# Show the plots
plt.tight_layout()
plt.show()

"""## VII. Evaluation"""
# tokenizer = BertTokenizer.from_pretrained('/scratch/user/ahphand/bert_tokenizer')
bert_model = AutoModel.from_pretrained("/scratch/user/ahphand/bert-model")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ImageCaptioningModel(embed_size, hidden_size, vocab_size, num_heads, num_layers).to(device)
# model.load_state_dict(torch.load('best_model_encdectrans.pth'))
model.eval()

# Function to calculate BLEU scores
def calculate_bleu(pred, refs):
    bleu_scores = []
    smoothing = SmoothingFunction()

    # Tokenization
    pred_tokens = pred.split()
    refs_tokens = [ref.replace('<start>', '').replace('<end>', '').split() for ref in refs]

    # Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4
    bleu1 = sentence_bleu(refs_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing.method1)
    bleu2 = sentence_bleu(refs_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing.method1)
    bleu3 = sentence_bleu(refs_tokens, pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing.method1)
    bleu4 = sentence_bleu(refs_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing.method1)

    bleu_scores.extend([bleu1, bleu2, bleu3, bleu4])

    return bleu_scores

# Function to calculate cosine similarity between predicted and ground truth captions
def calculate_cosine_similarity(pred, true):
    # Encode captions using BERT tokenizer
    pred_tokens = tokenizer(pred, return_tensors="pt", truncation=True, padding=True)
    true_tokens = tokenizer(true, return_tensors="pt", truncation=True, padding=True)

    # Get embeddings
    with torch.no_grad():
        pred_embeds = bert_model(**pred_tokens).last_hidden_state.mean(dim=1)  # Use mean pooling for sentence embedding
        true_embeds = bert_model(**true_tokens).last_hidden_state.mean(dim=1)

    # Compute cosine similarity
    cos_sim = cosine_similarity(pred_embeds.cpu(), true_embeds.cpu())
    return cos_sim[0][0]

# Evaluation
# model.eval()

# Initialize lists to store results
all_images = []
all_predicted_captions = []
all_ground_truth_captions = []
all_bleu_scores = []
all_cosine_scores = []

# Function for Greedy Decoding
def greedy_decoding(outputs):
    # Greedy decoding: select the word with the highest probability at each step
    predicted_caption = torch.argmax(outputs, dim=-1)
    return predicted_caption

# Function for Beam Search Decoding
def beam_search_decoding(model, input_ids, num_beams=3, max_length=20):
    # Use beam search for generating captions
    output = model.generate(input_ids=input_ids, num_beams=num_beams, max_length=max_length, early_stopping=True)
    return output

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing", unit="batch"):
        images_tensor = batch['images']
        captions_tensor = batch['captions']

        images_tensor, captions_tensor = images_tensor.to(device), captions_tensor.to(device)

        # Forward pass: Get the outputs from the model
        outputs = model(images_tensor, captions_tensor)

        for i in range(outputs.size(0)):  # Iterate over the batch
            # Use either greedy decoding or beam search for generating captions
            predicted_caption_greedy = greedy_decoding(outputs[i])
            predicted_caption_beam = beam_search_decoding(model, images_tensor[i])

            # Decode captions
            decoded_caption_greedy = tokenizer.decode(predicted_caption_greedy.squeeze(0).cpu().numpy(), skip_special_tokens=True)
            decoded_caption_beam = tokenizer.decode(predicted_caption_beam[0], skip_special_tokens=True)

            # Get the ground truth caption
            true_caption = captions_tensor[i].cpu().numpy()
            decoded_true_caption = tokenizer.decode(true_caption, skip_special_tokens=True)

            # Append results to the lists
            # all_predicted_captions.append(decoded_caption_greedy)  # You can switch to beam or greedy
            all_predicted_captions.append({
                'greedy': decoded_caption_greedy,
                'beam': decoded_caption_beam
            })
            all_ground_truth_captions.append(decoded_true_caption)

            # Convert image from CHW to HWC and append it
            all_images.append(images_tensor[i].cpu().numpy().transpose(1, 2, 0))

            # Calculate BLEU and Cosine Similarity scores
            # bleu_scores = calculate_bleu(decoded_caption_greedy, [decoded_true_caption])
            # cosine_sim = calculate_cosine_similarity(decoded_caption_greedy, decoded_true_caption)
            bleu_scores_greedy = calculate_bleu(decoded_caption_greedy, [decoded_true_caption])
            bleu_scores_beam = calculate_bleu(decoded_caption_beam, [decoded_true_caption])
            cosine_sim_greedy = calculate_cosine_similarity(decoded_caption_greedy, decoded_true_caption)
            cosine_sim_beam = calculate_cosine_similarity(decoded_caption_beam, decoded_true_caption)

            # all_bleu_scores.append(bleu_scores)
            # all_cosine_scores.append(cosine_sim)
            all_bleu_scores.append({
                'greedy': bleu_scores_greedy,
                'beam': bleu_scores_beam
            })
            all_cosine_scores.append({
                'greedy': cosine_sim_greedy,
                'beam': cosine_sim_beam
            })

# Calculate the average BLEU scores
average_bleu_scores = [0, 0, 0, 0]  # BLEU-1, BLEU-2, BLEU-3, BLEU-4
for bleu_scores in all_bleu_scores:
    for i in range(4):
        average_bleu_scores[i] += bleu_scores[i]

average_bleu_scores = [score / len(all_bleu_scores) for score in average_bleu_scores]

# Calculate the average Cosine Similarity score
average_cosine_similarity = np.mean(all_cosine_scores)

# Print results
print("Average BLEU Scores:")
print("Average BLEU-1:", average_bleu_scores[0])
print("Average BLEU-2:", average_bleu_scores[1])
print("Average BLEU-3:", average_bleu_scores[2])
print("Average BLEU-4:", average_bleu_scores[3])
print("Average Cosine Similarity:", average_cosine_similarity)

# Visualize the results
selected_images = all_images[0:10]
selected_prediction = all_predicted_captions[0:10]
selected_ground_truth_caption = all_ground_truth_captions[0:10]

for i in range(len(selected_images)):
    plt.imshow(selected_images[i])
    plt.axis('off')
    plt.title(f"Predicted: {selected_prediction[i]}\nTrue: {selected_ground_truth_caption[i]}")
    plt.show()

# Save the model
torch.save(model.state_dict(), 'final_model_encdectrans.pth')

# Optionally zip the results
folder_path = './results/runs'
zip_file_path = './results/runs.zip'
shutil.make_archive(zip_file_path.replace('.zip', ''), 'zip', folder_path)
print(f'Folder zipped successfully to {zip_file_path}')