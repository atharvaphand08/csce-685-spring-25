import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from PIL import Image

import sys
sys.path.insert(0, "$SCRATCH/nltk_local")
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from transformers import BertTokenizer, AutoModel
from torch.nn.utils.rnn import pad_sequence

import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

import torch.optim as optim
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import json
from concurrent.futures import ThreadPoolExecutor
import random
import numpy as np

file_path = '/scratch/user/ahphand/flickr8k/flickr8k/captions.txt'
flickr = pd.read_csv(file_path)

print(flickr.head())

print(flickr.info())

print(flickr['image'].nunique())

# Step 1: Get unique image names
image_names = flickr['image'].unique()

# Step 2: Split image names into training and temporary (validation + test) sets
train_image_names, temp_image_names = train_test_split(image_names, test_size=0.2, random_state=42)

# Step 3: Split the temporary image names into validation and test sets
val_image_names, test_image_names = train_test_split(temp_image_names, test_size=0.5, random_state=42)

# Step 4: Filter the original DataFrame based on the image names in each split
train_df = flickr[flickr['image'].isin(train_image_names)]
val_df = flickr[flickr['image'].isin(val_image_names)]
test_df = flickr[flickr['image'].isin(test_image_names)]

print("Length of traning set",len(train_df))
print("Length of validation set",len(val_df))
print("Length of testing set",len(test_df))

# Define a function to filter images based on DataFrame
def filter_images(df):
    filtered_images = [("/scratch/user/ahphand/flickr8k/flickr8k/Images/" +row['image'], row['caption']) for index, row in df.iterrows()]
    return filtered_images

# Filter images for training and testing sets
train_images = filter_images(train_df)
test_images = filter_images(test_df)
val_images = filter_images(val_df)

tokenizer = BertTokenizer.from_pretrained('/scratch/user/ahphand/bert_tokenizer')

class CaptionDataset(Dataset):
    def __init__(self, data, transform=None, tokenizer=tokenizer, max_length=50):
        self.data = data
        self.transform = transform
        self.max_length = max_length
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained('/scratch/user/ahphand/bert_tokenizer')

    def encode_caption(self, caption):
        encoding = self.tokenizer.encode_plus(
            caption,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        return encoding['input_ids'].squeeze(0)

    def decode_caption(self, encoded_caption):
        return self.tokenizer.decode(encoded_caption, skip_special_tokens=True)

    def convert_to_tokens(self, encoded_caption):
        return self.tokenizer.convert_ids_to_tokens(encoded_caption)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, caption = self.data[idx]

        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image: {image_path}, skipping. Error: {e}")
            return self.__getitem__((idx + 1) % len(self.data))

        encoded_caption = self.encode_caption(caption)
        return image, encoded_caption, image_path

# !pip install --upgrade torchvision


# Training set transformations: Includes augmentations
# train_transform = transforms.Compose([
#     transforms.Resize((299, 299)),               # Resize to 299x299 for InceptionV3
#     transforms.RandomHorizontalFlip(),           # Randomly flip the image
#     transforms.RandomRotation(20),  # Random rotation with options to control cropping
#     transforms.ToTensor(),                      # Convert image to tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet statistics
# ])

# Validation and test set transformations: Only resizing and normalization
# val_test_transform = transforms.Compose([
#     transforms.Resize((299, 299)),               # Resize to 299x299 for InceptionV3
#     transforms.ToTensor(),                       # Convert image to tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet statistics
# ])

# Should use Inception-specific normalization
transform = models.Inception_V3_Weights.DEFAULT.transforms()

# Create the CaptionDataset instance
train_dataset = CaptionDataset(data=train_images, transform=transform)
val_dataset = CaptionDataset(data=val_images, transform=transform)
test_dataset = CaptionDataset(data=test_images, transform=transform)

# Save Dataset (as an example, saving image paths and captions)
def save_dataset(dataset, file_path):
    data = []
    for image_path, caption in dataset.data:  # Access raw data instead of tensors
        data.append({"image_path": image_path, "caption": caption})  # Store only file paths and text captions
    with open(file_path, 'w') as f:
        json.dump(data, f)

# Save dataset
save_dataset(train_dataset, 'vanilla_train_dataset.json')
save_dataset(val_dataset, 'vanilla_val_dataset.json')
save_dataset(test_dataset, 'vanilla_test_dataset.json')

print("Saved datasets!")

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

    for item in batch:
        if len(item) == 3:
            image, caption, _ = item
        else:
            image, caption = item
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
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn,num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn,num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn,num_workers=2)

print("Created Loaders!")

# Example of iterating through the DataLoader
for batch in train_loader:
    images = batch['images']
    captions = batch['captions']
    print("Batch of images:", images.shape)
    print("Batch of captions:", captions.shape)  # Should be [batch_size, max_caption_length]
    print("Captions:", captions)
    break  # For illustration, we break after the first batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        self.inception = inception_v3(pretrained=False, aux_logits=True)
        state_dict = torch.load('/scratch/user/ahphand/inception_v3_google-0cc3c7bd.pth', map_location=device)
        self.inception.load_state_dict(state_dict)
        self.inception.aux_logits = False  # Disable after initialization
        #weights = models.Inception_V3_Weights.DEFAULT
        #self.inception = models.inception_v3(weights=weights, aux_logits=True)  # Initialize with True
        

        # Freeze parameters
        for param in self.inception.parameters():
            param.requires_grad = False

        # Replace final FC layer
        self.inception.fc = nn.Identity()  # Directly get 2048D features

        # Projection layer
        self.linear = nn.Linear(2048, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.inception(images)  # [batch_size, 2048]
        return self.bn(self.linear(features))


# 2. Decoder RNN
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions[:, :-1]))  # remove <eos>
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs

# Hyperparameters
embed_size = 256
hidden_size = 512
num_epochs = 15
learning_rate = 1e-3

# Model init
encoder = EncoderCNN(embed_size).to(device)
vocab_size = tokenizer.vocab_size
print("Vocab size", vocab_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = optim.Adam(params, lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6)

# Checkpoint directory
os.makedirs('checkpoints', exist_ok=True)

def train_one_epoch(epoch, encoder, decoder, dataloader):
    encoder.train()
    decoder.train()
    running_loss = 0.0
    running_perplexity = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")

    for batch in progress_bar:
        images = batch['images'].to(device)
        captions = batch['captions'].to(device)

        targets = captions[:, 1:]

        # Forward
        features = encoder(images)
        outputs = decoder(features, captions)

        # Slice the output to match target length
        outputs = outputs[:, :targets.size(1), :]

        loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_perplexity += min(torch.exp(loss).item(), 1e6)  # Prevents overflow
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(dataloader)
    avg_perplexity = running_perplexity / len(dataloader)
    return avg_loss, avg_perplexity

@torch.no_grad()
def validate(epoch, encoder, decoder, dataloader):
    encoder.eval()
    decoder.eval()
    running_loss = 0.0
    running_perplexity = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")

    for batch in progress_bar:
        images = batch['images'].to(device)
        captions = batch['captions'].to(device)

        targets = captions[:, 1:]

        features = encoder(images)
        outputs = decoder(features, captions)
        outputs = outputs[:, :targets.size(1), :]

        loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
        running_loss += loss.item()
        running_perplexity += min(torch.exp(loss).item(), 1e6)  # Prevents overflow
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(dataloader)
    avg_perplexity = running_perplexity / len(dataloader)
    
    scheduler.step(avg_loss)
    return avg_loss, avg_perplexity

best_val_loss = float('inf')
counter = 0

for epoch in range(num_epochs):
    train_loss, train_perplexity = train_one_epoch(epoch, encoder, decoder, train_loader)
    val_loss, val_perplexity = validate(epoch, encoder, decoder, val_loader)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {train_loss:.4f}, Train Perplexity: {train_perplexity:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.4f} | "
          f"Learning Rate: {learning_rate:.6f}")

    # print(f"\nEpoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Save best model
    if best_val_loss - val_loss > 1e-3:
        best_val_loss = val_loss
        counter = 0
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss
        }, f'checkpoints/best_model_epoch{epoch+1}.pth')
        print("Saved new best model\n")
    else:
        counter += 1
    
    # Early stopping
    if counter >= 3:
        print(f"Early stopping triggered after {epoch + 1} epochs.")
        break

# Evaluation

encoder.eval()
decoder.eval()

bert_model = AutoModel.from_pretrained("/scratch/user/ahphand/bert-model").to(device)

# Function to generate a caption greedily
def generate_caption(encoder, decoder, image, max_length=50):
    with torch.no_grad():
        feature = encoder(image.unsqueeze(0))  # Add batch dimension: [1, 3, H, W] -> [1, 256]
        inputs = feature
        caption = []

        for _ in range(max_length):
            outputs = decoder.lstm(inputs.unsqueeze(1))[0]  # [1, 1, hidden]
            outputs = decoder.linear(outputs.squeeze(1))     # [1, vocab_size]
            predicted = outputs.argmax(1)                    # [1]
            caption.append(predicted.item())
            if predicted.item() == tokenizer.sep_token_id:
                break
            inputs = decoder.embed(predicted)

    return tokenizer.decode(caption, skip_special_tokens=True)

def evaluate_cosine_batched(preds, refs, batch_size=32):
    cosine_similarities = []
    for i in range(0, len(preds), batch_size):
        pred_batch = preds[i:i+batch_size]
        ref_batch = refs[i:i+batch_size]

        pred_tokens = tokenizer(pred_batch, return_tensors="pt", padding=True, truncation=True).to(device)
        ref_tokens = tokenizer(ref_batch, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            pred_embeds = bert_model(**pred_tokens).last_hidden_state.mean(dim=1)
            ref_embeds = bert_model(**ref_tokens).last_hidden_state.mean(dim=1)

        sims = cosine_similarity(pred_embeds.cpu(), ref_embeds.cpu())
        cosine_similarities.extend(np.diag(sims))  # diagonal = self-to-self sim

    return np.array(cosine_similarities)

# BLEU and Cosine Similarity
def calculate_bleu(pred, refs):
    smoothing = SmoothingFunction().method1
    pred_tokens = pred.split()
    refs_tokens = [ref.split() for ref in refs]
    bleu1 = sentence_bleu(refs_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = sentence_bleu(refs_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = sentence_bleu(refs_tokens, pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu4 = sentence_bleu(refs_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    return [bleu1, bleu2, bleu3, bleu4]
    
def evaluate_bleu_parallel(preds, refs, num_threads=8):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        bleu_scores = list(executor.map(lambda x: calculate_bleu(*x), zip(preds, refs)))
    return np.array(bleu_scores)

# Main evaluation loop
def evaluate_model(encoder, decoder, test_loader):
    preds = []
    refs = []

    for batch in tqdm(test_loader, desc="Evaluating"):
        images = batch['images'].to(device)
        gt_captions = batch['captions']

        for i in range(images.size(0)):
            image = images[i]
            caption_tensor = gt_captions[i]
            gt_caption = tokenizer.decode(caption_tensor, skip_special_tokens=True)

            predicted_caption = generate_caption(encoder, decoder, image)
            preds.append(predicted_caption)
            refs.append([gt_caption])  # reference is a list of strings
    
    # Parallel BLEU
    bleu_scores = evaluate_bleu_parallel(preds, refs)
    
    print("\n=== Evaluation Results ===")
    print("Average BLEU-1: ", np.mean(bleu_scores[:, 0]))
    print("Average BLEU-2: ", np.mean(bleu_scores[:, 1]))
    print("Average BLEU-3: ", np.mean(bleu_scores[:, 2]))
    print("Average BLEU-4: ", np.mean(bleu_scores[:, 3]))
    
    # Batched Cosine Similarity
    cosine_similarities = evaluate_cosine_batched(preds, [r[0] for r in refs])
    
    print("Average Cosine Similarity: ", np.mean(cosine_similarities))

# Run evaluation
evaluate_model(encoder, decoder, test_loader)

# Sample 20 indices randomly from test_dataset
num_samples = 20
random_indices = random.sample(range(len(test_dataset)), num_samples)

images_to_display = []
ground_truth_captions = []
predicted_captions = []

for idx in random_indices:
    image, encoded_caption, image_path = test_dataset[idx]
    # image_path = "Unknown (path not available from dataset)"
    image = image.unsqueeze(0).to(device)  # [1, 3, H, W]
    encoded_caption = encoded_caption.to(device)
    
    # caption_input = encoded_caption[:-1].unsqueeze(0).to(device)  # Exclude [SEP] token

    # Predict caption
    with torch.no_grad():
        features = encoder(image)
    
    # Start decoding with [CLS] token
    generated_ids = [tokenizer.cls_token_id]
    max_len = 50
    
    for _ in range(max_len):
        current_input = torch.tensor(generated_ids).unsqueeze(0).to(device)  # [1, seq_len]
        with torch.no_grad():
            outputs = decoder(features, current_input)  # [1, seq_len, vocab_size]
        next_token_id = outputs[0, -1].argmax().item()
        
        # Stop if [SEP] token is generated
        if next_token_id == tokenizer.sep_token_id:
            break
        generated_ids.append(next_token_id)

    # Decode predictions
    predicted_caption = tokenizer.decode(generated_ids[1:], skip_special_tokens=True)
    ground_truth_caption = tokenizer.decode(encoded_caption.tolist(), skip_special_tokens=True)

    images_to_display.append(image_path)
    ground_truth_captions.append(ground_truth_caption)
    predicted_captions.append(predicted_caption)

# Print results
for idx in range(num_samples):
    print(f"Image Path: {images_to_display[idx]}")
    print(f"Ground Truth Caption: {ground_truth_captions[idx]}")
    print(f"Predicted Caption: {predicted_captions[idx]}")
    print("-" * 50)
