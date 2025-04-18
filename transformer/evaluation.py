import torch.multiprocessing as mp
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from transformers import BertTokenizer, AutoModel
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import timm
import math
import shutil
import os
from concurrent.futures import ThreadPoolExecutor

# Set the multiprocessing start method to 'spawn'
mp.set_start_method('spawn', force=True)

# Dataset loading function
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return [(item["image_path"], item["caption"]) for item in data]

# Image transforms for training and validation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom dataset class
class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None, tokenizer=None, max_length=50):
        self.data = data
        self.transform = transform
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained('/scratch/user/ahphand/bert_tokenizer')
        self.max_length = max_length

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, caption = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        encoded_caption = self.encode_caption(caption)
        return image, encoded_caption

# DataLoader collate function
def collate_fn(batch):
    images = []
    captions = []
    tokenizer = BertTokenizer.from_pretrained('/scratch/user/ahphand/bert_tokenizer')

    for image, caption in batch:
        images.append(image)
        captions.append(caption.clone().detach())

    images = torch.stack(images, dim=0)
    captions = pad_sequence(captions, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    if torch.cuda.is_available():
        images = images.cuda(non_blocking=True)
        captions = captions.cuda(non_blocking=True)

    return {'images': images, 'captions': captions}

# Load dataset instances
train_images = load_dataset('train_dataset.json')
#val_images = load_dataset('val_dataset.json')
test_images = load_dataset('test_dataset.json')

train_dataset = CaptionDataset(train_images, transform=train_transform)
#val_dataset = CaptionDataset(val_images, transform=val_test_transform)
test_dataset = CaptionDataset(test_images, transform=val_test_transform)

# DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=False)
#val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=False)

# EfficientNet Encoder
class EfficientNetEncoder(nn.Module):
    def __init__(self, embed_size):
        super(EfficientNetEncoder, self).__init__()
        self.efficientnet = timm.create_model("tf_efficientnetv2_s", pretrained=False, features_only=True)
        weights_path = "/scratch/user/ahphand/efficientnetv2_s_weights.pth"
        state_dict = torch.load(weights_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.efficientnet.load_state_dict(state_dict, strict=False)
        for param in self.efficientnet.parameters():
            param.requires_grad = False  
        feature_dim = self.efficientnet.feature_info[-1]['num_chs']
        self.fc = nn.Linear(feature_dim, embed_size)   

    def forward(self, images):
        with torch.no_grad():
            features = self.efficientnet(images)[-1]  
        features = features.mean([2, 3])  
        return self.fc(features).unsqueeze(1) 

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

# Transformer Decoder
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
        embeddings = self.embed(captions)  
        embeddings = self.pos_encoder(embeddings)  
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(captions.size(1)).to(captions.device)
        output = self.transformer_decoder(
            embeddings,  
            features,  
            tgt_mask=tgt_mask
        )
        return self.fc_out(output)  

# Combined Image Captioning Model
class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_heads=8, num_layers=6):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EfficientNetEncoder(embed_size)
        self.decoder = TransformerDecoder(embed_size, hidden_size, vocab_size, num_heads, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)  
        outputs = self.decoder(features, captions)  
        return outputs

# Initialize model and move to device
tokenizer = BertTokenizer.from_pretrained('/scratch/user/ahphand/bert_tokenizer')
bert_model = AutoModel.from_pretrained("/scratch/user/ahphand/bert-model")
embed_size = 512
hidden_size = 512
vocab_size = len(train_dataset.tokenizer.get_vocab())
num_heads = 8
num_layers = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageCaptioningModel(embed_size, hidden_size, vocab_size, num_heads, num_layers).to(device)
model.load_state_dict(torch.load('best_model_encdectrans.pth'))
# bert_model.to(device)
model.eval()

# BLEU and Cosine Similarity calculations (as before)
def calculate_bleu(pred, refs):
    bleu_scores = []
    smoothing = SmoothingFunction()
    pred_tokens = pred.split()  # Tokenize prediction
    # Iterate through all reference captions for BLEU calculation
    refs_tokens = [ref.replace('<start>', '').replace('<end>', '').split() for ref in refs]
    bleu1 = sentence_bleu(refs_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing.method1)
    bleu2 = sentence_bleu(refs_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing.method1)
    bleu3 = sentence_bleu(refs_tokens, pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing.method1)
    bleu4 = sentence_bleu(refs_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing.method1)
    bleu_scores.extend([bleu1, bleu2, bleu3, bleu4])
    return bleu_scores


def calculate_cosine_similarity(pred, true):
    pred_tokens = tokenizer(pred, return_tensors="pt", truncation=True, padding=True)
    true_tokens = tokenizer(true, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        pred_embeds = bert_model(**pred_tokens).last_hidden_state.mean(dim=1)
        true_embeds = bert_model(**true_tokens).last_hidden_state.mean(dim=1)
    cos_sim = cosine_similarity(pred_embeds.cpu(), true_embeds.cpu())
    return cos_sim[0][0]

# Parallelized BLEU calculation
def parallel_calculate_bleu(preds, refs):
    with ThreadPoolExecutor() as executor:
        bleu_scores = list(executor.map(calculate_bleu, preds, refs))
    return bleu_scores

# Parallelized Cosine Similarity calculation
def parallel_calculate_cosine_similarity(preds, refs):
    with ThreadPoolExecutor() as executor:
        cosine_sim = list(executor.map(calculate_cosine_similarity, preds, refs))
    return cosine_sim
    
# Main evaluation loop
def main():
    refs = []
    preds = []
    for batch in tqdm(test_loader):
        images = batch['images'].to(device)
        captions = batch['captions']
        captions_input = captions[:, :-1]
        outputs = model(images, captions_input)
        
        predicted_caption_greedy = outputs.argmax(dim=-1)
        for idx in range(predicted_caption_greedy.size(0)):
            decoded_pred_caption = tokenizer.decode(predicted_caption_greedy[idx], skip_special_tokens=True)
            decoded_gt_caption = tokenizer.decode(captions[idx], skip_special_tokens=True)
            preds.append(decoded_pred_caption)
            refs.append([decoded_gt_caption])  # Add a list of references for each prediction

    #bleu_scores = [calculate_bleu(pred, refs[idx]) for idx, pred in enumerate(preds)]  # Pass single reference list
    #cosine_sim = [calculate_cosine_similarity(pred, refs[idx][0]) for idx, pred in enumerate(preds)]  # Pass single reference
    
    # Parallel computation of BLEU and Cosine Similarity
    bleu_scores = parallel_calculate_bleu(preds, refs)
    print("parallel_calculate_bleu done")
    cosine_sim = parallel_calculate_cosine_similarity(preds, refs)
    print("parallel_calculate_cosine_similarity done")

    # Compute the averages
    bleu_scores = np.array(bleu_scores)
    cosine_sim = np.array(cosine_sim)

    print("Average BLEU Scores: ", np.mean(bleu_scores, axis=0))
    print("Average Cosine Similarity: ", np.mean(cosine_sim))

if __name__ == "__main__":
    main()
