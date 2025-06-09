import os
import pandas as pd
import numpy as np
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from bionemo.evo2.model import Evo2Model
from bionemo.evo2.tokenizer import Evo2Tokenizer

class SpikeDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=4000):
        self.df = pd.read_csv(csv_file)
        self.sequences = self.df['sequence'].values
        self.country_cols = [col for col in self.df.columns if col != 'sequence']
        self.labels = self.df[self.country_cols].values
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = np.argmax(self.labels[idx])
        
        # Tokenize sequence
        tokens = self.tokenizer.tokenize(sequence)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Convert to tensor
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_model(preprocessed_dir, output_dir, model_name, epochs=3):
    # Load country names
    with open(os.path.join(preprocessed_dir, "country_names.txt"), "r") as f:
        country_names = [line.strip() for line in f.readlines()]
    
    print(f"Loading model {model_name}...")
    
    # Initialize tokenizer and model
    try:
        # Try to use BioNeMo's Evo2 model
        tokenizer = Evo2Tokenizer.from_pretrained(model_name)
        model = Evo2Model.from_pretrained(model_name)
        
        # Add classification head
        num_labels = len(country_names)
        model.add_classification_head(num_labels=num_labels)
        
    except Exception as e:
        print(f"Error loading Evo2 model: {e}")
        print("Using a simplified model for demonstration purposes")
        
        # Create a simple model for demonstration
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            num_labels=len(country_names)
        )
    
    # Create datasets
    train_dataset = SpikeDataset(os.path.join(preprocessed_dir, "train.csv"), tokenizer)
    val_dataset = SpikeDataset(os.path.join(preprocessed_dir, "val.csv"), tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss
                
                val_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        
        print(f"Train loss: {train_loss/len(train_loader):.4f}")
        print(f"Val loss: {val_loss/len(val_loader):.4f}")
        print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    # Save model
    os.makedirs(os.path.join(output_dir, "final_model"), exist_ok=True)
    model.save_pretrained(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    
    # Save country names
    with open(os.path.join(output_dir, "final_model", "country_names.txt"), "w") as f:
        for country in country_names:
            f.write(f"{country}\n")
    
    print(f"Model saved to {os.path.join(output_dir, 'final_model')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model for COVID-19 country classification")
    parser.add_argument("--preprocessed-dir", type=str, default="preprocessed_data", help="Directory with preprocessed data")
    parser.add_argument("--output-dir", type=str, default="covid_model", help="Directory to save model")
    parser.add_argument("--model-name", type=str, default="arcinstitute/evo2_7b", help="Model name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    
    args = parser.parse_args()
    
    train_model(args.preprocessed_dir, args.output_dir, args.model_name, args.epochs)
