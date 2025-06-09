import os
import torch
import pandas as pd
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class SpikeDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=1024):
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
        
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def train_model(preprocessed_dir, output_dir, model_name, epochs=3):
    # Load country names
    with open(os.path.join(preprocessed_dir, "country_names.txt"), "r") as f:
        country_names = [line.strip() for line in f.readlines()]
    
    print(f"Loading model {model_name}...")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(country_names),
        load_in_8bit=True,
        device_map="auto"
    )
    
    # Apply LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value", "dense"]
    )
    model = get_peft_model(model, peft_config)
    
    # Create datasets
    train_dataset = SpikeDataset(os.path.join(preprocessed_dir, "train.csv"), tokenizer)
    val_dataset = SpikeDataset(os.path.join(preprocessed_dir, "val.csv"), tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        per_device_eval_batch_size=1,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True,
        gradient_checkpointing=True
    )
    
    # Create trainer and train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    # Save model and tokenizer
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
    parser.add_argument("--model-name", type=str, default="arcinstitute/evo2_7b", help="Hugging Face model name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    
    args = parser.parse_args()
    
    train_model(args.preprocessed_dir, args.output_dir, args.model_name, args.epochs)
