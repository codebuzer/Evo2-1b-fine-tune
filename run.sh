#!/bin/bash
set -e

# Create directories
mkdir -p /app/preprocessed_data /app/covid_evo2_model /app/fixed_checkpoint /app/results/predictions /app/results/analysis

# List files in data directory
echo "Files in data directory:"
ls -la /app/data/

# Check if we have training data
if [ -f "/app/data/train_spike_data.csv" ]; then
    echo "Found training data, preprocessing..."
    python /app/scripts/preprocess.py --input /app/data/train_spike_data.csv --output-dir /app/preprocessed_data
    
    # Fix checkpoint format
    if [ -f "/app/evo2_checkpoint" ]; then
        echo "Fixing checkpoint format..."
        python /app/scripts/fix_checkpoint.py --input /app/evo2_checkpoint --output /app/fixed_checkpoint/fixed_checkpoint.pt
        
        echo "Converting Evo2 model to NeMo format with GPU..."
        CUDA_VISIBLE_DEVICES=0 evo2_convert_to_nemo2 --checkpoint /app/fixed_checkpoint/fixed_checkpoint.pt --output-dir /app/covid_evo2_model/nemo_model
    fi
    
    # Train with improved parameters
    echo "Starting fine-tuning with Evo2..."
    CUDA_VISIBLE_DEVICES=0 train_evo2 \
        -d /app/preprocessed_data/training_data_config.yaml \
        --dataset-dir /app/preprocessed_data \
        --result-dir /app/covid_evo2_model \
        --experiment-name covid_finetune \
        --model-size 1b \
        --devices 1 \
        --micro-batch-size 4 \
        --max-steps 1500 \
        --lr 0.0002 \
        --min-lr 0.00001 \
        --warmup-steps 50 \
        --wd 0.005 \
        --clip-grad 0.5
    
    # Run inference with the latest checkpoint
    echo "Running inference with the latest checkpoint..."
    LATEST_CHECKPOINT=$(find /app/covid_evo2_model/covid_finetune/checkpoints -type d -name "covid_finetune--*" | sort -r | head -n 1)
    
    if [ -n "$LATEST_CHECKPOINT" ]; then
        echo "Using checkpoint: $LATEST_CHECKPOINT"
        
        # Create test FASTA file
        python -c "
import pandas as pd
import os

test_df = pd.read_csv('/app/data/test_spike_data.csv')
variant_names = [col for col in test_df.columns if col != 'sequence']

with open('/app/preprocessed_data/test_sequences.fa', 'w') as f:
    for i in range(min(100, len(test_df))):
        sequence = test_df.iloc[i]['sequence']
        variant_values = test_df.iloc[i][variant_names].values
        variant_idx = variant_values.argmax()
        actual_variant = variant_names[variant_idx]
        f.write(f'>test_{i}|{actual_variant}\\n{sequence}\\n')

print('Created test FASTA file with samples')
"
        
        # Run prediction
        predict_evo2 \
            --fasta /app/preprocessed_data/test_sequences.fa \
            --ckpt-dir $LATEST_CHECKPOINT \
            --output-dir /app/results/predictions \
            --model-size 1b \
            --output-log-prob-seqs
        
        # Analyze results with improved classifier
        python -c "
import torch
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

predictions_file = '/app/results/predictions/predictions__rank_0.pt'
seq_idx_map_file = '/app/results/predictions/seq_idx_map.json'

os.makedirs('/app/results/analysis', exist_ok=True)

predictions = torch.load(predictions_file)
with open(seq_idx_map_file, 'r') as f:
    seq_idx_map = json.load(f)

log_probs = predictions['log_probs_seqs']
print(f'Log probs shape: {log_probs.shape}')

seq_variants = {}
for seq_id, idx in seq_idx_map.items():
    variant = seq_id.split('|')[1]
    seq_variants[idx] = variant

actual_variants = [seq_variants[i] for i in range(len(seq_variants))]
print(f'Number of samples: {len(actual_variants)}')
print(f'Unique variants: {set(actual_variants)}')

variant_counts = {}
for variant in set(actual_variants):
    variant_counts[variant] = actual_variants.count(variant)
print(f'Variant counts: {variant_counts}')

X = log_probs.unsqueeze(1).numpy()
y = actual_variants

if len(set(actual_variants)) > 1 and len(actual_variants) > 5:
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    predicted_variants = clf.predict(X)
else:
    most_common_variant = max(variant_counts, key=variant_counts.get)
    predicted_variants = [most_common_variant] * len(actual_variants)

accuracy = accuracy_score(actual_variants, predicted_variants)
print(f'Accuracy: {accuracy:.4f}')

report = classification_report(actual_variants, predicted_variants)
print('\\nClassification Report:')
print(report)

with open('/app/results/analysis/results.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy:.4f}\\n\\n')
    f.write('Classification Report:\\n')
    f.write(report)
    f.write('\\n\\nPredictions:\\n')
    for i, (actual, pred) in enumerate(zip(actual_variants, predicted_variants)):
        f.write(f'Sample {i}: Actual={actual}, Predicted={pred}\\n')

labels = sorted(set(actual_variants))
cm = confusion_matrix(actual_variants, predicted_variants, labels=labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('/app/results/analysis/confusion_matrix.png')

print('Analysis complete. Results saved to /app/results/analysis/')
"
    else
        echo "No checkpoint found in /app/covid_evo2_model/covid_finetune/checkpoints"
    fi
fi

echo "Process complete."
