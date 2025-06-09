import os
import pandas as pd
import yaml
import argparse

def preprocess_data(input_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    
    # Create FASTA file with proper formatting
    fasta_path = os.path.join(output_dir, "covid_sequences.fa")
    with open(fasta_path, "w") as f:
        for i, row in df.iterrows():
            sequence = row['sequence'].strip()
            if len(sequence) > 10:  # Ensure sequence is not empty and has reasonable length
                country_cols = [col for col in df.columns if col != 'sequence']
                country_idx = row[country_cols].argmax()
                country = country_cols[country_idx]
                f.write(f">seq_{i}|{country}\n{sequence}\n")
    
    # Create preprocessing config
    preprocess_config = [{
        "datapaths": [fasta_path],
        "output_dir": output_dir,
        "output_prefix": "covid_sequences",
        "train_split": 0.9,
        "valid_split": 0.05,
        "test_split": 0.05,
        "overwrite": True,
        "embed_reverse_complement": True,
        "tokenizer_type": "Byte-Level",
        "append_eod": True,
        "workers": 1
    }]
    
    config_path = os.path.join(output_dir, "preprocess_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(preprocess_config, f)
    
    # Run preprocessing
    os.system(f"preprocess_evo2 --config {config_path}")
    
    # Create training data config
    output_pfx = os.path.join(output_dir, "covid_sequences_byte-level")
    training_config = [
        {"dataset_prefix": f"{output_pfx}_train", "dataset_split": "train", "dataset_weight": 1.0},
        {"dataset_prefix": f"{output_pfx}_val", "dataset_split": "validation", "dataset_weight": 1.0},
        {"dataset_prefix": f"{output_pfx}_test", "dataset_split": "test", "dataset_weight": 1.0}
    ]
    
    training_config_path = os.path.join(output_dir, "training_data_config.yaml")
    with open(training_config_path, "w") as f:
        yaml.dump(training_config, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="preprocessed_data")
    args = parser.parse_args()
    preprocess_data(args.input, args.output_dir)
