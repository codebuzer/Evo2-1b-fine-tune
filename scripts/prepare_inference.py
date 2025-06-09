import os
import pandas as pd
import argparse

def prepare_inference_data(input_csv, output_fa):
    """Prepare test data for Evo2 inference"""
    
    # Load data
    df = pd.read_csv(input_csv)
    
    # Create FASTA file
    with open(output_fa, "w") as f:
        for i, row in df.iterrows():
            f.write(f">{i}\n{row['sequence']}\n")
    
    print(f"Prepared {len(df)} sequences for inference at {output_fa}")
    
    return output_fa

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare test data for Evo2 inference")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file with sequences")
    parser.add_argument("--output", type=str, required=True, help="Output FASTA file")
    
    args = parser.parse_args()
    
    prepare_inference_data(args.input, args.output)