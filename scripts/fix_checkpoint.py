#!/usr/bin/env python3

import os
import torch
import argparse

def fix_checkpoint_format(checkpoint_path, output_path):
    """Fix the checkpoint format to include the 'module' key."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create the expected structure with 'module' key
    if 'module' not in checkpoint:
        print("Adding 'module' key to checkpoint")
        new_checkpoint = {'module': {}}
        
        # Move all model weights under the 'module' key
        for key, value in checkpoint.items():
            if isinstance(value, torch.Tensor):
                new_checkpoint['module'][key] = value
            else:
                # Keep non-tensor items at the top level
                new_checkpoint[key] = value
        
        checkpoint = new_checkpoint
    
    print(f"Saving modified checkpoint to {output_path}")
    torch.save(checkpoint, output_path)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix Evo2 checkpoint format for NeMo conversion")
    parser.add_argument("--input", type=str, required=True, help="Input checkpoint path")
    parser.add_argument("--output", type=str, required=True, help="Output checkpoint path")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fixed_path = fix_checkpoint_format(args.input, args.output)
    
    print(f"Checkpoint fixed and saved to {fixed_path}")
    print("Now you can use this checkpoint with evo2_convert_to_nemo2")


