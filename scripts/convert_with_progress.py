# scripts/convert_with_progress.py
import os
import sys
import time
import subprocess
import argparse
from datetime import datetime, timedelta

def run_conversion_with_progress(model_path, model_size, output_dir):
    print(f"Starting conversion of {model_path} to NeMo format...")
    print(f"Output directory: {output_dir}")
    
    # Start the conversion process
    cmd = f"evo2_convert_to_nemo2 --model-path {model_path} --model-size {model_size} --output-dir {output_dir}"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    # Monitor the process
    start_time = datetime.now()
    last_update = start_time
    
    print("\nConversion progress:")
    print("=" * 50)
    
    while process.poll() is None:
        # Check if output directory exists and show its size
        if os.path.exists(output_dir):
            dir_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f)))
            dir_size_mb = dir_size / (1024 * 1024)
            
            current_time = datetime.now()
            elapsed = current_time - start_time
            
            # Update every 5 seconds
            if (current_time - last_update).total_seconds() > 5:
                print(f"Elapsed time: {elapsed}, Output size: {dir_size_mb:.2f} MB")
                last_update = current_time
        
        time.sleep(1)
    
    # Get the return code
    return_code = process.returncode
    
    # Print final status
    total_time = datetime.now() - start_time
    print("=" * 50)
    if return_code == 0:
        print(f"Conversion completed successfully in {total_time}")
    else:
        print(f"Conversion failed with return code {return_code}")
    
    return return_code

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Evo2 model to NeMo format with progress tracking")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--model-size", type=str, required=True, help="Model size (1b, 7b, 40b)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    
    args = parser.parse_args()
    
    sys.exit(run_conversion_with_progress(args.model_path, args.model_size, args.output_dir))
