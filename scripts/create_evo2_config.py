import os
import yaml
import argparse

def create_evo2_config(preprocessed_dir, output_dir):
    """Create configuration for Evo2 fine-tuning"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the preprocessed data paths
    data_prefix = os.path.join(preprocessed_dir, "covid_sequences_Byte-Level_document")
    
    # Create fine-tuning config
    finetune_config = {
        "trainer": {
            "devices": 1,
            "num_nodes": 1,
            "precision": "bf16",
            "logger": True,
            "max_epochs": 3,
            "max_steps": -1,
            "val_check_interval": 0.5,
            "limit_val_batches": 50,
            "gradient_clip_val": 1.0,
            "accumulate_grad_batches": 1,
            "log_every_n_steps": 10
        },
        "exp_manager": {
            "explicit_log_dir": output_dir,
            "exp_dir": output_dir,
            "name": "evo2_covid_finetune",
            "create_wandb_logger": False,
            "wandb_logger_kwargs": {
                "project": "covid_classification",
                "name": "evo2_finetune"
            }
        },
        "model": {
            "restore_from_path": "arcinstitute/evo2_7b",
            "micro_batch_size": 1,
            "global_batch_size": 4,
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "megatron_amp_O2": True,
            "seed": 42,
            "seq_length": 1024,
            "data": {
                "data_prefix": {
                    "train": [f"{data_prefix}_train.bin"],
                    "validation": [f"{data_prefix}_val.bin"],
                    "test": [f"{data_prefix}_test.bin"]
                }
            },
            "optim": {
                "name": "distributed_fused_adam",
                "lr": 5e-5,
                "weight_decay": 0.01,
                "betas": [0.9, 0.98],
                "sched": {
                    "name": "CosineAnnealing",
                    "warmup_steps": 100,
                    "constant_steps": 0,
                    "min_lr": 5e-6
                }
            }
        }
    }
    
    # Write config to YAML
    config_path = os.path.join(output_dir, "finetune_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(finetune_config, f)
    
    print(f"Created Evo2 fine-tuning config at {config_path}")
    
    return config_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Evo2 fine-tuning configuration")
    parser.add_argument("--preprocessed-dir", type=str, required=True, help="Directory with preprocessed data")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for fine-tuning")
    
    args = parser.parse_args()
    
    create_evo2_config(args.preprocessed_dir, args.output_dir)