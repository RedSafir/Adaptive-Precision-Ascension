import argparse
from typing import Dict, Any

from .config import LEVEL_FP8, LEVEL_FP16, LEVEL_TF32

def estimate_vram_mb(num_params: int, batch_size: int, seq_len_or_img_size: int, 
                     dtype_level: int = LEVEL_FP8, include_optimizer: bool = True, 
                     optimizer_type: str = 'adamw') -> Dict[str, Any]:
                     
    master_weights_bytes = num_params * 4
    
    if dtype_level == LEVEL_FP8:
        working_bytes_per_param = 1
    elif dtype_level == LEVEL_FP16:
        working_bytes_per_param = 2
    else:
        working_bytes_per_param = 4
        
    working_weights_bytes = num_params * working_bytes_per_param
    
    optimizer_states_bytes = 0
    if include_optimizer:
        if optimizer_type.lower() in ['adam', 'adamw']:
            optimizer_states_bytes = num_params * 4 * 2
        elif optimizer_type.lower() == 'sgd':
            optimizer_states_bytes = num_params * 4
            
    activations_bytes = batch_size * seq_len_or_img_size * (num_params / 1000) * 4
    
    total_bytes = master_weights_bytes + working_weights_bytes + optimizer_states_bytes + activations_bytes
    
    return {
        'master_weights_mb': master_weights_bytes / (1024 ** 2),
        'working_weights_mb': working_weights_bytes / (1024 ** 2),
        'optimizer_states_mb': optimizer_states_bytes / (1024 ** 2),
        'activations_est_mb': activations_bytes / (1024 ** 2),
        'total_mb': total_bytes / (1024 ** 2)
    }

def print_vram_report(num_params: int, batch_size: int, seq_len_or_img_size: int, 
                      dtype_level: int = LEVEL_FP8, include_optimizer: bool = True, 
                      optimizer_type: str = 'adamw'):
    
    est = estimate_vram_mb(num_params, batch_size, seq_len_or_img_size, dtype_level, include_optimizer, optimizer_type)
    
    print("-" * 50)
    print("VRAM Estimation Report")
    print("-" * 50)
    print(f"Master Weights (FP32): {est['master_weights_mb']:.2f} MB")
    print(f"Working Weights:       {est['working_weights_mb']:.2f} MB")
    print(f"Optimizer States:      {est['optimizer_states_mb']:.2f} MB")
    print(f"Activations (est):     {est['activations_est_mb']:.2f} MB")
    print("-" * 50)
    print(f"Total Estimated VRAM:  {est['total_mb']:.2f} MB")
    print("-" * 50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Estimate VRAM usage for APA models.")
    parser.add_argument('--params', type=int, required=True, help="Number of parameters")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size")
    parser.add_argument('--seq-len', type=int, default=512, help="Sequence length or image size")
    parser.add_argument('--level', type=int, default=0, choices=[0, 1, 2], help="APA Level (0=FP8, 1=FP16, 2=TF32)")
    parser.add_argument('--no-opt', action='store_true', help="Exclude optimizer states")
    parser.add_argument('--opt-type', type=str, default='adamw', help="Optimizer type (adamw, sgd)")
    
    args = parser.parse_args()
    print_vram_report(args.params, args.batch_size, args.seq_len, args.level, not args.no_opt, args.opt_type)
