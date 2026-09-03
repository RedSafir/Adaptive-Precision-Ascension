import sys
import os
import argparse
import time
import urllib.request
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))
from apa import APAConfig, APAManager
from model import GPT

def get_args():
    parser = argparse.ArgumentParser(description="Train nanoGPT on Tiny Shakespeare with APA or pure FP32 baseline.")
    parser.add_argument('--max_steps', type=int, default=1000, help="Maximum training steps")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--apa_preset', type=str, default='research', choices=['conservative', 'aggressive', 'research'], help="APA config preset")
    parser.add_argument('--fp8_sim', action='store_true', help="Use FP8 simulation mode")
    parser.add_argument('--fp32_baseline', action='store_true', help="Run pure FP32 baseline training WITHOUT APA (uses standard nn.Linear)")
    parser.add_argument('--forensic', action='store_true', help="Enable detailed forensic logging on escalation")
    parser.add_argument('--forensic_argmax', action='store_true', help="Capture flat argmax index in forensic snapshots (opt-in)")
    parser.add_argument('--log_file', type=str, default='apa_gpt_log.jsonl', help="Path to output JSONL log file")
    parser.add_argument('--data_url', type=str, default='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')
    parser.add_argument('--block_size', type=int, default=128, help="Context length")
    return parser.parse_args()

def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

def main():
    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_apa = not args.fp32_baseline
    mode_str = "APA (Adaptive Precision)" if use_apa else "Pure FP32 Baseline (No APA)"
    
    print("=" * 60)
    print(f"nanoGPT Character-Level Language Model Training")
    print(f"Mode   : {mode_str}")
    print(f"Device : {device}")
    print(f"Steps  : {args.max_steps}, Batch Size: {args.batch_size}, LR: {args.lr}")
    print("=" * 60)
    
    # Download data
    data_path = os.path.join(os.path.dirname(__file__), 'input.txt')
    if not os.path.exists(data_path):
        print(f"Downloading data from {args.data_url}...")
        urllib.request.urlretrieve(args.data_url, data_path)
    
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # Model Setup
    if use_apa:
        preset_map = {
            'conservative': APAConfig.conservative,
            'aggressive': APAConfig.aggressive,
            'research': APAConfig.research_default,
        }
        config_kwargs = {
            'device': str(device),
            'log_file': args.log_file,
            'enable_forensic_logging': args.forensic,
            'forensic_capture_argmax_index': args.forensic_argmax,
        }
        config = preset_map[args.apa_preset](**config_kwargs)
        if args.fp8_sim:
            config.fp8_simulation_mode = True
            
        model = GPT(vocab_size=vocab_size, block_size=args.block_size, config=config, use_apa=True).to(device)
        apa_manager = APAManager(model, config)
        trainable_params = apa_manager.get_trainable_parameters()
    else:
        config = None
        model = GPT(vocab_size=vocab_size, block_size=args.block_size, config=None, use_apa=False).to(device)
        apa_manager = None
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    
    with open(args.log_file, 'w', encoding='utf-8') as f:
        header_data = {
            "mode": "apa" if use_apa else "fp32_baseline",
            "args": vars(args),
            "config": config.__dict__ if config is not None else {"precision": "FP32"}
        }
        f.write(json.dumps(header_data) + "\n")
        
    start_time = time.time()
    model.train()
    
    pbar = tqdm(range(args.max_steps), desc=f"Training [{mode_str}]")
    for step in pbar:
        x, y = get_batch(train_data, args.batch_size, args.block_size, device)
        
        if apa_manager is not None:
            apa_manager.pre_step()
            
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        
        if apa_manager is not None:
            if apa_manager.post_backward_sync_and_eval():
                optimizer.step()
            else:
                optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.step()
            
        if step % 50 == 0:
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        if step % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step:4d}/{args.max_steps} ({elapsed:.1f}s): Loss = {loss.item():.4f}")
            log_entry = {
                "step": step, 
                "loss": loss.item(), 
                "elapsed_sec": round(elapsed, 2),
                "mode": "apa" if use_apa else "fp32_baseline"
            }
            with open(args.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + "\n")
                
        if step > 0 and step % 500 == 0:
            model.eval()
            with torch.no_grad():
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
                for _ in range(100):
                    logits = model(context[:, -args.block_size:])
                    logits = logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    context = torch.cat((context, idx_next), dim=1)
                print(f"\n--- Sample at step {step} ---\n{decode(context[0].tolist())}\n-------------------------\n")
            model.train()

    total_time = time.time() - start_time
    print("=" * 60)
    print(f"Training Complete! Total Time: {total_time:.2f}s ({args.max_steps/total_time:.1f} steps/s)")
    print(f"Log saved to: {args.log_file}")
    print("=" * 60)

if __name__ == '__main__':
    main()
