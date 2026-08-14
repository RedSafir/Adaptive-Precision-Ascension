import sys
import os
import argparse
import urllib.request
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from apa import APAConfig, APAManager
from model import GPT

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--apa_preset', type=str, default='research', choices=['conservative', 'aggressive', 'research'])
    parser.add_argument('--fp8_sim', action='store_true')
    parser.add_argument('--log_file', type=str, default='apa_gpt_log.jsonl')
    parser.add_argument('--data_url', type=str, default='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')
    parser.add_argument('--block_size', type=int, default=128)
    return parser.parse_args()

def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

def main():
    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    # Model
    preset_map = {
        'conservative': APAConfig.conservative,
        'aggressive': APAConfig.aggressive,
        'research': APAConfig.research_default,
    }
    config = preset_map[args.apa_preset](device=str(device), log_file=args.log_file)
    if args.fp8_sim:
        config.fp8_simulation_mode = True
        
    model = GPT(vocab_size=vocab_size, block_size=args.block_size, config=config).to(device)
    
    # APA Manager
    apa_manager = APAManager(model, config)
    
    # Optimizer
    optimizer = torch.optim.AdamW(apa_manager.get_trainable_parameters(), lr=args.lr, weight_decay=0.01)
    
    with open(args.log_file, 'w') as f:
        f.write(json.dumps({"config": config.__dict__, "args": vars(args)}) + "\n")
        
    model.train()
    for step in tqdm(range(args.max_steps)):
        x, y = get_batch(train_data, args.batch_size, args.block_size, device)
        
        apa_manager.pre_step()
        optimizer.zero_grad(set_to_none=True)
        
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        
        if apa_manager.post_backward_sync_and_eval():
            optimizer.step()
        else:
            optimizer.zero_grad(set_to_none=True)
            
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
            with open(args.log_file, 'a') as f:
                f.write(json.dumps({"step": step, "loss": loss.item()}) + "\n")
                
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
                print(f"--- Sample at step {step} ---\n{decode(context[0].tolist())}\n-------------------------")
            model.train()

if __name__ == '__main__':
    main()
