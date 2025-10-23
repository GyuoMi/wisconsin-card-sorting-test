import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import argparse
import tqdm

import warnings

# class imports
from wcst import WCST
from model import Transformer

warnings.filterwarnings("ignore")

def generate_causal_mask(seq_len):
    # https://medium.com/@swarms/understanding-masking-in-pytorch-for-attention-mechanisms-e725059fd49f
    # creates a mask to prevent attention to future tokens
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.logical_not() # invert mask

@torch.no_grad() 
def run_validation(model, validation_generator, device, num_val_steps=50):
    # This function is unchanged, but we'll pass it the main generator
    model.eval() 
    total_correct = 0
    total_samples = 0
    
    for _ in range(num_val_steps):
        context_batch, question_batch = next(validation_generator.gen_batch())
        
        input_data = torch.cat([
            torch.from_numpy(context_batch),
            torch.from_numpy(question_batch[:, :-1])
        ], dim=1).long().to(device)
        target = torch.from_numpy(question_batch[:, -1]).long().to(device)
        
        seq_len = input_data.size(1)
        mask = generate_causal_mask(seq_len).to(device)
        
        output_logits, _ = model(input_data, mask)
        final_token_logits = output_logits[:, -1, :]
        preds = torch.argmax(final_token_logits, dim=-1)
        
        total_correct += (preds == target).sum().item()
        total_samples += target.size(0)
        
    model.train() 
    return total_correct / total_samples

def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- FIX: Create ONLY ONE generator ---
    wcst_generator = WCST(args.batch_size)
    # We no longer create a separate validation_generator

    # transofmer model construction
    model = Transformer(
        vocab_size=70, # 64 cards + 4 categories + SEP + EOS
        d_model=args.d_model,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        max_seq_length=32
    ).to(device)
    
    if args.load_model:
        try:
            model.load_state_dict(torch.load(args.load_model, map_location=device))
            print(f"Successfully loaded pre-trained model from {args.load_model}")
        except Exception as e:
            print(f"Warning: Could not load pre-trained model. Training from scratch. Error: {e}")
    else:
        print("No --load_model path given, training from scratch.")

    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # loss function and optimiser
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    model.train()
    best_val_accuracy = 0.0
    save_path = "wcst_transformer_finetuned.pth" # New save name

    # training loop
    for step in tqdm.tqdm(range(args.n_steps)):

        # we'll change the rule after x steps
        if (step > 0) and (step % 10000 == 0): # Using 10k steps
            print(f"\n---!!! CONTEXT SWITCH at step {step} !!!---") 
            wcst_generator.context_switch()
            # No need to switch the validation_generator, it's the same object

        # get a batch of data
        context_batch, question_batch = next(wcst_generator.gen_batch())

        input_data = torch.cat([
            torch.from_numpy(context_batch),
            torch.from_numpy(question_batch[:, :-1]),
        ], dim=1).long().to(device)

        target = torch.from_numpy(question_batch[:, -1]).long().to(device)

        seq_len = input_data.size(1)
        mask = generate_causal_mask(seq_len).to(device)

        #forward pass
        optimiser.zero_grad()
        output_logits, _ = model(input_data, mask)
        final_token_logits = output_logits[:, -1, :]
        loss = criterion(final_token_logits, target)

        # backward pass and update weights
        loss.backward()
        optimiser.step()

        # just a logging loop
        if (step + 1) % 500 == 0: 
            
            preds = torch.argmax(final_token_logits, dim=-1)
            train_accuracy = (preds == target).float().mean().item()
            
            # --- FIX: Pass the ONE generator to validation ---
            val_accuracy = run_validation(model, wcst_generator, device)
            
            # Now, Train Acc and Val Acc will be testing the SAME rule
            print(f"\nStep [{step+1}/{args.n_steps}], Loss: {loss.item():.4f}, Train Acc: {train_accuracy*100:.2f}%, Val Acc: {val_accuracy*100:.2f}%")
            
            if val_accuracy > best_val_accuracy:
                print(f"  New best model Saving to {save_path} (Val Acc: {val_accuracy*100:.2f}%)")
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), save_path)
    
    print(f"Model trained. Best validation accuracy: {best_val_accuracy*100:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or Fine-tune a Transformer for the WCST task.')
    # These args should match your 100% model
    parser.add_argument('--batch_size', type=int, default=128) 
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_blocks', type=int, default=6)
    
    # Use the 100% model
    parser.add_argument('--load_model', type=str, default="wcst_transformer_best_single.pth", help='Optional: Path to a pre-trained model file to load weights from.')
    
    # Fine-tuning params
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
    parser.add_argument('--n_steps', type=int, default=50000, help='Number of training steps.')
    
    args = parser.parse_args()
    train_model(args)
