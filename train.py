import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import pandas
import GPUtil
import argparse
import tqdm

import warnings

# class imports
from wcst import WCST
from model import Transformer
# from utils import adapt_batch_for_encoder
# we do the encoder model later for now the decoder is most important

warnings.filterwarnings("ignore")

def generate_causal_mask(seq_len):
    # https://medium.com/@swarms/understanding-masking-in-pytorch-for-attention-mechanisms-e725059fd49f
    # creates a mask to prevent attention to future tokens
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.logical_not() # invert mask

def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wcst_generator = WCST(args.batch_size)

    # transofmer model construction
    model = Transformer(
        vocab_size=70, # 64 cards + 4 categories + SEP + EOS
        d_model=args.d_model,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        max_seq_length=32
    ).to(device)

    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # loss function and optimiser
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    model.train()
    model_acc = 0

    # training loop
    for step in tqdm.tqdm(range(args.n_steps)):
        # get a batch of data
        context_batch, question_batch = next(wcst_generator.gen_batch())

        # the model's task is to predict the final token of the question
        # so, the input is everything 'before' that final token

        # we'll change the rule after 2000 steps
        if (step > 0) and (step % 10000 == 0):
            wcst_generator.context_switch()

        input_data = torch.cat([
            torch.from_numpy(context_batch),
            torch.from_numpy(question_batch[:, :-1]),
        ], dim=1).long().to(device)

        # the target is just the final token (the correct category)
        target = torch.from_numpy(question_batch[:, -1]).long().to(device)

        #create mask for the input sequence
        seq_len = input_data.size(1)
        mask = generate_causal_mask(seq_len).to(device)

        #forward pass
        optimiser.zero_grad()
        # _ is not used, we ignore the attention weights
        output_logits, _ = model(input_data, mask)

        # want to only care about the pred for the verrry last token
        final_token_logits = output_logits[:, -1, :]
        loss = criterion(final_token_logits, target)

        # backward pass and update weights
        loss.backward()
        optimiser.step()

        # just a logging loop
        if (step + 1) % 100 == 0:
            preds = torch.argmax(final_token_logits, dim=-1)
            accuracy = (preds == target).float().mean().item()
            print(f"step [{step+1}/{args.n_steps}],  loss: {loss.item():.4f}, accuracy: {accuracy*100:.2f}%")
            model_acc = accuracy
            # validation check against wcst maybe ???
            # TODO: possible validation check
    
    print("Model trained")
    torch.save(model.state_dict(), f'wcst_transformer_{model_acc*100:.2f}.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Transformer for the WCST task.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--d_model', type=int, default=384, help='Dimension of the model embeddings.')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--num_blocks', type=int, default=6, help='Number of transformer blocks.')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate.')
    parser.add_argument('--n_steps', type=int, default=70000, help='Number of training steps.')
    
    args = parser.parse_args()
    train_model(args) 