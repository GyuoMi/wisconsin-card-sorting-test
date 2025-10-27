import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- CHANGE 1: Import the new v2 model ---
from model_v2 import Transformer
from wcst_gen_rule import WCST
# We don't need the train.py mask function if we import the v2 model
# but we do need it if train.py has it... let's keep it simple
# and re-define it here just in case.

def generate_causal_mask(seq_len):
    # creates a mask to prevent attention to future tokens
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.logical_not() # invert mask

def run_test_set_evaluation(model, device, test_generator, num_test_steps):
    """
    Checks the model's overall accuracy on a fresh batch of test data.
    """
    model.eval() # switch to eval mode
    
    total_correct = 0
    total_samples = 0
    
    print(f"Running a proper evaluation for {num_test_steps} steps...")
    
    with torch.no_grad():
        for _ in range(num_test_steps):
            context_batch, question_batch = next(test_generator.gen_batch())
            
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
            
    accuracy = total_correct / total_samples
    
    print(f"\n--- Test Set Results ---")
    print(f"Final Accuracy: {accuracy*100:.2f}% ({total_correct} out of {total_samples} correct)")
    print("------------------------\n")

def show_predictions_human_readable(model, device, test_generator):
    """
    Shows a few examples of what the model is thinking, in plain English.
    """
    model.eval()
    
    category_map = {64: 'C1', 65: 'C2', 66: 'C3', 67: 'C4'}
    
    print("--- Having a look at some individual predictions... ---")
    
    context_batch, question_batch = next(test_generator.gen_batch())
    
    num_to_show = 5
    input_data = torch.cat([
        torch.from_numpy(context_batch[:num_to_show]),
        torch.from_numpy(question_batch[:num_to_show, :-1])
    ], dim=1).long().to(device)
    target = torch.from_numpy(question_batch[:num_to_show, -1]).long().to(device)
    
    seq_len = input_data.size(1)
    mask = generate_causal_mask(seq_len).to(device)

    with torch.no_grad():
        output_logits, _ = model(input_data, mask)
        preds = torch.argmax(output_logits[:, -1, :], dim=-1)

    for i in range(num_to_show):
        pred_token = preds[i].item()
        actual_token = target[i].item()
        
        # We need to cast to int() because the batches can be floats
        context_cards = [test_generator.cards[int(c)] for c in context_batch[i] if c < 64]
        context_answer = category_map[int(context_batch[i][-2])]
        question_card = test_generator.cards[int(question_batch[i][0])]
        
        print(f"\n--- Example {i+1} ---")
        print(f"Context given:  The example card {' '.join(context_cards[-1])} matched category {context_answer}.")
        print(f"Question:       Which category does {' '.join(question_card)} belong to?")
        print(f"Model guessed:  {category_map[pred_token]}")
        print(f"Correct answer: {category_map[actual_token]}")
        print(f"Result:         {'Good' if pred_token == actual_token else 'Bad'}")

def plot_card_embeddings(model, test_generator):
    """
    Visualises the card embeddings by reconstructing them from the
    compositional embedding tables.
    """
    print("\n--- Analysing the card embeddings (v2)... ---")
    
    model.eval()
    
    # Get the individual embedding tables
    colour_embed = model.colour_embed.weight.data.cpu()
    shape_embed = model.shape_embed.weight.data.cpu()
    quantity_embed = model.quantity_embed.weight.data.cpu()
    
    # We need to manually reconstruct the 64 card vectors
    card_vectors = []
    for i in range(64):
        # Decode the indices from the token ID
        quant_idx = i % 4
        shape_idx = (i // 4) % 4
        colour_idx = (i // 16) % 4
        
        # The final vector is the sum of its parts
        vec = colour_embed[colour_idx] + shape_embed[shape_idx] + quantity_embed[quant_idx]
        card_vectors.append(vec)
        
    # Stack them into a single tensor for PCA
    card_embeddings = torch.stack(card_vectors).numpy()
    
    # use the generator's deck to label our points
    card_details = test_generator.cards
    
    features_to_plot = {
        'colour': (0, test_generator.colours),
        'shape': (1, test_generator.shapes),
        'quantity': (2, test_generator.quantities)
    }
    
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(card_embeddings)
    
    for feature_name, (feature_idx, feature_map) in features_to_plot.items():
        labels = list(card_details[:, feature_idx])
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=[feature_map.index(l) for l in labels], cmap='viridis')
        
        plt.title(f'2D PCA of Card Embeddings (Coloured by {feature_name.capitalize()})')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=scatter.cmap(scatter.norm(i))) for i, label in enumerate(feature_map)]
        plt.legend(title=feature_name.capitalize(), handles=handles)

        save_path = f"card_embeddings_pca_{feature_name}_v2.png"
        plt.savefig(save_path)
        print(f"Embedding plot saved to {save_path}")

def plot_attention_map(model, device, test_generator):
    """
    Shows which past tokens the model was 'looking at' to make its final prediction.
    """
    print("\n--- Visualising the attention mechanism... ---")
    
    model.eval()
    context_batch, question_batch = next(test_generator.gen_batch())
    
    input_data = torch.cat([
        torch.from_numpy(context_batch[0:1]),
        torch.from_numpy(question_batch[0:1, :-1])
    ], dim=1).long().to(device)
    
    seq_len = input_data.size(1)
    mask = generate_causal_mask(seq_len).to(device)

    with torch.no_grad():
        _, attention_weights = model(input_data, mask)
        
    last_block = model.blocks[-1]
    attention_map = last_block.attention_weights.cpu().numpy()
    
    attention_map = attention_map.squeeze(0)
    final_token_attention = attention_map[:, -1, :] # shape (num_heads, seq_len)
    
    tokens = [
        "Cat1", "Cat2", "Cat3", "Cat4", # 4 category cards
        "ExCard", # the example card
        "SEP", 
        "ExAns", # answer to the example
        "EOS", # end of seq
        "Q_Card", # card to classify
        "SEP" # final separator before prediction
    ]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(final_token_attention, cmap='viridis', aspect='auto')
    
    ax.set_yticks(np.arange(final_token_attention.shape[0]))
    ax.set_yticklabels([f"Head {i+1}" for i in range(final_token_attention.shape[0])])
    
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=70)
    
    plt.colorbar(im, ax=ax, label='Attention Weight')
    plt.title("What the model 'looks at' when making its final prediction")
    plt.ylabel("Attention Head")
    plt.xlabel("Token in Sequence")
    
    save_path = "attention_map_v2.png"
    plt.tight_layout() 
    plt.savefig(save_path)
    print(f"Attention map saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained WCST Transformer (v2).')
    
    # --- These must match train_v2.py ---
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_blocks', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1) # New argument
    
    parser.add_argument('--model_path', type=str, default='wcst_transformer_final_v2.pth', help='Path to your saved model file.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_test_steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for the data generator (for reproducible tests).')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Use the V2 Transformer and pass all args ---
    model = Transformer(
        vocab_size=70, 
        d_model=args.d_model, 
        num_blocks=args.num_blocks, 
        num_heads=args.num_heads,
        max_seq_length=32, # This must match what it was trained with
        dropout=args.dropout
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Successfully loaded model from {args.model_path}")
    except FileNotFoundError:
        print(f"Couldn't find the model file at '{args.model_path}'")
        return
    except Exception as e:
        print(f"Something went wrong loading the model. It might be an architecture mismatch.")
        print(f"Please check your args (d_model, num_heads, num_blocks, dropout).")
        print(f"Error details: {e}")
        return
        
    # --- Use the seeded generator ---
    test_generator = WCST(args.batch_size, seed=args.seed)
    
    # --- Run all analysis functions ---
    run_test_set_evaluation(model, device, test_generator, args.num_test_steps)
    show_predictions_human_readable(model, device, test_generator)
    
    plot_card_embeddings(model, test_generator) 
    
    plot_attention_map(model, device, test_generator)

if __name__ == '__main__':
    main()