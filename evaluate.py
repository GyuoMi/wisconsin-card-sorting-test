import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from model import Transformer
from wcst import WCST
from train import generate_causal_mask

def run_test_set_evaluation(model, device, test_generator, n_test_steps):
    model.eval() # switch to eval mode to turn off dropout, etc.

    total_correct = 0
    total_samples = 0

    # run eval for n_test_steps

    # no need to track grads here, saves mem and computation
    with torch.no_grad():
        for _ in range(n_test_steps):
            context_batch, question_batch = next(test_generator.gen_batch())
            input_data = torch.cat([
                torch.from_numpy(context_batch),
                torch.from_numpy(question_batch[:, :-1])
            ], dim=1).long().to(device)

            target = torch.from_numpy(question_batch[:, -1]).long().to(device)

            seq_len = input_data.size(1)
            mask = generate_causal_mask(seq_len).to(device)

            # we only need the logits, can ignore the attention weights here
            output_logits, _ = model(input_data, mask)
            final_token_logits = output_logits[:, -1, :]

            preds = torch.argmax(final_token_logits, dim=-1)

            total_correct += (preds == target).sum().item()
            total_samples += target.size(0)

    accuracy = total_correct / total_samples

    print(f"Final test set accuracies: {accuracy*100:.2f}% ({total_correct} out of {total_samples} correct)")
    print("-------------------------\n")


def show_preds(model, device, test_generator):
    model.eval()
    # just a little map for our category tokens
    category_map = {64: 'C1', 65: 'C2', 66: 'C3', 67: 'C4'}

    context_batch, question_batch = next(test_generator.gen_batch())

    num_examples = 5
    input_data = torch.cat([
        torch.from_numpy(context_batch[:num_examples]),
        torch.from_numpy(question_batch[:num_examples, :-1])
    ], dim=1).long().to(device)
    target = torch.from_numpy(question_batch[:num_examples,  -1]).long().to(device)

    seq_len = input_data.size(1)
    mask = generate_causal_mask(seq_len).to(device)

    with torch.no_grad():
        output_logits,  _ = model(input_data, mask)
        preds = torch.argmax(output_logits[:, -1, :], dim=-1)

    for i in range(num_examples):
        pred_token = preds[i].item()
        actual_token = target[i].item()

        # grab actual card details from the generator
        context_cards = [test_generator.cards[int(c)] for c in context_batch[i] if c < 64]
        context_answer = category_map[int(context_batch[i][-2])]

        question_card = test_generator.cards[int(question_batch[i][0])]

        print (f"\nExample {i+1}")
        print (f"Context given: the example card {' '.join(context_cards[-1])} matched category {context_answer}")
        print (f"Question: Which category does {' '.join(question_card)} belong to?")
        print (f"Model guessed: {category_map[pred_token]}")
        print (f"Correct answer: {category_map[actual_token]}")
        print(f"Res: {'Good' if pred_token == actual_token else 'Bad'}")

def plot_card_embeddings(model, test_generator):
    # pull embeddings right out of the model
    embeddings = model.token_embedding.weight.data.cpu().numpy()
    card_embeddings = embeddings[:64] # only concern is those 64 cards since the rest of specials

    # use the generator's generated deck to label points
    card_details = test_generator.cards

    # we'll try colouring by three different features
    features_to_plot = {
        'colour': (0, test_generator.colours),
        'shape': (1, test_generator.shapes),
        'quantity': (2, test_generator.quantities)
    }

    # smoosh the 256 dims down to 2 so we can plot it
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(card_embeddings)

    for feature_name, (feature_idx, feature_map) in features_to_plot.items():
        labels = list(card_details[:, feature_idx])

        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=[feature_map.index(l) for l in labels])
        plt.title(f"2D PCA of Card Embeddings (Coloured by {feature_name.capitalize()})")
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')

        handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=scatter.cmap(scatter.norm(i))) for i, label in enumerate(feature_map)]
        plt.legend(title=feature_name.capitalize(), handles=handles)

        save_path = f"card_embeddings_pca_{feature_name}.png"
        plt.savefig(save_path)

def plot_attention_map(model, device, test_generator):
    # this should try and visualise the attention mechanism as a heatmap

    model.eval()
    context_batch, question_batch = next(test_generator.gen_batch())

    input_data = torch.cat([
        torch.from_numpy(context_batch[0:1]),
        torch.from_numpy(question_batch[0:1, :-1])
    ], dim=1).long().to(device)
    
    seq_len = input_data.size(1)
    mask = generate_causal_mask(seq_len).to(device)

    # run the model to get the outputs and populate the attention weights
    with torch.no_grad():
        _, attention_weights = model(input_data, mask)
        
    # the weights we want are from the very last block
    # shape is (batch, num_heads, seq_len, seq_len)
    last_block = model.blocks[-1]
    attention_map = last_block.attention_weights.cpu().numpy()
    
    # squeeze out the batch dimension
    attention_map = attention_map.squeeze(0)
    
    # we only care about the last token's attention (the one making the prediction)
    final_token_attention = attention_map[:, -1, :] # shape (num_heads, seq_len)
    
    # this should cover the 10 possible tokens the model sees as input
    tokens = [
        "Cat1", "Cat2", "Cat3", "Cat4", # 4 category cards
        "ExCard", # the example card
        "SEP", 
        "ExAns", # answer to the example
        "EOS", # end of seq
        "Q_Card", # card to classify
        "SEP" # final separator before prediction
    ]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(final_token_attention, cmap='viridis', aspect='auto')
    
    ax.set_yticks(np.arange(final_token_attention.shape[0]))
    ax.set_yticklabels([f"Head {i+1}" for i in range(final_token_attention.shape[0])])
    
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    
    plt.colorbar(im, ax=ax, label='Attention Weight')
    plt.title("What the model 'looks at' when making its final prediction")
    plt.ylabel("Attention Head")
    plt.xlabel("Token in Sequence")
    
    save_path = "attention_map.png"
    plt.tight_layout()
    plt.savefig(save_path)

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained WCST Transformer.')
    #  IMPORTANT prereq: these params need to match the saved model
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_blocks', type=int, default=6)
    
    parser.add_argument('--model_path', type=str, default='wcst_transformer_final_v2.pth')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_test_steps', type=int, default=100)
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Transformer(vocab_size=70, d_model=args.d_model, num_blocks=args.num_blocks, num_heads=args.num_heads, max_seq_length=32).to(device)
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Successfully loaded model from {args.model_path}")
    except FileNotFoundError:
        print(f"bad file path '{args.model_path}'")
        return
    except Exception as e:
        print(f"Something went wrong loading the model. It might be an architecture mismatch.")
        print(f"double check params d_model, num_heads, and num_blocks arguments.")
        print(f"Error details: {e}")
        return
        
    test_generator = WCST(args.batch_size)
    
    run_test_set_evaluation(model, device, test_generator, args.n_test_steps)
    show_preds(model, device, test_generator)
    plot_card_embeddings(model, test_generator)
    plot_attention_map(model, device, test_generator)

if __name__ == '__main__':
    main()  