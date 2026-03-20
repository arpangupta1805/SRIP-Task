import argparse
import torch
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.transformer.model import CustomTransformer
from src.transformer.dataset import TextVocab

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    print("Loading vocab and mappings...")
    vocab = TextVocab()
    vocab.load(os.path.join(args.model_dir, 'vocab.json'))
    
    with open(os.path.join(args.model_dir, 'topic2idx.json'), 'r') as f:
        topic2idx = json.load(f)
    idx2topic = {v: k for k, v in topic2idx.items()}
    
    print("Loading model...")
    model = CustomTransformer(
        vocab_size=len(vocab.word2idx),
        num_classes=len(topic2idx),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        max_len=args.max_len
    )
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model.pt'), map_location=device))
    model.to(device)
    model.eval()
    
    print("Ready for inference. Type 'quit' to exit.")
    while True:
        text = input("\nEnter text: ")
        if text.lower() == 'quit':
            break
        
        encoded = vocab.encode(text, args.max_len)
        tensor = torch.tensor([encoded], dtype=torch.long).to(device)
        
        with torch.no_grad():
            output = model(tensor)
            _, pred = torch.max(output, 1)
        
        topic = idx2topic[pred.item()]
        print(f"Predicted Topic: {topic}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='final_models/')
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    args = parser.parse_args()
    main(args)
