import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score
import json
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils import load_data, get_train_val_split
from src.transformer.dataset import TextVocab, ParquetDataset
from src.transformer.model import CustomTransformer

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    df = load_data(args.data_path, clean=True)
    # We evaluate on the validation split
    _, X_val, _, y_val = get_train_val_split(df)
    
    df_val = pd.DataFrame({'DATA': X_val, 'TOPIC': y_val})

    # Load Vocab
    vocab = TextVocab(max_size=50000)
    vocab.load(os.path.join(args.model_dir, 'vocab.json'))

    # Load Topic mappings
    with open(os.path.join(args.model_dir, 'topic2idx.json'), 'r') as f:
        topic2idx = json.load(f)
    
    idx2topic = {v: k for k, v in topic2idx.items()}

    val_dataset = ParquetDataset(df_val, vocab, topic2idx, max_len=args.max_len)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Model
    model = CustomTransformer(
        vocab_size=len(vocab.word2idx),
        num_classes=len(topic2idx),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        max_len=args.max_len
    ).to(device)

    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model.pt'), map_location=device))
    model.eval()

    print("Evaluating")
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            
    # Map back to topics for classification report
    target_names = [idx2topic[i] for i in range(len(topic2idx))]
    
    report = classification_report(val_labels, val_preds, target_names=target_names, digits=4)
    acc = accuracy_score(val_labels, val_preds)
    f1_weighted = f1_score(val_labels, val_preds, average='weighted')
    f1_macro = f1_score(val_labels, val_preds, average='macro')
    
    print("\n" + report)
    print(f"Accuracy: {acc:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")

    # Write metrics to transformer_metrics.txt
    output_file = 'transformer_metrics.txt'
    with open(output_file, 'w') as f:
        f.write("Transformer Model Evaluation on Validation Set\n")
        f.write(report + "\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Weighted F1: {f1_weighted:.4f}\n")
        f.write(f"Macro F1: {f1_macro:.4f}\n")
            
    print(f"\nMetrics saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_dir', type=str, default='final_models/')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    args = parser.parse_args()
    main(args)
