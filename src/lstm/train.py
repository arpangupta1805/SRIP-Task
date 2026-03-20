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
from src.lstm.model import LSTMModel

def main(args):
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    df = load_data(args.data_path, clean=True)
    X_train, X_val, y_train, y_val = get_train_val_split(df)

    df_train = pd.DataFrame({'DATA': X_train, 'TOPIC': y_train})
    df_val = pd.DataFrame({'DATA': X_val, 'TOPIC': y_val})

    # Build Vocab
    vocab = TextVocab(max_size=50000)
    vocab.build_vocab(df_train['DATA'].tolist())
    os.makedirs(args.save_dir, exist_ok=True)
    vocab.save(os.path.join(args.save_dir, 'vocab.json'))

    # Encode Topics
    topics = df['TOPIC'].unique()
    topic2idx = {t: i for i, t in enumerate(topics)}
    with open(os.path.join(args.save_dir, 'topic2idx.json'), 'w') as f:
        json.dump(topic2idx, f)

    # Save Config
    config = {
        'model_type': 'Bidirectional LSTM',
        'max_len': args.max_len,
        'd_model': args.d_model,
        'hidden_size': args.hidden_size,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr
    }
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Datasets
    train_dataset = ParquetDataset(df_train, vocab, topic2idx, max_len=args.max_len)
    val_dataset = ParquetDataset(df_val, vocab, topic2idx, max_len=args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Model
    model = LSTMModel(
        vocab_size=len(vocab.word2idx),
        num_classes=len(topics),
        d_model=args.d_model,
        hidden_size=args.hidden_size
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Starting LSTM training...")
    best_f1 = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for i, (texts, labels) in enumerate(train_loader):
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 100 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds, average='weighted')
        print(f"Epoch {epoch+1} Evaluation - Acc: {acc:.4f}, F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'model.pt'))
            
            target_names = [k for k, v in sorted(topic2idx.items(), key=lambda item: item[1])]
            report = classification_report(val_labels, val_preds, target_names=target_names, digits=4)
            with open(os.path.join(args.save_dir, 'metrics.txt'), 'w') as fh:
                fh.write("LSTM Model Evaluation\n")
                fh.write(report + "\n")
                fh.write(f"Accuracy: {acc:.4f}\n")
                fh.write(f"Weighted F1: {f1:.4f}\n")
            print("Saved new best model and metrics!")

    print(f"Total training time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='dataset_sample_200k.parquet')
    parser.add_argument('--save_dir', type=str, default='experiments/lstm/')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=64)
    args = parser.parse_args()
    main(args)
