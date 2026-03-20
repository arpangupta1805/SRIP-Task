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
from src.transformer.model import CustomTransformer, count_parameters

def main(args):
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    df = load_data(args.data_path, clean=True)
    X_train, X_val, y_train, y_val = get_train_val_split(df)

    # Convert to DataFrames
    df_train = pd.DataFrame({'DATA': X_train, 'TOPIC': y_train})
    df_val = pd.DataFrame({'DATA': X_val, 'TOPIC': y_val})

    # Build Vocab
    vocab = TextVocab(max_size=50000)
    vocab.build_vocab(df_train['DATA'].tolist())
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        vocab.save(os.path.join(args.save_dir, 'vocab.json'))

    # Encode Topics
    topics = df['TOPIC'].unique()
    topic2idx = {t: i for i, t in enumerate(topics)}
    if args.save_dir:
        with open(os.path.join(args.save_dir, 'topic2idx.json'), 'w') as f:
            json.dump(topic2idx, f)

    # Datasets and DataLoaders
    train_dataset = ParquetDataset(df_train, vocab, topic2idx, max_len=args.max_len)
    val_dataset = ParquetDataset(df_val, vocab, topic2idx, max_len=args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Model
    model = CustomTransformer(
        vocab_size=len(vocab.word2idx),
        num_classes=len(topics),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        max_len=args.max_len
    ).to(device)

    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,} (must be < 5,000,000,000)")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_f1 = 0.0
    start_epoch = 0
    checkpoint_path = os.path.join(args.save_dir, 'checkpoint.pt') if args.save_dir else None
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('best_f1', 0.0)
    else:
        print("Starting training from scratch...")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        for i, (texts, labels) in enumerate(train_loader):
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
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

        if args.save_dir:
            # Save latest checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'checkpoint.pt'))
            
            # Save best model
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'model.pt'))
                print("Saved new best model!")

    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")

    # Final Evaluation with best model
    if args.save_dir and os.path.exists(os.path.join(args.save_dir, 'model.pt')):
        print("Loading best model for final evaluation...")
        model.load_state_dict(torch.load(os.path.join(args.save_dir, 'model.pt'), map_location=device))
        
    model.eval()
    val_preds, val_labels_final = [], []
    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels_final.extend(labels.cpu().numpy())
    
    idx2topic = {v: k for k, v in topic2idx.items()}
    target_names = [idx2topic[i] for i in range(len(topic2idx))]
    
    report = classification_report(val_labels_final, val_preds, target_names=target_names, digits=4)
    acc = accuracy_score(val_labels_final, val_preds)
    f1_weighted = f1_score(val_labels_final, val_preds, average='weighted')
    f1_macro = f1_score(val_labels_final, val_preds, average='macro')
    
    print("\n--- Final Evaluation ---")
    print(report)
    print(f"Accuracy: {acc:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")

    if args.save_dir:
        output_file = os.path.join(args.save_dir, 'metrics.txt')
        with open(output_file, 'w') as f:
            f.write("Transformer Model Final Evaluation\n")
            f.write(report + "\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Weighted F1: {f1_weighted:.4f}\n")
            f.write(f"Macro F1: {f1_macro:.4f}\n")
        print(f"Metrics saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='final_models/')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    args = parser.parse_args()
    import pandas as pd
    main(args)
