import argparse
import os
import sys
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import load_data, get_train_val_split

def main(args):
    print(f"Loading data from {args.data_path}")
    df = load_data(args.data_path, clean=True)
    X_train, X_val, y_train, y_val = get_train_val_split(df)
    
    # Save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save config
    config = {
        'model_type': 'TF-IDF + Logistic Regression',
        'max_features': args.max_features,
        'random_state': 42
    }
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
        
    print("Training TF-IDF Vectorizer")
    vectorizer = TfidfVectorizer(max_features=args.max_features)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    
    print("Training Logistic Regression")
    model = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    print("Evaluating")
    val_preds = model.predict(X_val_tfidf)
    report = classification_report(y_val, val_preds, digits=4)
    acc = accuracy_score(y_val, val_preds)
    f1_weighted = f1_score(y_val, val_preds, average='weighted')
    
    print("\n" + report)
    print(f"Accuracy: {acc:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")
    
    with open(os.path.join(args.save_dir, 'metrics.txt'), 'w') as f:
        f.write("TF-IDF + Logistic Regression Evaluation\n")
        f.write(report + "\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Weighted F1: {f1_weighted:.4f}\n")
        
    # Save model pipeline
    print("Saving model pipeline")
    pipeline = {
        'vectorizer': vectorizer,
        'model': model
    }
    joblib.dump(pipeline, os.path.join(args.save_dir, 'model.joblib'))
    print(f"Saved to {args.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='dataset_sample_200k.parquet')
    parser.add_argument('--save_dir', type=str, default='experiments/tfidf_lr/')
    parser.add_argument('--max_features', type=int, default=50000)
    args = parser.parse_args()
    main(args)
