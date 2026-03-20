import pandas as pd
from sklearn.model_selection import train_test_split
import re
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_data(filepath, clean=False):
    print(f"Loading data from {filepath}...")
    df = pd.read_parquet(filepath)
    df['DATA'] = df['DATA'].fillna('').astype(str)
    if clean:
        print("Cleaning text data...")
        df['DATA'] = df['DATA'].apply(clean_text)
    return df

def get_train_val_split(df, test_size=0.1, random_state=42):
    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        df['DATA'], df['TOPIC'], test_size=test_size, random_state=random_state, stratify=df['TOPIC']
    )
    return X_train, X_val, y_train, y_val
