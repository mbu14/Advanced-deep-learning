import os
import re
import gc
import hashlib
import multiprocessing as mp
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def preprocess_pandas(data, columns):
    data = data.copy()
    data['Sentence'] = data['Sentence'].str.lower()
    data['Sentence'] = data['Sentence'].str.replace(r'[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)
    data['Sentence'] = data['Sentence'].str.replace(r'((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)
    data['Sentence'] = data['Sentence'].str.replace(r'[^\w\s]', '', regex=True)
    data['Sentence'] = data['Sentence'].str.replace(r'\d', '', regex=True)

    stop_words = set(stopwords.words('english'))
    data['Sentence'] = data['Sentence'].apply(
        lambda s: ' '.join(w for w in word_tokenize(s) if w not in stop_words)
    )
    return data[columns]


_stop_words_cache: set | None = None


def _worker_init():
    global _stop_words_cache
    _stop_words_cache = set(stopwords.words('english'))


def _process_chunk(rows: list[tuple]) -> list[tuple]:
    results = []
    for idx, cls, sentence in rows:
        t = str(sentence).lower()
        t = re.sub(r'[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', t)
        t = re.sub(r'((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', t)
        t = re.sub(r'[^\w\s]', '', t)
        t = re.sub(r'\d+', '', t)
        words = word_tokenize(t)
        results.append((idx, cls, ' '.join(w for w in words if w not in _stop_words_cache)))
    return results


def fast_preprocess(filepath, columns, cache_dir="../data/cache", delimiter='\t', n_workers=None):
    os.makedirs(cache_dir, exist_ok=True)

    stat      = os.stat(filepath)
    cache_key = hashlib.md5(f"{filepath}:{stat.st_mtime}:{stat.st_size}".encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{os.path.basename(filepath)}_{cache_key}.parquet")

    if os.path.exists(cache_path):
        print(f"Cache hit — loading from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Cache miss — preprocessing {filepath} ...")
    data = pd.read_csv(filepath, delimiter=delimiter, header=None, names=['Sentence', 'Class'])
    data['index'] = data.index

    n_workers  = n_workers or min(4, mp.cpu_count())
    rows       = list(zip(data['index'], data['Class'], data['Sentence']))
    chunk_size = 10000
    chunks     = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]

    with mp.Pool(processes=n_workers, initializer=_worker_init) as pool:
        results = pool.map(_process_chunk, chunks)

    flat = [item for chunk in results for item in chunk]
    df   = pd.DataFrame(flat, columns=['index', 'Class', 'Sentence'])[columns]

    df.to_parquet(cache_path, index=False)
    print(f"Done — cached to {cache_path}")
    return df


def load_and_preprocess_hf(dataset_name='mteb/amazon_polarity', split='train', cache_dir="../data/cache", n_workers=None):
    from datasets import load_dataset

    os.makedirs(cache_dir, exist_ok=True)

    print(f"Loading {dataset_name} ({split}) ...")
    dataset = load_dataset(dataset_name, split=split)

    cache_path = os.path.join(cache_dir, f"{dataset_name.replace('/', '_')}_{split}_{dataset._fingerprint}.parquet")

    if os.path.exists(cache_path):
        print(f"Cache hit — loading preprocessed data from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Cache miss — preprocessing {len(dataset):,} rows ...")

    n_workers  = n_workers or min(4, mp.cpu_count())
    rows       = list(zip(range(len(dataset)), dataset['label'], dataset['text']))
    chunk_size = 10000
    chunks     = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]

    with mp.Pool(processes=n_workers, initializer=_worker_init) as pool:
        results = pool.map(_process_chunk, chunks)

    flat = [item for chunk in results for item in chunk]
    df   = pd.DataFrame(flat, columns=['index', 'Class', 'Sentence'])

    df.to_parquet(cache_path, index=False)
    print(f"Done — cached to {cache_path}")
    return df


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return str(self.texts[item]), self.labels[item]


def get_collate_fn(tokenizer, max_len=256):
    def collate_fn(batch):
        texts, labels = zip(*batch)
        encodings = tokenizer(
            list(texts),
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long),
        }
    return collate_fn


def get_dataloaders(data, tokenizer, test_size=0.15, val_size=0.15, batch_size=64, num_workers=4, random_state=0):
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data['Sentence'].tolist(),
        data['Class'].tolist(),
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    val_fraction = val_size / (1.0 - test_size)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts,
        train_labels,
        test_size=val_fraction,
        random_state=random_state,
        shuffle=True,
    )

    print(f"Train size: {len(train_texts)} | Val size: {len(val_texts)} | Test size: {len(test_texts)}")

    train_dataset = SentimentDataset(train_texts, train_labels)
    val_dataset   = SentimentDataset(val_texts,   val_labels)
    test_dataset  = SentimentDataset(test_texts,  test_labels)

    collate_fn = get_collate_fn(tokenizer)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=pin_memory, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=pin_memory, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=pin_memory, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            running_loss += outputs.loss.item() * labels.size(0)
            correct      += outputs.logits.argmax(dim=1).eq(labels).sum().item()
            total        += labels.size(0)

    return running_loss / total, 100 * correct / total


def train_transformer(model, train_loader, val_loader, optimizer, device, epochs=10, earlystopping=3, save_path="best_transformer.pth"):
    use_amp = device.type == 'cuda'
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None

    best_val_loss      = float('inf')
    early_stop_counter = 0

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                scaler.scale(outputs.loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                outputs.loss.backward()
                optimizer.step()

            train_loss    += outputs.loss.item() * labels.size(0)
            train_correct += outputs.logits.argmax(dim=1).eq(labels).sum().item()
            train_total   += labels.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc  = 100 * train_correct / train_total
        current_lr       = optimizer.param_groups[0]['lr']
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss      = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
            print("  --> Validation loss decreased. Model saved.")
        else:
            early_stop_counter += 1
            print(f"  --> Early stopping counter: {early_stop_counter}/{earlystopping}")
            if early_stop_counter >= earlystopping:
                print("Early stopping triggered.")
                break

    return model


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    nltk.download('punkt_tab')
    nltk.download('stopwords')

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated")
        print(f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB reserved")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading and preprocessing data...")

    columns = ['index', 'Class', 'Sentence']

    model_name = 'distilbert-base-uncased'
    tokenizer  = AutoTokenizer.from_pretrained(model_name)

    # Stage 1: train on 25K Amazon dataset
    print("\nStage 1: Training on 25K Amazon dataset")
    data_25k = fast_preprocess("../data/amazon_cells_labelled_LARGE_25K.txt", columns)
    train_loader, val_loader, test_loader = get_dataloaders(data_25k, tokenizer)

    model     = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    print(f"Trainable parameters: {count_trainable_params(model):,}")

    model = train_transformer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=10000,
        earlystopping=2,
        save_path="../data/stage1_transformer.pth",
    )

    # Stage 2: fine-tune on Amazon Polarity (HuggingFace)
    print("\nStage 2: Fine-tuning on mteb/amazon_polarity")
    data_hf = load_and_preprocess_hf('mteb/amazon_polarity', split='train')
    train_loader, val_loader, test_loader = get_dataloaders(data_hf, tokenizer)

    model.load_state_dict(torch.load("../data/stage1_transformer.pth", weights_only=True))
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    model = train_transformer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=10000,
        earlystopping=100,
        save_path="../data/stage2_transformer.pth",
    )

    # Final evaluation
    model.load_state_dict(torch.load("../data/stage2_transformer.pth", weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\nFinal Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")