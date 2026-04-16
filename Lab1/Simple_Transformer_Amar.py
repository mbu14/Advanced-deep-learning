import os
import gc
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from datasets import load_dataset, Dataset as HFDataset
import wandb
from tqdm import tqdm

global_start_time = time.time()


# ── Data loading ────────────────────────────────────────────────────────────

def load_local(filepath, delimiter='\t'):
    data = pd.read_csv(filepath, delimiter=delimiter, header=None, names=['Sentence', 'Class'])
    return data[['Sentence', 'Class']]


def load_hf(dataset_name='mteb/amazon_polarity', split='train[:10%]'):
    print(f"Loading {dataset_name} ({split}) ...")
    dataset = load_dataset(dataset_name, split=split)
    df = pd.DataFrame({'Sentence': dataset['text'], 'Class': dataset['label']})
    return df


# ── Collate / DataLoaders ────────────────────────────────────────────────────

class TransformerCollate:
    def __init__(self, tokenizer, max_len=256):
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __call__(self, batch):
        texts  = [str(b['Sentence']) for b in batch]
        labels = [b['Class']         for b in batch]
        enc = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'],
            'attention_mask': enc['attention_mask'],
            'labels':         torch.tensor(labels, dtype=torch.long),
        }


def get_dataloaders(df, tokenizer, batch_size=8, test_size=0.15, val_size=0.15, random_state=0):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
    val_fraction = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(train_df, test_size=val_fraction, random_state=random_state, shuffle=True)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    collate_fn = TransformerCollate(tokenizer)
    train_loader = DataLoader(HFDataset.from_pandas(train_df), batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(HFDataset.from_pandas(val_df),   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(HFDataset.from_pandas(test_df),  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


# ── Eval ─────────────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item() * labels.size(0)
            correct    += outputs.logits.argmax(dim=1).eq(labels).sum().item()
            total      += labels.size(0)
    return total_loss / total, 100 * correct / total


# ── Train ─────────────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, test_loader, optimizer, device,
          epochs=10, early_stopping=3, save_path="best_model.pth", scheduler=None):

    use_amp = device.type == 'cuda'
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None

    best_val_loss = float('inf')
    patience      = 0

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in bar:
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

            if scheduler:
                scheduler.step()

            train_loss += outputs.loss.item() * labels.size(0)
            correct    += outputs.logits.argmax(dim=1).eq(labels).sum().item()
            total      += labels.size(0)
            bar.set_postfix(elapsed=f"{(time.time()-global_start_time)/60:.1f}min")

        tr_loss = train_loss / total
        tr_acc  = 100 * correct / total
        vl_loss, vl_acc   = evaluate(model, val_loader,  device)
        tst_loss, tst_acc = evaluate(model, test_loader, device)
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train {tr_loss:.4f}/{tr_acc:.2f}% | "
              f"Val {vl_loss:.4f}/{vl_acc:.2f}% | "
              f"Test {tst_loss:.4f}/{tst_acc:.2f}% | LR {lr:.2e}")

        wandb.log({"Epoch": epoch+1,
                   "Train Loss": tr_loss, "Train Acc": tr_acc,
                   "Val Loss": vl_loss,   "Val Acc": vl_acc,
                   "Test Loss": tst_loss, "Test Acc": tst_acc,
                   "LR": lr})

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience      = 0
            torch.save(model.state_dict(), save_path)
            print("  --> Saved best model.")
        else:
            patience += 1
            print(f"  --> No improvement ({patience}/{early_stopping})")
            if patience >= early_stopping:
                print("Early stopping.")
                break

    return model


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    wandb.init(project="Lab1", name="DistilBERT-Amazon-Polarity-LinearWarmup")
    wandb.define_metric("*", step_metric="Epoch")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'distilbert-base-uncased'
    tokenizer  = AutoTokenizer.from_pretrained(model_name)

    print(f"Using device: {device}")

    # # ── Stage 1: local 25K dataset ──
    # print("\nStage 1: Training on 25K Amazon dataset")
    # df = load_local("../data/amazon_cells_labelled_LARGE_25K.txt")
    # train_loader, val_loader, test_loader = get_dataloaders(df, tokenizer, batch_size=8)

    # model     = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    # optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    # epochs1   = 100
    # total_steps = len(train_loader) * epochs1
    # warmup_steps = int(total_steps * 0.1)
    # #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * epochs1)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=total_steps
    # )
    # print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # model = train(model, train_loader, val_loader, test_loader, optimizer, device,
    #               epochs=epochs1, early_stopping=5,
    #               save_path="../data/stage1.pth", scheduler=scheduler)

    # ── Stage 2: HuggingFace Amazon Polarity (10%) ──
    print("\nFine-tuning on mteb/amazon_polarity (100%)")
    df2 = load_hf('mteb/amazon_polarity', split='train[:100%]')
    train_loader, val_loader, test_loader = get_dataloaders(df2, tokenizer, batch_size=128)

    # model.load_state_dict(torch.load("../data/stage1.pth", weights_only=True))
    model     = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
    epochs2   = 300
    total_steps = len(train_loader) * epochs2
    warmup_steps = int(total_steps * 0.1)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * epochs1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    model = train(model, train_loader, val_loader, test_loader, optimizer, device,
                  epochs=epochs2, early_stopping=10,
                  save_path="../data/stage2.pth", scheduler=scheduler)

    # ── Final eval ──
    model.load_state_dict(torch.load("../data/stage2.pth", weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\nFinal — Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%")
    wandb.log({"Final Test Loss": test_loss, "Final Test Acc": test_acc})
    wandb.finish()