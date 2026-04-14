import os
import ast
import gc
import copy
import pandas as pd
import numpy as np
import kagglehub

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import wandb

# Model Definition

class LSTM_model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size, num_layers=2, dropout=0.5, bidirectional=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        num_directions = 2 if self.bidirectional else 1
        self.fc = nn.Linear(hidden_dim * num_directions, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.long()
        x = self.dropout(self.embedding(x))
        x, (h, c) = self.lstm(x)
        
        # Take the output of the last time step
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Training & Evaluation Functions

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    for sequences, labels in loader:
        sequences, labels = sequences.to(device), labels.to(device)
        outputs = model(sequences)
        
        loss = criterion(outputs, labels)
        running_loss += loss.item() * labels.size(0)
        
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
        
    return running_loss / total, 100 * correct / total

def train(model, train_loader, val_loader, criterion, optimizer, device, tag, num_epochs=10, early_stopping_patience=3, scheduler=None):
    model.to(device)
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss += loss.item() * labels.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
            
        train_loss = running_loss / total
        train_accuracy = 100 * correct / total
        
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), f'{tag}_best_model.pth')
            print(f"  --> Validation loss decreased to {val_loss:.4f}. Model saved.")
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement >= early_stopping_patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break

        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({
            "Train Loss": train_loss, 
            "Train Accuracy": train_accuracy, 
            "Val Loss": val_loss, 
            "Val Accuracy": val_accuracy, 
            "epoch": epoch + 1, 
            "LR": current_lr
        })
        
        print(f'Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | LR: {current_lr:.2e}')
        
    return best_model

# Data Preparation

def clean_description(text):
    try:
        parsed = ast.literal_eval(text)
        return ' '.join(parsed) if isinstance(parsed, list) else str(parsed)
    except Exception:
        return str(text)

def prepare_data():
    print("Downloading/Locating dataset via kagglehub...")
    path = kagglehub.dataset_download("rogate16/amazon-reviews-2018-full-dataset")
    csv_path = os.path.join(path, "amazon_reviews.csv")
    print(f"Dataset path: {csv_path}")

    sentence_list = []
    label_chunks = []

    print("Loading chunks from CSV...")
    # Load data in chunks to manage memory
    for chunk in pd.read_csv(csv_path, chunksize=100000, usecols=['description', 'rating']):
        chunk = chunk.dropna(subset=['description', 'rating'])
        sentence_list.extend(chunk['description'].astype(str).tolist())
        label_chunks.append(chunk['rating'].values.astype('int32'))
        print(f"Chunks loaded: {len(label_chunks)}, Samples so far: {len(sentence_list)}")

    sentences = sentence_list
    labels = np.concatenate(label_chunks)
    
    # Map labels to a 0-indexed range (e.g. 1-5 becomes 0-4)
    min_label = np.min(labels)
    if min_label > 0:
        labels = labels - min_label
        
    num_classes = len(np.unique(labels))
    print(f"Total samples: {len(sentences)}, Number of classes: {num_classes}")

    print("Cleaning descriptions...")
    sentences = [clean_description(s) for s in sentences]

    print("Splitting data into train and validation sets...")
    training_data_a, validation_data_a, training_labels_a, validation_labels_a = train_test_split(
        sentences,
        labels,
        test_size=0.15,
        random_state=42,
        shuffle=True
    )

    print("Tokenizing texts...")
    max_words = 20000  # Cap the vocabulary size to prevent overfitting on rare words and avoid OOM
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(training_data_a)
    
    sequences_train = tokenizer.texts_to_sequences(training_data_a)
    sequences_val   = tokenizer.texts_to_sequences(validation_data_a)
    
    vocab_size = min(len(tokenizer.word_index) + 1, max_words + 1)
    
    # Find optimal maxlen, capping at 256 for efficiency
    maxlen = min(max(len(seq) for seq in sequences_train) + 10, 256)
    print(f"Padding sequences to length: {maxlen}...")
    
    sequences_pad_train = pad_sequences(sequences_train, maxlen=maxlen, truncating='post')
    sequences_pad_val   = pad_sequences(sequences_val,   maxlen=maxlen, truncating='post')
    
    train_x = torch.from_numpy(np.array(sequences_pad_train)).type(torch.FloatTensor)
    train_y = torch.from_numpy(np.array(training_labels_a)).long()
    test_x  = torch.from_numpy(np.array(sequences_pad_val)).type(torch.FloatTensor)
    test_y  = torch.from_numpy(np.array(validation_labels_a)).long()
    
    print(f"Final Vocab size: {vocab_size}, Max sequence length: {maxlen}")
    
    return train_x, train_y, test_x, test_y, vocab_size, num_classes


if __name__ == "__main__":
    # Config & Hyperparameters
    LR = 2e-3
    BATCH_SIZE = 512  # Increased for faster training on large datasets
    NUM_EPOCHS = 300
    EARLY_STOP = 100
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Prepare dataset
    train_x, train_y, test_x, test_y, vocab_size, num_classes = prepare_data()
    
    print("Creating DataLoaders...")
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    LSTMmodel = LSTM_model(
        vocab_size=vocab_size, 
        embedding_dim=64, 
        hidden_dim=64, 
        output_size=num_classes,
        num_layers=2,
        dropout=0.6,
        bidirectional=True
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer_LSTM = optim.AdamW(LSTMmodel.parameters(), lr=LR, weight_decay=5e-2)
    
    # Scheduler to reduce LR when validation loss plateaus
    # scheduler_LSTM = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer_LSTM,
    #     mode='min',        
    #     factor=0.5,        
    #     patience=2,        
    #     threshold=1e-3,    
    #     min_lr=1e-6
    # )
    # T_max is the total number of batches (epochs * len(train_loader))
    scheduler_LSTM = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_LSTM, 
        T_max=NUM_EPOCHS, # Or NUM_EPOCHS * len(train_loader) if stepping per batch
        eta_min=1e-6
    )

    # Clean memory before starting the training loop
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Initialize Weights & Biases
    wandb.init(project="Lab1", name="LSTM_Amazon_Reviews_Large - scheduler: CosineAnnealingLR", config={
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "early_stopping": EARLY_STOP,
        "dropout": 0.5,
        "dataset": "rogate16/amazon-reviews-2018-full-dataset"
    })

    print("\n--- Starting Training ---")
    train(
        model=LSTMmodel, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        criterion=criterion, 
        optimizer=optimizer_LSTM, 
        device=device, 
        tag="LSTM_Large", 
        num_epochs=NUM_EPOCHS, 
        early_stopping_patience=EARLY_STOP, 
        scheduler=scheduler_LSTM
    )

    wandb.finish()