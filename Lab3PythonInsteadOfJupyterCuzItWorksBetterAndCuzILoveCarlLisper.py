import os
import json
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu
import kagglehub
import wandb
from tqdm.auto import tqdm
# ==========================================
# 1. PATH CONFIGURATION & DATASET DOWNLOAD
# ==========================================

# Use kagglehub to ensure the path is correct dynamically on Ubuntu
print("Checking/Downloading dataset...")
dataset_root = kagglehub.dataset_download("nikhil7280/coco-image-caption")
print("Dataset root path:", dataset_root)

# Construct paths to pictures and annotations
train_img_dir = os.path.join(dataset_root, "train2014", "train2014")
ann_path = os.path.join(
    dataset_root,
    "annotations_trainval2014",
    "annotations",
    "captions_train2014.json"
)

# Read the JSON file
with open(ann_path, "r") as f:
    data = json.load(f)

# Associate id of image with the name of the file
image_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

# Construct the lists of images and captions
image_paths = []
captions = []

for ann in data["annotations"]:
    img_file = image_id_to_file[ann["image_id"]]
    image_paths.append(os.path.join(train_img_dir, img_file))
    captions.append(ann["caption"])

# Split: 70% train, 15% validation, 15% test
train_imgs, temp_imgs, train_caps, temp_caps = train_test_split(
    image_paths, captions, test_size=0.3, random_state=42
)
val_imgs, test_imgs, val_caps, test_caps = train_test_split(
    temp_imgs, temp_caps, test_size=0.5, random_state=42
)

print(f"Number of training samples: {len(train_imgs)}")
print(f"Number of validation samples: {len(val_imgs)}")
print(f"Number of test samples: {len(test_imgs)}")

# ==========================================
# 2. VOCABULARY & DATASET CLASSES
# ==========================================

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def tokenizer(self, text):
        return text.lower().split()

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

vocab = Vocabulary(freq_threshold=2)
vocab.build_vocabulary(train_caps)

class CocoDataset(Dataset):
    def __init__(self, image_paths, captions, vocab, transform=None):
        self.image_paths = image_paths
        self.captions = captions
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        caption = self.captions[idx]

        if self.transform:
            image = self.transform(image)

        numericalized = [self.vocab.stoi["<SOS>"]]
        numericalized += [self.vocab.stoi.get(word, self.vocab.stoi["<UNK>"]) for word in caption.lower().split()]
        numericalized.append(self.vocab.stoi["<EOS>"])
        
        return image, torch.tensor(numericalized)

def get_transform():
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def collate_fn(batch):
    images = []
    captions = []
    for img, cap in batch:
        images.append(img)
        captions.append(cap.detach().clone() if isinstance(cap, torch.Tensor) else torch.as_tensor(cap))

    images = torch.stack(images, dim=0)
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions

# Create DataLoaders
train_dataset = CocoDataset(train_imgs, train_caps, vocab, transform=get_transform())
val_dataset   = CocoDataset(val_imgs, val_caps, vocab, transform=get_transform())
test_dataset  = CocoDataset(test_imgs, test_caps, vocab, transform=get_transform())

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def decode_caption(tokens, vocab):
    """Converts a list of token IDs back into a string of words."""
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    words = []
    for token in tokens:
        word = vocab.itos.get(token, "<UNK>")
        if word == "<SOS>": continue
        if word == "<EOS>" or word == "<PAD>": break
        words.append(word)
    return " ".join(words)

# ==========================================
# 4. MODEL ARCHITECTURE
# ==========================================

class EncodeCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncodeCNN, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(weights="DEFAULT", aux_logits=True)
        self.inception.aux_logits = False
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)
        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.train_CNN
        return self.dropout(self.relu(features))

class DecodeRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecodeRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)  
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = embeddings.permute(1, 0, 2)
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        output = self.linear(hiddens)
        return output[1:]

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncodeCNN(embed_size)
        self.decoderRNN = DecodeRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image, vocab, max_length=50):
        res_caption = []
        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None # Fixed state initialization
            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states) # Fixed LSTM inference call
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                
                res_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)
                
                if vocab.itos[predicted.item()] == "<EOS>":
                    break
        return res_caption

# ==========================================
# 5. TRAINING & EVALUATION LOOP
# ==========================================

def train(model, train_loader, val_loader, vocab, config):
    save_dir = os.path.expanduser("~/D7047E_Lab3_Models")
    os.makedirs(save_dir, exist_ok=True)
    SAVE_PATH = os.path.join(save_dir, "BestModel.pth")
    print(f"Models will be saved to: {SAVE_PATH}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    best_loss = float("inf")
    logging = wandb.run is not None

    for epoch in range(config["epochs"]):
        # ==========================================
        # TRAINING PHASE
        # ==========================================
        model.train()
        total_train_loss = 0

        # 1. Wrap the train_loader in tqdm
        train_loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch+1}/{config['epochs']}] Train")

        for imgs, captions in train_loop:
            imgs, captions = imgs.to(device), captions.to(device)

            outputs = model(imgs, captions[:, :-1])   
            target = captions[:, 1:]                 

            loss = criterion(outputs.reshape(-1, outputs.shape[2]), target.reshape(-1))

            optimizer.zero_grad()
            loss.backward()  
            optimizer.step()

            total_train_loss += loss.item()
            
            # 2. Update the progress bar text with the live loss
            train_loop.set_postfix(loss=loss.item())
            
        # ==========================================
        # VALIDATION PHASE
        # ==========================================
        model.eval()
        total_val_loss = 0
        candidates = []
        references = []

        with torch.no_grad():
            # 3. Wrap the val_loader in tqdm
            val_loop = tqdm(val_loader, total=len(val_loader), desc=f"Epoch [{epoch+1}/{config['epochs']}] Val")
            
            for imgs, captions in val_loop:
                imgs, captions = imgs.to(device), captions.to(device)

                # Validation Loss
                outputs = model(imgs, captions[:, :-1])
                target = captions[:, 1:]
                val_loss = criterion(outputs.reshape(-1, outputs.shape[2]), target.reshape(-1))
                total_val_loss += val_loss.item()
                
                # Update progress bar
                val_loop.set_postfix(val_loss=val_loss.item())

                # BLEU Score Calculation (Batch iteration)
                for i in range(imgs.size(0)):
                    single_img = imgs[i].unsqueeze(0)
                    
                    # Predict
                    predicted_tokens = model.caption_image(single_img, vocab)
                    predicted_words = decode_caption(predicted_tokens, vocab).split()
                    candidates.append(predicted_words)

                    # Ground Truth
                    target_tokens = captions[i].tolist()
                    ref_words = decode_caption(target_tokens, vocab).split()
                    references.append([ref_words])

        # Calculate epoch metrics
        train_loss_avg = total_train_loss / len(train_loader)
        val_loss_avg = total_val_loss / len(val_loader)
        bleu = corpus_bleu(references, candidates)
        
        # --- SAVE ---
        if val_loss_avg < best_loss:
            best_loss = val_loss_avg
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss_avg,
                "vocab": vocab,
            }, SAVE_PATH)
            print(f"--> Saved new best model! (Val Loss: {val_loss_avg:.4f})")
            
        # --- LOGGING ---
        if logging:
            wandb.log({
                "epoch": epoch + 1,
                "Train Loss": train_loss_avg,
                "Val Loss": val_loss_avg,
                "Bleu-4": bleu
            })
            
        print(f"Epoch [{epoch+1}/{config['epochs']}] Summary -> Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f} | BLEU: {bleu:.4f}\n")

    return model

# ==========================================
# 6. EXECUTION
# ==========================================

# Initialize W&B Settings
config = {
    "learning_rate": 3e-4,
    "batch_size": 32,
    "epochs": 10,
    "embed_size": 256,
    "hidden_size": 256,
    "num_layers": 1
}

# Start Weights & Biases
wandb.init(project="Lab3", name="Test_run2", config=config)

print("Initializing model...")
model = CNNtoRNN(
    embed_size=config["embed_size"], 
    hidden_size=config["hidden_size"], 
    vocab_size=len(vocab), 
    num_layers=config["num_layers"]
)

print("Starting training...")
trained_model = train(
    model=model, 
    train_loader=train_loader, 
    val_loader=val_loader, 
    vocab=vocab, 
    config=config
)

# Finish the W&B run cleanly
wandb.finish()