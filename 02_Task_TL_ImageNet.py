import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time

BATCH_SIZE  = 64
EPOCHS      = 10
LR          = 1e-4
NUM_CLASSES = 10


def build_finetune_model(device):
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    return model.to(device)


def build_feature_extraction_model(device):
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)   # unfrozen by default
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        running_loss += criterion(outputs, labels).item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    return running_loss / total, 100.0 * correct / total


def run_experiment(model, train_loader, test_loader, experiment_name, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)

    print(f"\n{'='*60}")
    print(f"{experiment_name}")
    print(f"{'='*60}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss,  test_acc  = evaluate(model, test_loader, criterion, device)

        print(f"Epoch [{epoch:2d}/{EPOCHS}] | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% | "
              f"Time: {time.time()-t0:.1f}s")

    print(f"\n>>> Final Test Accuracy with {experiment_name}: {test_acc:.2f}%")
    return test_acc


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # AlexNet expects 224x224 inputs
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])  # mean and std for ImageNet - found online

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True,  download=True, transform=transform)
    test_dataset  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    # Fine-Tuning 
    finetune_model = build_finetune_model(DEVICE)
    ft_acc = run_experiment(finetune_model, train_loader, test_loader,
                            "Fine-Tuning", DEVICE)

    # Feature Extraction 
    fe_model = build_feature_extraction_model(DEVICE)
    fe_acc = run_experiment(fe_model, train_loader, test_loader,
                            "Feature Extraction", DEVICE)

   
    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    print(f"  Fine-Tuning Accuracy       : {ft_acc:.2f}%")
    print(f"  Feature Extraction Accuracy: {fe_acc:.2f}%")
    print("""
  Fine-Tuning updates all weights across the network. The model
  can adapt its low-level and high-level feature extractors to the
  CIFAR-10 dataset, which should result in good performance.

  Feature Extraction keeps the ImageNet CNN layer frozen
  and only trains the final linear layer. The CNN layer feature extractors
  were trained on a different data from CIFAR-10. Thus they may not perfectly
  generalise to CIFAR-10 objects, so performance is typically
  lower. On the other hand, feature extraction trains far fewer parameters
  and converges much faster.
""")