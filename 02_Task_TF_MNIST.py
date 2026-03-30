import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time

BATCH_SIZE   = 128
EPOCHS_MNIST = 10
EPOCHS_SVHN  = 10
LR           = 1e-3
NUM_CLASSES  = 10


class MnistCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.feature_extractor(x))


def get_mnist_loaders():
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]) # Mean and std for MNIST dataset - found online
    train = torchvision.datasets.MNIST(
        root="./data", train=True,  download=True, transform=transform)
    test  = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform)
    return (DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True),
            DataLoader(test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True))


def get_svhn_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728),
                             (0.1980, 0.2010, 0.1970)),
    ])  # Mean and std for each channel in SVHN dataset - found online
    train = torchvision.datasets.SVHN(
        root="./data", split="train", download=True, transform=transform)
    test  = torchvision.datasets.SVHN(
        root="./data", split="test",  download=True, transform=transform)
    return (DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True),
            DataLoader(test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True))


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
        correct += outputs.argmax(1).eq(labels).sum().item()
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
        correct += outputs.argmax(1).eq(labels).sum().item()
        total   += labels.size(0)
    return running_loss / total, 100.0 * correct / total


def run_training(model, train_loader, test_loader, epochs, tag, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR) # Only optimize parameters that require gradients, this is for for transfer learning
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print(f"\n{'='*60}")
    print(f"  Training: {tag}")
    print(f"{'='*60}")
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss,  test_acc  = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        best_acc = max(best_acc, test_acc)
        print(f"Epoch [{epoch:2d}/{epochs}] | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% | "
              f"Time: {time.time()-t0:.1f}s")

    print(f"\n>>> Best Test Accuracy with {tag}: {best_acc:.2f}%")
    return best_acc


def build_svhn_transfer_model(mnist_model, device):
    svhn_model = MnistCNN(in_channels=3, num_classes=NUM_CLASSES).to(device)

    mnist_sd = mnist_model.state_dict()
    svhn_sd  = svhn_model.state_dict()

    # Block 1 has different in_channels (1 vs 3), so skip it
    skip_prefixes = (
        "feature_extractor.0.",   # conv1 block1
        "feature_extractor.1.",   # bn1   block1
        "feature_extractor.3.",   # conv2 block1
        "feature_extractor.4.",   # bn2   block1
        "classifier.",            # replace head
    )

    for name, param in mnist_sd.items():
        if any(name.startswith(p) for p in skip_prefixes):
            continue
        if name in svhn_sd and svhn_sd[name].shape == param.shape:
            svhn_sd[name] = param # overwrite random weights with trained ones

    svhn_model.load_state_dict(svhn_sd)

    # Freeze pretrained layers; keep new/replaced layers trainable
    for name, param in svhn_model.named_parameters():
        if any(name.startswith(p) for p in skip_prefixes):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # We train the first block and the classifier
    return svhn_model


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Train on MNIST
    mnist_train, mnist_test = get_mnist_loaders()
    mnist_model = MnistCNN(in_channels=1, num_classes=NUM_CLASSES).to(DEVICE)
    mnist_acc = run_training(mnist_model, mnist_train, mnist_test,
                             EPOCHS_MNIST, "MNIST", DEVICE)

    torch.save(mnist_model.state_dict(), "mnist_cnn.pth")
    print("\nMNIST model weights saved to mnist_cnn.pth")

    # Transfer to SVHN
    svhn_train, svhn_test = get_svhn_loaders()
    svhn_model = build_svhn_transfer_model(mnist_model, DEVICE)
    svhn_acc = run_training(svhn_model, svhn_train, svhn_test,
                            EPOCHS_SVHN, "SVHN (pretrained from MNIST)", DEVICE)

    # Summary
    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    print(f"  MNIST Test Accuracy : {mnist_acc:.2f}%")
    print(f"  SVHN  Test Accuracy : {svhn_acc:.2f}%")
    print("""
  The two datasets are different, grayscale vs colour, clean
  vs noisy background. Thus, the weight transfer can be limited. We tried to
  reuse the deeper convolutional blocks (the more abstract
  layers) and replace the first blocks and classifier.
  Despite the difference in domain, the pretrained CNN layer still preformed well.
""")