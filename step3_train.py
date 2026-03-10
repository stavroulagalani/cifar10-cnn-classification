
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import numpy as np

from step1_data import trainloader, testloader
from step2_model import Net

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()  # training mode (ενεργοποιεί dropout)

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # 1)μηδενίζουμε τα gradients
        optimizer.zero_grad()

        # 2)forward pass
        outputs = model(images)          # [B,10]

        # 3)loss
        loss = criterion(outputs, labels)

        # 4)back-propagation
        loss.backward()

        # 5)ενημέρωση βαρών
        optimizer.step()

        #στατιστικά
        running_loss += loss.item() * images.size(0)

        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()   # eval mode
    running_loss = 0.0
    correct = 0
    total = 0

    # δεν χρειάζεται gradient στο evaluation
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    
    model = Net().to(device)

    # loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 20  

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, criterion, device)
        print(f"  Train  loss: {train_loss:.4f}, acc: {train_acc*100:.2f}%")

        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        print(f"  Test   loss: {test_loss:.4f}, acc: {test_acc*100:.2f}%")

        train_losses.append(train_loss)
        train_accuracies.append(train_acc * 100)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc * 100)

    # αποθήκευση εκπαιδευμένου μοντέλου
    torch.save(model.state_dict(), "cnn_cifar10.pth")

    # αποθήκευση γραφημάτων
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Test Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_curve.png")
    plt.show()

    print("\nSaved trained model to cnn_cifar10.pth")
