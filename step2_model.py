import torch
import torch.nn as nn
import torch.nn.functional as F

#loaders από step1
from step1_data import trainloader, testloader

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Block 1: 3x32x32 -> 32x32x32 -> pool -> 32x16x16
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # Block 2: 32x16x16 -> 64x16x16 -> pool -> 64x8x8
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Block 3: 64x8x8 -> 128x8x8 -> pool -> 128x4x4
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # 128 feature maps * 4 * 4 = 2048 features
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # Conv blocks
        x = self.pool(F.relu(self.conv1(x)))   # -> [B, 32, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))   # -> [B, 64, 8, 8]
        x = self.pool(F.relu(self.conv3(x)))   # -> [B, 128, 4, 4]

        # Flatten
        x = x.view(x.size(0), -1)              # -> [B, 2048]

        # Fully connected + dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                        # -> [B, 10]

        return x

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    
    model = Net().to(device)
    print(model)
    print("Trainable params:", count_parameters(model))

    # παίρνει 1 batch από τον trainloader
    data_iter = iter(trainloader)
    images, labels = next(data_iter)           # images: [B,3,32,32]
    images, labels = images.to(device), labels.to(device)

    # πέρασμα από το δίκτυο
    with torch.no_grad():
        outputs = model(images)                # [B,10]

    # sanity checks
    print("Batch input shape :", images.shape)
    print("Batch output shape:", outputs.shape)
    #top-1 προβλέψεις για τα 5 πρώτα δείγματα
    preds = outputs.argmax(dim=1)
    print("Preds (first 5):", preds[:5].tolist(), "| Labels:", labels[:5].tolist())
