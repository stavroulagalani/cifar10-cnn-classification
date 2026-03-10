import torch
import matplotlib.pyplot as plt
import numpy as np

from step1_data import testloader
from step2_model import Net

# Βάλε τις ίδιες κλάσεις που έχεις ήδη
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def imshow(img):
    # img: [3,32,32] σε range περίπου [-1,1] λόγω Normalize(0.5,0.5,0.5)
    img = img.cpu().numpy()
    img = (img * 0.5) + 0.5          # unnormalize -> [0,1]
    img = np.clip(img, 0, 1)
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    plt.imshow(img)
    plt.axis('off')

@torch.no_grad()
def show_predictions(num_images=12):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net().to(device)
    model.load_state_dict(torch.load("cnn_cifar10.pth", map_location=device))
    model.eval()

    images, labels = next(iter(testloader))
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    preds = outputs.argmax(dim=1)

    cols = 4
    rows = int(np.ceil(num_images / cols))
    plt.figure(figsize=(12, 3 * rows))

    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        imshow(images[i])
        true_label = classes[labels[i].item()]
        pred_label = classes[preds[i].item()]
        ok = "✓" if pred_label == true_label else "✗"
        plt.title(f"pred: {pred_label}\ntrue: {true_label} {ok}", fontsize=10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_predictions(num_images=12)
