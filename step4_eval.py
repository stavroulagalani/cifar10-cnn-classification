
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from step1_data import testloader
from step2_model import Net
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']



def evaluate(model, loader, criterion, device):
    """Υπολογίζει μέσο loss και accuracy πάνω σε ένα loader (π.χ. testloader)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

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


def per_class_accuracy(model, loader, device, class_names):
    """Υπολογίζει accuracy ξεχωριστά για κάθε κλάση."""
    model.eval()
    num_classes = len(class_names)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = outputs.max(1)

            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = preds[i].item()
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1

    class_acc = []
    for i in range(num_classes):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
        else:
            acc = 0.0
        class_acc.append(acc)
    return class_acc


def show_sample_predictions(model, loader, device, class_names, num_samples=10):
    """Εμφανίζει μερικές τυχαίες προβλέψεις από το test set."""
    model.eval()

    data_iter = iter(loader)
    images, labels = next(data_iter)

    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, preds = outputs.max(1)

    print(f"\nSample predictions (first {num_samples} από το batch):")
    for i in range(min(num_samples, labels.size(0))):
        true_label = class_names[labels[i].item()]
        pred_label = class_names[preds[i].item()]
        correct_str = "✓" if true_label == pred_label else "✗"
        print(f"{i:2d}. pred = {pred_label:10s} | true = {true_label:10s} {correct_str}")

def plot_confusion_matrix(model, loader, device, class_names):
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # φτιάχνουμε το μοντέλο και φορτώνουμε τα βάρη από το training
    model = Net().to(device)

    state_dict = torch.load("cnn_cifar10.pth", map_location=device)
    model.load_state_dict(state_dict)
    print("Loaded trained model from cnn_cifar10.pth")

    # loss function
    criterion = nn.CrossEntropyLoss()

    # 1) συνολικό test loss & accuracy
    test_loss, test_acc = evaluate(model, testloader, criterion, device)
    print(f"\nFinal test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_acc*100:.2f}%")

    # 2) accuracy ανά κλάση
    class_acc = per_class_accuracy(model, testloader, device, classes)
    print("\nAccuracy per class:")
    for name, acc in zip(classes, class_acc):
        print(f"  {name:10s}: {acc*100:5.2f}%")

    # 3) μερικά παραδείγματα προβλέψεων
    show_sample_predictions(model, testloader, device, classes, num_samples=10)

    plot_confusion_matrix(model, testloader, device, classes)