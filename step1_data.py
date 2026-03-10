import torch
import torchvision
import torchvision.transforms as transforms

print("PyTorch version:", torch.__version__)
print("CUDA available?:", torch.cuda.is_available())


#Data augmentation για το TRAINING SET
train_transform = transforms.Compose([
     # 50% πιθανότητα να γίνει flip
    transforms.RandomHorizontalFlip(),   
    # τυχαίο crop με padding γύρω γύρω      
    transforms.RandomCrop(32, padding=4),  
    # μετατροπή σε tensor     
    transforms.ToTensor(),    
     # κανονικοποίηση                  
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))      
])


#Χωρίς augmentation για TEST SEΤ
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])


# Φόρτωση training set
trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=train_transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True
)


# Φόρτωση test set 
testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=test_transform
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=64,
    shuffle=False
)

print("Train samples:", len(trainset))
print("Test samples:", len(testset))
print("Classes:", trainset.classes)
