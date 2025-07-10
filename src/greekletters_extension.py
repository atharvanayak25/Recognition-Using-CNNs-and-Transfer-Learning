# Atharva Nayak - 002322653

import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from train_MNIST import MyNetwork

def build_greek_model(model_path, num_classes, device):

    model = MyNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    
    for p in model.parameters():
        p.requires_grad = False

    
    model.fc2 = nn.Linear(model.fc2.in_features, num_classes).to(device)
    
    
    for name, param in model.named_parameters():
        if name.startswith("conv2") or name.startswith("fc"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss/len(loader), correct/len(loader.dataset)

def main(argv):
    parser = argparse.ArgumentParser(description="Transfer-learn MNIST model for multi-class Greek letters.")
    parser.add_argument('--model-path', required=True, help="Path to the pre-trained MNIST .pth file.")
    parser.add_argument('--data-dir', required=True, help="Folder containing subfolders of Greek letters.")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs to train the new final layer.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    transform = transforms.Compose([

        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),

        transforms.Lambda(lambda x: transforms.functional.rgb_to_grayscale(x)),

        transforms.Lambda(lambda x: transforms.functional.affine(x, 0, (0,0), 36/128, 0)),
        transforms.CenterCrop((28,28)),
        transforms.Lambda(lambda x: transforms.functional.invert(x)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    num_classes = len(dataset.classes)  
    loader = DataLoader(dataset, batch_size=5, shuffle=True)


    model = build_greek_model(args.model_path, num_classes, device)
    print("Detected classes:", dataset.classes)
    print("Model:\n", model)


    optimizer = optim.Adam(model.fc2.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    losses, accs = [], []
    for epoch in range(1, args.epochs+1):
        loss, acc = train_epoch(model, loader, criterion, optimizer, device)
        losses.append(loss)
        accs.append(acc)
        print(f"Epoch {epoch}: Loss={loss:.4f}, Accuracy={acc:.2%}")


    plt.figure()
    plt.plot(range(1, args.epochs+1), losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Greek Letters Training Loss')
    plt.savefig('greek_training_loss.png')
    plt.close()

    plt.figure()
    plt.plot(range(1, args.epochs+1), accs, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Greek Letters Training Accuracy')
    plt.savefig('greek_training_acc.png')
    plt.close()

    torch.save(model.state_dict(), 'greek_model_extension.pth')
    print("Saved Greek model to greek_model_extension.pth")

if __name__=='__main__':
    main(sys.argv)
