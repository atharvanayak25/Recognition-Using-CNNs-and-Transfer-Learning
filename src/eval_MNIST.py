#Atharva Nayak- 002322653

import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from train_MNIST import MyNetwork  

def evaluate_first_ten(model, device, loader):
    model.eval()
    imgs, labels = next(iter(loader))
    imgs, labels = imgs.to(device), labels.to(device)
    outputs = model(imgs[:10]).cpu().exp()
    for i in range(10):
        out_vals = ["{:.2f}".format(v) for v in outputs[i]]
        print(f"Image {i}: {out_vals}  Pred={outputs[i].argmax().item()}  True={labels[i].item()}")

    fig, axes = plt.subplots(3,3, figsize=(6,6))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i].cpu().squeeze(), cmap='gray')
        ax.set_title(f"Pred: {outputs[i].argmax().item()}")
        ax.axis('off')
    plt.savefig('plot-predictions.png')

def main(argv):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyNetwork().to(device)
    model.load_state_dict(torch.load('C:/Users/nayak/PRCV_Projects/project_5/Task_1_Outputs/mnist_model.pth', map_location=device))

    test_loader = DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=10, shuffle=False
    )
    evaluate_first_ten(model, device, test_loader)

if __name__ == "__main__":
    main(sys.argv)
