# Atharva Nayak - 002322653

import sys
import argparse
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torchvision import datasets, transforms
from train_MNIST import MyNetwork

def show_filters(weights):
    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    for i, ax in enumerate(axes.flatten()):
        if i < weights.shape[0]:
            filt = weights[i, 0].detach().cpu().numpy()
            ax.imshow(filt, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.savefig('conv1_filters.png')
    plt.close()

def apply_filters(image, filters):
    img_np = image.squeeze().cpu().numpy().astype(np.float32)
    results = []
    for i in range(filters.shape[0]):
        kernel = filters[i, 0].detach().cpu().numpy().astype(np.float32)
        filtered = cv2.filter2D(img_np, -1, kernel)
        results.append(filtered)
    return results

def show_filter_results(results):
    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    for i, ax in enumerate(axes.flatten()):
        if i < len(results):
            ax.imshow(results[i], cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.savefig('conv1_filterResults.png')
    plt.close()

def show_combined(filters, results):
    num_filters = len(results)
    fig, axes = plt.subplots(5, 4, figsize=(8, 10))
    axes = axes.flatten()
    
    for i in range(num_filters):
        left_index = 2 * i
        right_index = 2 * i + 1
        
        # Display filter
        filt = filters[i, 0].detach().cpu().numpy()
        axes[left_index].imshow(filt, cmap='gray')
        axes[left_index].set_title(f"Filter {i}")
        axes[left_index].set_xticks([])
        axes[left_index].set_yticks([])
        
        # Display filtered image
        axes[right_index].imshow(results[i], cmap='gray')
        axes[right_index].set_title(f"Output {i}")
        axes[right_index].set_xticks([])
        axes[right_index].set_yticks([])
    
    for idx in range(len(axes) - 2 * num_filters):
        axes[-(idx + 1)].axis('off')

    plt.tight_layout()
    plt.savefig('conv1_combined.png')
    plt.close()

def main(argv):
    parser = argparse.ArgumentParser(description="Examine MNIST CNN first layer filters with OpenCV")
    parser.add_argument('--model-path', required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyNetwork().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(model)

    conv1_weights = model.conv1.weight
    print(f"conv1 weight tensor shape: {conv1_weights.shape}")
    print(f"First filter weights:\n{conv1_weights[0,0]}")

    # Show the conv1 filters in a 3x4 grid
    show_filters(conv1_weights)

    # Apply conv1 filters to the first training example
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    first_img, _ = train_ds[0]
    results = apply_filters(first_img, conv1_weights)

    # Display the filtered images in a 3x4 grid
    show_filter_results(results)

    # Show a combined view: each filter next to its filtered output in a 4x5 grid
    show_combined(conv1_weights, results)

if __name__ == '__main__':
    main(sys.argv)
