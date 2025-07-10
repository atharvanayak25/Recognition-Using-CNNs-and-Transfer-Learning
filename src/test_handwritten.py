#Atharva Nayak- 002322653

import sys
import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from train_MNIST import MyNetwork

def load_and_preprocess(path):
    img = Image.open(path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    tensor = transform(img)
    tensor = 1.0 - tensor
    return transforms.Normalize((0.1307,), (0.3081,))(tensor).unsqueeze(0)

def evaluate_images(model, device, image_dir):
    filenames = sorted(os.listdir(image_dir))
    inputs, labels = [], []
    for fname in filenames:
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            label = int(os.path.splitext(fname)[0])
            tensor = load_and_preprocess(os.path.join(image_dir, fname)).to(device)
            inputs.append(tensor)
            labels.append(label)

    model.eval()
    with torch.no_grad():
        outputs = torch.cat([model(inp) for inp in inputs]).exp()

    # Print formatted results
    for i, out in enumerate(outputs):
        probs = " ".join(f"{p:.2f}" for p in out)
        pred = out.argmax().item()
        print(f"{i}: [{probs}]  Pred={pred}  True={labels[i]}")

    # Plot first 9
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for idx, ax in enumerate(axes.flatten()):
        if idx < len(inputs):
            img = inputs[idx].cpu().squeeze()
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Pred: {outputs[idx].argmax().item()}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("plot-handwritten-predictions.png")

def main(argv):
    parser = argparse.ArgumentParser(description="Test MNIST model on handwritten digits")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-dir", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyNetwork().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    evaluate_images(model, device, args.image_dir)

if __name__ == "__main__":
    main(sys.argv)
