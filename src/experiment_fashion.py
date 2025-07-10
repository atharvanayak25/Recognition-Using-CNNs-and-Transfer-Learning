# Atharva Nayak - 002322653

import time, csv, argparse
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SimpleCNN(nn.Module):
    def __init__(self, filters, dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(1, filters, 5)
        self.pool  = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(filters, filters*2, 5)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(filters*2*4*4, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(self.pool(self.conv1(x)))
        x = torch.relu(self.pool(self.dropout(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return torch.log_softmax(self.fc2(x), dim=1)

def run(params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train = datasets.FashionMNIST('./data', train=True, download=True, transform=tf)
    test  = datasets.FashionMNIST('./data', train=False, download=True, transform=tf)
    tr_loader = DataLoader(train, batch_size=params['batch_size'], shuffle=True)
    te_loader = DataLoader(test, batch_size=256, shuffle=False)

    model = SimpleCNN(params['num_filters'], params['dropout']).to(device)
    opt = optim.Adam(model.parameters())
    crit = nn.NLLLoss()

    start = time.time()
    for _ in range(params['epochs']):
        model.train()
        for imgs, lbls in tr_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt.zero_grad()
            loss = crit(model(imgs), lbls)
            loss.backward()
            opt.step()
    train_time = time.time() - start

    correct = 0
    model.eval()
    with torch.no_grad():
        for imgs, lbls in te_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            correct += (model(imgs).argmax(1)==lbls).sum().item()
    accuracy = correct / len(test)

    row = params.copy()
    row.update({'accuracy': accuracy, 'train_time': train_time})
    return row

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='fashion_experiment.csv')
    args = parser.parse_args()

    grid = []
    for nf in [10,20,40]:
        for dr in [0.1,0.25,0.5]:
            for bs in [32,64,128]:
                for ep in [5,10]:
                    grid.append({'num_filters':nf, 'dropout':dr, 'batch_size':bs, 'epochs':ep})

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(grid[0].keys())+['accuracy','train_time'])
        writer.writeheader()
        for params in grid:
            print("Running:", params)
            writer.writerow(run(params))

    print("Results saved to", args.output)

if __name__=='__main__':
    main()
