import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class BayesianConvNet(nn.Module):
    def __init__(self):
        super(BayesianConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class BayesianTrainer:
    def __init__(self, model, train_loader, test_loader, epochs, lr):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.lr = lr

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            for data, target in self.train_loader:
                optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)

            self.model.eval()
            test_loss = 0.0
            correct = 0
            with torch.no_grad():
                for data, target in self.test_loader:
                    output = self.model(data)
                    test_loss += F.nll_loss(output, target, reduction='sum').item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            train_loss /= len(self.train_loader.dataset)
            test_loss /= len(self.test_loader.dataset)

            print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)} ({100.0 * correct / len(self.test_loader.dataset):.2f}%)')

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_set = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('../data', train=False, transform=transform)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=True)

    model = BayesianConvNet()
    trainer = BayesianTrainer(model, train_loader, test_loader, epochs=100, lr=0.001)
    trainer.train()
