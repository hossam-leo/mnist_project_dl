import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import json

# Define the CNN Model
class MNIST_CNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(MNIST_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model(config):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyper-parameters
    num_epochs = config.get('num_epochs', 5)
    batch_size = config.get('batch_size', 64)
    learning_rate = config.get('learning_rate', 0.001)
    optimizer_type = config.get('optimizer', 'Adam')
    model_name = config.get('name', 'Model')

    # MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = MNIST_CNN().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Training loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        v_loss = val_loss / len(test_loader)
        v_acc = 100 * val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.2f}%')

    # Save results
    os.makedirs('results', exist_ok=True)
    result_path = f'results/{model_name}_history.json'
    with open(result_path, 'w') as f:
        json.dump(history, f)
    
    print(f"Results saved to {result_path}")
    return history

if __name__ == "__main__":
    # Experiment 1: Adam Optimizer
    print("Starting Experiment 1: Adam Optimizer")
    config1 = {
        'name': 'Model_Adam',
        'optimizer': 'Adam',
        'learning_rate': 0.001,
        'num_epochs': 5,
        'batch_size': 64
    }
    history1 = train_model(config1)

    # Experiment 2: SGD Optimizer
    print("\nStarting Experiment 2: SGD Optimizer")
    config2 = {
        'name': 'Model_SGD',
        'optimizer': 'SGD',
        'learning_rate': 0.01,
        'num_epochs': 5,
        'batch_size': 64
    }
    history2 = train_model(config2)
