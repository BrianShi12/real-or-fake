import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import RealFakeDataset

def get_transforms(split="train"):
    if split == "train":
        # Harder transforms for training
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        # Clean transforms for testing
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def model_arch():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

def prepare_data():
    df = pd.read_csv("data/train.csv")
    
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )

    trainset = RealFakeDataset(train_df, transform=get_transforms("train"), root_dir='data/')
    testset  = RealFakeDataset(test_df,  transform=get_transforms("test"),  root_dir='data/')

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader  = DataLoader(testset, batch_size=32, shuffle=False)

    return trainloader, testloader

def train_model(model, trainloader, testloader, criterion, optimizer, device, num_epochs=5):
    train_losses = []
    test_accuracies = []
    
    print("Starting Training...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Calculate stats for this epoch
        epoch_loss = running_loss / len(trainloader)
        accuracy, _, _ = evaluate_detailed(model, testloader, device)
        
        train_losses.append(epoch_loss)
        test_accuracies.append(accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Test Acc: {accuracy*100:.2f}%")
        
    return train_losses, test_accuracies

def evaluate_detailed(model, testloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            
    all_preds = np.array(all_preds).astype(int)
    all_labels = np.array(all_labels).astype(int)
    accuracy = (all_preds == all_labels).mean()
    
    return accuracy, all_labels, all_preds

def plot_results(train_losses, test_accuracies, all_labels, all_preds):
    # Setup the plot
    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Loss & Accuracy Curves
    ax[0].plot(train_losses, label='Training Loss', color='tab:orange', linewidth=2)
    ax[0].set_ylabel('Loss', color='tab:orange')
    ax[0].tick_params(axis='y', labelcolor='tab:orange')
    
    # Create a twin axis to plot accuracy on the same graph
    ax2 = ax[0].twinx()
    ax2.plot(test_accuracies, label='Test Accuracy', color='tab:blue', linewidth=2)
    ax2.set_ylabel('Accuracy', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    ax[0].set_title("Training Dynamics")
    ax[0].set_xlabel("Epochs")
    
    # Plot 2: Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[1],
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    ax[1].set_title("Confusion Matrix")
    ax[1].set_ylabel("Actual")
    ax[1].set_xlabel("Predicted")
    
    plt.tight_layout()
    plt.show() # This opens the window with the graphs

if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model_arch().to(device)
    trainloader, testloader = prepare_data()
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # RUN TRAINING
    losses, accuracies = train_model(model, trainloader, testloader, criterion, optimizer, device, num_epochs=10)
    
    # RUN FINAL EVALUATION
    final_acc, labels, preds = evaluate_detailed(model, testloader, device)
    
    # SAVE
    torch.save(model.state_dict(), "augmented_real_fake_resnet18.pth")
    print("Model saved!")
    
    # PLOT
    plot_results(losses, accuracies, labels, preds)