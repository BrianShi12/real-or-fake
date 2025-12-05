import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import torchvision
from dataset import RealFakeDataset
from custom_resnet import CustomResNet
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_transforms(split="train"):
    """Get transforms with augmentation for training, clean for testing"""
    if split == "train":
        # Data augmentation for training
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomRotation(15),
            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        # No augmentation for testing
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def model():
    """Initialize custom ResNet trained from scratch"""
    model = CustomResNet()
    print(model)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(f"{name}: kernel_size={module.kernel_size}, stride={module.stride}")
    return model

def prepare_data():
    """Load FULL dataset (no sampling) with augmentation"""
    df = pd.read_csv("data/train.csv")
    print(f"Loaded {len(df)} total images")
    
    # Remove the sampling line - use full dataset
    # df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)  # REMOVED
    
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    print(f"Training set: {len(train_df)} images")
    print(f"Test set: {len(test_df)} images")

    # Use different transforms for train vs test
    train_transform = get_transforms("train")
    test_transform = get_transforms("test")
    
    trainset = RealFakeDataset(train_df, transform=train_transform, root_dir='data/')
    testset = RealFakeDataset(test_df, transform=test_transform, root_dir='data/')

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)
    return trainloader, testloader

def evaluate(model, testloader, device):
    """Evaluate model and return metrics"""
    model.eval() 
    print("Evaluating...")
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    batch_count = 0
    with torch.no_grad(): 
        for images, labels in testloader:
            batch_count += 1
            if batch_count % 50 == 0:
                print(f"  Eval batch {batch_count}/{len(testloader)}...")
            
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = probs > 0.5
            
            # Store predictions and labels
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.int().squeeze().cpu().numpy())
            all_probs.extend(probs.squeeze().cpu().numpy())
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_probs)
    
    # Get precision, recall, f1 from classification report
    report = classification_report(all_labels, all_preds, 
                                  target_names=['Real', 'Fake'],
                                  digits=4, output_dict=True)
    
    # Print metrics in your requested format
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"✓ {accuracy*100:.2f}% accuracy (overall prediction correctness)")
    print(f"✓ {report['Fake']['precision']*100:.2f}% precision (minimal false positives)")
    print(f"✓ {report['Fake']['recall']*100:.2f}% recall (caught most fake images)")
    print(f"✓ {report['Fake']['f1-score']*100:.2f}% F1-Score (balanced recall and precision)")
    print(f"✓ {auroc*100:.2f}% AUROC (separation between real and fake categories)")
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, 
                                target_names=['Real', 'Fake'],
                                digits=4))
    print("="*60 + "\n")
    
    return accuracy, all_labels, all_preds

def plot_results(train_losses, test_accuracies, all_labels, all_preds):
    """Plot training curves and confusion matrix"""
    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Loss & Accuracy Curves
    epochs = range(1, len(train_losses) + 1)
    
    ax[0].plot(epochs, train_losses, label='Training Loss', color='tab:orange', linewidth=2, marker='o')
    ax[0].set_ylabel('Loss', color='tab:orange', fontsize=12)
    ax[0].tick_params(axis='y', labelcolor='tab:orange')
    ax[0].set_xlabel('Epochs', fontsize=12)
    ax[0].grid(True, alpha=0.3)
    
    # Twin axis for accuracy
    ax2 = ax[0].twinx()
    ax2.plot(epochs, test_accuracies, label='Test Accuracy', color='tab:blue', linewidth=2, marker='s')
    ax2.set_ylabel('Accuracy', color='tab:blue', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    ax[0].set_title("Training Dynamics (Custom ResNet from Scratch)", fontsize=14, fontweight='bold')
    
    # Add legends
    lines1, labels1 = ax[0].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax[0].legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    # Plot 2: Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[1],
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
                cbar_kws={'label': 'Count'})
    ax[1].set_title("Confusion Matrix", fontsize=14, fontweight='bold')
    ax[1].set_ylabel("Actual Label", fontsize=12)
    ax[1].set_xlabel("Predicted Label", fontsize=12)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'training_results.png'")
    plt.show()

if __name__ == "__main__":
    print("="*60)
    print("CUSTOM RESNET18 - TRAINING FROM SCRATCH")
    print("Full Dataset + Data Augmentation")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Initialize model
    net = model().to(device)
    trainloader, testloader = prepare_data()
    print(f"\nData loaded: {len(trainloader)} train batches, {len(testloader)} test batches\n")
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    # Track metrics for plotting
    train_losses = []
    test_accuracies = []
    
    # Train for 10 epochs (increased from 5 for full dataset)
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        net.train()
        running_loss = 0.0
        batch_count = 0
        
        for images, labels in trainloader:
            batch_count += 1
            if batch_count % 100 == 0:
                print(f"  Processing batch {batch_count}/{len(trainloader)} - Running loss: {running_loss/batch_count:.4f}")
            
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)
            preds = net(images)
            loss = criterion(preds, labels)  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Calculate epoch loss
        epoch_loss = running_loss / len(trainloader)
        train_losses.append(epoch_loss)
        
        print(f"\nEpoch {epoch+1} Training Loss: {epoch_loss:.4f}")
        
        # Evaluate after each epoch
        accuracy, all_labels, all_preds = evaluate(net, testloader, device)
        test_accuracies.append(accuracy)

    # Save model
    torch.save(net.state_dict(), "custom_resnet18_full_augmented.pth")
    print("\n" + "="*60)
    print("Model saved as 'custom_resnet18_full_augmented.pth'")
    print("="*60)
    
    # Plot results
    plot_results(train_losses, test_accuracies, all_labels, all_preds)
    
    print("\nTraining complete!")