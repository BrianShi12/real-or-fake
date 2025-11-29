import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from dataset import RealFakeDataset
import torchvision

# Import custom ResNet instead of pretrained
from custom_resnet import CustomResNet

# ADD THESE for metrics:
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import numpy as np

def get_transforms():
    return torchvision.transforms.Compose([
         torchvision.transforms.Resize((224, 224)),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5),  (0.5, 0.5, 0.5))
    ])

def model():
    # Use custom ResNet instead
    model = CustomResNet()
    
    print(model)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(f"{name}: kernel_size={module.kernel_size}, stride={module.stride}")
    return model

def prepare_data():
    df = pd.read_csv("data/train.csv")
    df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)
    
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    transform = get_transforms()
    trainset = RealFakeDataset(train_df, transform=transform)
    testset = RealFakeDataset(test_df, transform=transform)

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)
    return trainloader, testloader

def evaluate(model, testloader, device):
    model.eval() 
    print("Evaluating...")
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    batch_count = 0
    with torch.no_grad(): 
        for images, labels in testloader:
            batch_count += 1
            if batch_count % 10 == 0:
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
    
    # Print detailed metrics
    print("\n" + "="*50)
    print("FINAL MODEL PERFORMANCE")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}  (Overall correctness)")
    print(f"AUROC: {auroc:.4f}  (Ability to distinguish classes)")
    print("\nDetailed Report:")
    print(classification_report(all_labels, all_preds, 
                                target_names=['Real', 'Fake'],
                                digits=2))
    print("="*50 + "\n")
    
    return accuracy

if __name__ == "__main__":
    print("starting")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    net = model().to(device)
    trainloader, testloader = prepare_data()
    print(f"Data loaded: {len(trainloader)} train batches")
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    # Train for more epochs since no pretrained weights
    for epoch in range(5):
        net.train()
        batch_count = 0
        for images, labels in trainloader:
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"  Processing batch {batch_count}/{len(trainloader)}")
            
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)
            preds = net(images)
            loss = criterion(preds, labels)  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/5 is done")
        accuracy = evaluate(net, testloader, device)

    torch.save(net.state_dict(), "custom_resnet18.pth")
    print("Done training")