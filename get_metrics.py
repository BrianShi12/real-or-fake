import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    roc_curve, 
    confusion_matrix, 
    classification_report
)
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

# Import your custom dataset class
from dataset import RealFakeDataset
# Import your transforms function from train.py
from train import get_transforms

# --- CONFIGURATION ---
MODEL_PATH = "augmented_real_fake_resnet18.pth" # CHANGE THIS to your actual file name
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_test_data():
    # We must replicate the split exactly as it was during training
    df = pd.read_csv("data/train.csv")
    _, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    
    # Use the "test" clean transforms
    test_transform = get_transforms(split="test")
    
    # ROOT DIR must be 'data/' based on your previous fix
    testset = RealFakeDataset(test_df, transform=test_transform, root_dir='data/')
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
    
    return testloader

def load_model():
    # Re-create the architecture
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    
    # Load the weights you trained
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"✅ Loaded weights from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {MODEL_PATH}. Check the filename!")
        exit()
        
    model.to(DEVICE)
    model.eval()
    return model

def get_predictions(model, testloader):
    y_true = []
    y_scores = [] # Raw probabilities (0.0 to 1.0)
    y_preds = []  # Hard 0 or 1 labels

    print("Running inference on test set...")
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(DEVICE)
            
            # Get raw model outputs (logits)
            outputs = model(images)
            
            # Convert to probability (0-1) using Sigmoid
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            
            # Convert to hard prediction (0 or 1)
            preds = (probs > 0.5).astype(int)
            
            y_scores.extend(probs)
            y_preds.extend(preds)
            y_true.extend(labels.numpy().flatten())
            
    return np.array(y_true), np.array(y_preds), np.array(y_scores)

def print_metrics(y_true, y_preds, y_scores):
    # 1. Calculate Standard Metrics
    acc = accuracy_score(y_true, y_preds)
    prec = precision_score(y_true, y_preds)
    rec = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    auc = roc_auc_score(y_true, y_scores)

    print("\n" + "="*30)
    print("FINAL MODEL PERFORMANCE")
    print("="*30)
    print(f"Accuracy:  {acc:.4f}  (Overall correctness)")
    print(f"Precision: {prec:.4f}  (When it says 'Fake', is it really fake?)")
    print(f"Recall:    {rec:.4f}  (Did it catch all the fakes?)")
    print(f"F1 Score:  {f1:.4f}  (Balance of Precision/Recall)")
    print(f"AUROC:     {auc:.4f}  (Ability to distinguish classes)")
    print("-" * 30)
    
    print("\nDetailed Report:")
    print(classification_report(y_true, y_preds, target_names=["Real", "Fake"]))

    # 2. Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_preds)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # 3. Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    testloader = load_test_data()
    model = load_model()
    y_true, y_preds, y_scores = get_predictions(model, testloader)
    print_metrics(y_true, y_preds, y_scores)