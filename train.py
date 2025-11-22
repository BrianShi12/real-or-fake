import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch

from dataset import RealFakeDataset
import torchvision
from torchvision.models import resnet18, ResNet18_Weights

def get_transforms():
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

def model():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    return model

def prepare_data():
    # 1. load CSV
    df = pd.read_csv("data/train.csv")

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )

    # 2. transforms
    transform = get_transforms()

    # 3. datasets
    trainset = RealFakeDataset(train_df, transform=transform)
    testset  = RealFakeDataset(test_df,  transform=transform)

    # 4. dataloaders
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader  = DataLoader(testset, batch_size=32, shuffle=False)

    return trainloader, testloader

def evaluate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds.int() == labels.unsqueeze(1)).sum().item()
            total += labels.size(0)
    return correct / total

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model().to(device)

    trainloader, testloader = prepare_data()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(5):
        model.train()
        for images, labels in trainloader:
            labels = labels.float().unsqueeze(1)

            preds = model(images)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} done")
        accuracy = evaluate(model, testloader, device)
        print(f"Test Accuracy: {accuracy*100:.2f}%")

    torch.save(model.state_dict(), "real_fake_resnet18.pth")

    


