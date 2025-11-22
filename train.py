import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch

from dataset import RealFakeDataset
from utils import get_transforms
import torchvision.transforms as transforms

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

def main():
    # 1. load CSV
    df = pd.read_csv("data/labels.csv")

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

    # 5. model
    model = create_model()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)



    # 6. training loop
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







if name == "main":
    main()

