import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import os


###############################################
# DEVICE
###############################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


###############################################
# LOAD CSV (located at data/train.csv)
###############################################

csv_path = os.path.join("data", "train.csv")
df = pd.read_csv(csv_path)

print("CSV Loaded:")
print(df.head())


###############################################
# TRANSFORMS (Resize high-res images → 224)
###############################################

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    ),
])


###############################################
# DATASET CLASS (fixes your path problem)
###############################################

class AIDataset(Dataset):
    def __init__(self, df, root_dir="data", transform=None):
        self.df = df
        self.root = root_dir     # "data"
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # file_name column already contains "train_data/xxxxx.jpg"
        rel_path = row["file_name"]  # ex: "train_data/abcd.jpg"

        # final full path: data/train_data/abcd.jpg
        img_path = os.path.join(self.root, rel_path)

        # label 0 or 1
        label = int(row["label"])

        # load image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


###############################################
# CREATE DATASET + DATALOADERS
###############################################

dataset = AIDataset(df, root_dir="data", transform=transform)

train_size = int(0.7 * len(dataset))
val_size   = int(0.15 * len(dataset))
test_size  = len(dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

batch_size = 16   # can increase to 32 or 64 if GPU

# IMPORTANT for Windows → set num_workers=0
trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
valloader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
testloader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

print("Train size:", len(train_ds))
print("Val size:", len(val_ds))
print("Test size:", len(test_ds))


###############################################
# CNN MODEL (binary classification)
###############################################

class BinaryCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        # After 3 pools: 224 → 112 → 56 → 28
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)   # two classes

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


net = BinaryCNN().to(device)


###############################################
# TRAINING SETUP
###############################################

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)


###############################################
# TRAINING LOOP
###############################################

epochs = 3

for epoch in range(epochs):
    net.train()
    total_loss = 0.0

    for images, labels in trainloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} — Loss: {total_loss/len(trainloader):.4f}")


###############################################
# EVALUATION FUNCTION
###############################################

def evaluate(loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


###############################################
# VALIDATION ACCURACY
###############################################

val_acc = evaluate(valloader)
print(f"Validation accuracy: {val_acc*100:.2f}%")


###############################################
# TEST ACCURACY
###############################################

test_acc = evaluate(testloader)
print(f"TEST accuracy: {test_acc*100:.2f}%")
