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
         torchvision.transforms.Normalize((0.5, 0.5, 0.5),  (0.5, 0.5, 0.5))
    ])

def model():
    weights   = ResNet18_Weights.DEFAULT
    model  = resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    return model

def prepare_data():

    # 1. load CSV
    df =  pd.read_csv("data/train.csv")
    
    # Subsample for quicker testing so it doesn't take forever
    df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)
    
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    # 2. transforms
    transform = get_transforms()

    # 3. datasets
    trainset =  RealFakeDataset(train_df, transform=transform)
    testset = RealFakeDataset(test_df, transform=transform)

    # 4. dataloaders
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)
    return trainloader, testloader

def evaluate(model, testloader, device):
    model.eval() 
    print ("Evaluating") # print statement to indicate evaluation start
    correct = 0
    total = 0
    batch_count = 0
    with torch.no_grad(): 
        for images, labels in testloader:
            batch_count += 1
            if batch_count % 10  == 0:
                print(f"  Eval batch {batch_count}/{len(testloader)}...")
            images,  labels =  images.to(device), labels.to(device)
            outputs  = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            correct +=  (preds.int() == labels.unsqueeze(1)).sum().item()
            total +=  labels.size(0)
    return correct /  total

if __name__ == "__main__":

    print("starting")
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    
    net =  model().to(device)
    trainloader,  testloader = prepare_data()
    print(f"Data loaded: {len(trainloader)} train batches")
    
    criterion =  torch.nn.BCEWithLogitsLoss()
    optimizer =  torch.optim.Adam(net.parameters(), lr=1e-4)

    #training loop 
    for epoch in range(5):
        net.train()
        batch_count = 0
        for images, labels in trainloader:
            batch_count += 1
            if batch_count % 10 == 0: # print progress every 10 batches for timing 
                print(f"  Processing batch {batch_count}/{len(trainloader)}")
            
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)
            preds = net( images )
            loss =  criterion(preds, labels)  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        #epoch tracking 
        print(f"Epoch {epoch+1}/5 is done")

        #accuracy calculation
        accuracy =  evaluate(net, testloader,  device)
        print(f"Epoch {epoch +1} Accuracy: {accuracy*100:.2f}%")

    #finish testing 
    torch.save(net.state_dict(), "real_fake_resnet18.pth")
    print("Done training")