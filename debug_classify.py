import torch
from PIL import Image
import torchvision.transforms as transforms
from custom_resnet import CustomResNet
import pandas as pd
import os

def load_model(model_path, device):
    """Load the trained custom ResNet model from saved weights"""
    model = CustomResNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_transforms():
    """Same transforms used during training"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def test_on_training_data(model, device, num_samples=10):
    """Test the model on a few images from the training dataset to verify it works"""
    print("\n" + "="*60)
    print("TESTING ON TRAINING DATA (to verify model works)")
    print("="*60)
    
    df = pd.read_csv("data/train.csv")
    transform = get_transforms()
    
    # Get some real and fake examples
    real_samples = df[df['label'] == 0].head(num_samples//2)
    fake_samples = df[df['label'] == 1].head(num_samples//2)
    samples = pd.concat([real_samples, fake_samples])
    
    correct = 0
    total = 0
    
    for idx, row in samples.iterrows():
        img_path = os.path.join('data/', row['file_name'])
        true_label = row['label']
        
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(img_tensor)
                prob = torch.sigmoid(output).item()
            
            predicted_label = 1 if prob > 0.5 else 0
            correct += (predicted_label == true_label)
            total += 1
            
            label_name = "FAKE" if true_label == 1 else "REAL"
            pred_name = "FAKE" if predicted_label == 1 else "REAL"
            status = "✓" if predicted_label == true_label else "✗"
            
            print(f"{status} True: {label_name:4s} | Pred: {pred_name:4s} | Prob: {prob:.4f} | {row['file_name']}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"\nAccuracy on training samples: {correct}/{total} = {correct/total*100:.1f}%")
    print("="*60)

def classify_image(image_path, model_path="custom_resnet18.pth"):
    """
    Classify a single image as real or fake
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(model_path, device)
    
    # First, test on training data to verify model works
    test_on_training_data(model, device, num_samples=10)
    
    # Now classify the user's image
    print("\n" + "="*60)
    print("CLASSIFYING YOUR IMAGE")
    print("="*60)
    print(f"Image: {image_path}\n")
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    print(f"Image size: {image.size}")
    
    transform = get_transforms()
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        probability = torch.sigmoid(output).item()
    
    # Interpret result (1 = FAKE, 0 = REAL)
    prediction = "FAKE" if probability > 0.5 else "REAL"
    confidence = probability if probability > 0.5 else (1 - probability)
    
    print("\nRESULTS:")
    print(f"  Raw output: {output.item():.6f}")
    print(f"  Sigmoid probability: {probability:.6f}")
    print(f"  Prediction: {prediction}")
    print(f"  Confidence: {confidence*100:.2f}%")
    print(f"\nInterpretation: {probability*100:.2f}% probability of being FAKE")
    print("="*60)
    
    return prediction, confidence, output.item(), probability

if __name__ == "__main__":
    print("="*60)
    print("Custom ResNet Image Classifier - DEBUG MODE")
    print("="*60)
    
    # Update this path to your image
    image_path = "C:/Users/clcel/ai_image_4_fake.png"
    
    try:
        classify_image(image_path)
        
    except FileNotFoundError as e:
        print(f"\nERROR: Could not find file")
        print(f"Details: {e}")
        
    except Exception as e:
        print(f"\nERROR during classification: {e}")
        import traceback
        traceback.print_exc()