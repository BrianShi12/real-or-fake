import torch
from PIL import Image
import torchvision.transforms as transforms
from custom_resnet import CustomResNet

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

def classify_image(image_path, model_path="custom_resnet18.pth"):
    """
    Classify a single image as real or fake
    
    Args:
        image_path: Path to the image file
        model_path: Path to the saved model weights
    
    Returns:
        prediction: "FAKE" or "REAL"
        confidence: Confidence score (0-1)
        raw_output: Raw model output (for debugging)
        probability: Sigmoid probability (for debugging)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(model_path, device)
    
    # Load and preprocess image
    print(f"Loading image from: {image_path}")
    image = Image.open(image_path).convert('RGB')
    transform = get_transforms()
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)
    
    # Make prediction
    print("Making prediction...")
    with torch.no_grad():
        output = model(image_tensor)
        probability = torch.sigmoid(output).item()
    
    # Interpret result (1 = FAKE, 0 = REAL based on your training)
    prediction = "FAKE" if probability > 0.5 else "REAL"
    confidence = probability if probability > 0.5 else (1 - probability)
    
    return prediction, confidence, output.item(), probability

if __name__ == "__main__":
    print("="*60)
    print("Custom ResNet Image Classifier")
    print("="*60)
    
    # Update this path to your image
    image_path = "C:/Users/clcel/ai_image_3.png"
    
    print(f"\nImage: {image_path}")
    print(f"Model: custom_resnet18.pth\n")
    
    try:
        prediction, confidence, raw_output, probability = classify_image(image_path)
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence*100:.2f}%")
        print(f"\nDebug Info:")
        print(f"  Raw model output: {raw_output:.6f}")
        print(f"  Sigmoid probability: {probability:.6f}")
        print(f"  Interpretation: {probability:.2%} probability of being FAKE")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\nERROR: Could not find file")
        print(f"Details: {e}")
        print("\nPlease check:")
        print("  1. Image exists at the specified path")
        print("  2. Model file 'custom_resnet18.pth' exists in current directory")
        
    except Exception as e:
        print(f"\nERROR during classification: {e}")
        import traceback
        traceback.print_exc()