import torch
import torchvision.transforms as transforms
from PIL import Image
from model import MalariaCNN

class MalariaPredictor:
    def __init__(self, model_path='best_malaria_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MalariaCNN()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
    
    def predict(self, image_path):
        """
        Predict whether cell is infected or healthy
        
        Args:
            image_path: Path to cell image
            
        Returns:
            dict: {
                'prediction': 'Infected' or 'Healthy',
                'confidence': float (0-100),
                'probability': float (0-1)
            }
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            probability = output.item()
        
        # Interpret results
        if probability >= 0.5:
            prediction = 'Infected'
            confidence = probability * 100
        else:
            prediction = 'Healthy'
            confidence = (1 - probability) * 100
        
        return {
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'probability': round(probability, 4)
        }