import argparse
import json
import os
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--model', required=True, help='Path to model file')
    args = parser.parse_args()
    
    # First, try to use the trained model
    try:
        import torch
        import torch.nn.functional as F
        from torchvision import models, transforms
        from PIL import Image
        
        device = torch.device('cpu')
        
        # Load model
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        
        if not os.path.exists(args.model):
            raise FileNotFoundError(f"Model file not found: {args.model}")
        
        # Load trained weights
        state_dict = torch.load(args.model, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img = Image.open(args.image).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            real_prob = probabilities[0][0].item()
            fake_prob = probabilities[0][1].item()
        
        # Debug output to stderr (won't interfere with JSON output)
        print(f"Debug: Real prob = {real_prob:.4f}, Fake prob = {fake_prob:.4f}", file=sys.stderr)
        
        # Return result
        result = {
            'fake_probability': round(fake_prob, 4),
            'real_probability': round(real_prob, 4),
            'model': os.path.basename(args.model),
            'detection_method': 'neural_network'
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        # If model loading fails, use heuristic fallback
        print(f"Error: {str(e)}", file=sys.stderr)
        print(f"Falling back to heuristic analysis...", file=sys.stderr)
        
        # Simple heuristic fallback
        try:
            import numpy as np
            from PIL import Image, ImageStat, ImageFilter
            
            img = Image.open(args.image).convert('RGB')
            img_array = np.array(img)
            
            # Basic features
            stat = ImageStat.Stat(img)
            color_variance = np.mean(stat.var)
            brightness = np.mean(img_array)
            
            # Simple scoring
            score = 0.5  # Start neutral
            
            if color_variance < 1000:
                score += 0.2
            if brightness > 220 or brightness < 40:
                score += 0.15
            
            # Add some randomness
            import random
            score += random.uniform(-0.1, 0.1)
            score = max(0.0, min(1.0, score))
            
            result = {
                'fake_probability': round(score, 4),
                'real_probability': round(1.0 - score, 4),
                'model': os.path.basename(args.model),
                'detection_method': 'heuristic_fallback'
            }
            
            print(json.dumps(result))
            
        except Exception as e2:
            # Ultimate fallback
            print(f"Heuristic also failed: {str(e2)}", file=sys.stderr)
            result = {
                'fake_probability': 0.5,
                'real_probability': 0.5,
                'model': os.path.basename(args.model),
                'detection_method': 'error_fallback'
            }
            print(json.dumps(result))

if __name__ == '__main__':
    main()