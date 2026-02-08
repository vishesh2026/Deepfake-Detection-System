import os
import random
import json
import torch
import numpy as np
from torch import nn, optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

class DeepfakeDataset(Dataset):
    """Dataset that can load both real images and synthetic data"""
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.classes = ["real", "fake"]
        self.imgs = []
        self.labels = []
        
        # Try to load from sample_dataset
        sample_path = os.path.join("data", "sample_dataset", split)
        
        dataset_loaded = False
        
        # Try loading from sample_dataset
        if os.path.exists(sample_path):
            for cls_idx, cls in enumerate(self.classes):
                cls_dir = os.path.join(sample_path, cls)
                if os.path.exists(cls_dir):
                    for img_file in os.listdir(cls_dir):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                            self.imgs.append(os.path.join(cls_dir, img_file))
                            self.labels.append(cls_idx)
            
            if len(self.imgs) > 0:
                dataset_loaded = True
                print(f"✓ Loaded {len(self.imgs)} images from sample_dataset/{split}")
        
        if not dataset_loaded:
            print(f"❌ No images found in data/sample_dataset/{split}/")
            print(f"Please add images to:")
            print(f"  - data/sample_dataset/{split}/real/")
            print(f"  - data/sample_dataset/{split}/fake/")
            raise FileNotFoundError("No training images found!")
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

def train_model():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Strong data augmentation for small dataset
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("Loading datasets...")
    try:
        train_ds = DeepfakeDataset("data/sample_dataset", split="train", transform=train_transform)
        val_ds = DeepfakeDataset("data/sample_dataset", split="val", transform=val_transform)
    except FileNotFoundError as e:
        print("\n" + "="*60)
        print("ERROR: Please add images to your dataset first!")
        print("="*60)
        print("\nQuick guide:")
        print("1. Download 10 images from thispersondoesnotexist.com")
        print("2. Download 10 portraits from unsplash.com")
        print("3. Put 8 of each in data/sample_dataset/train/")
        print("4. Put 2 of each in data/sample_dataset/val/")
        print("\nThen run this script again.")
        return
    
    # Check if dataset is too small
    if len(train_ds) < 4:
        print(f"\n⚠️  WARNING: Only {len(train_ds)} training images found!")
        print("You need at least 4 images (2 real + 2 fake) to train.")
        print("Recommended: 16 images for train, 4 for val")
        return
    
    print(f"✓ Training images: {len(train_ds)}")
    print(f"✓ Validation images: {len(val_ds)}\n")
    
    # Small batch size for small dataset
    batch_size = min(4, len(train_ds))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Load pretrained ResNet18
    print("Loading ResNet18 model...")
    model = models.resnet18(pretrained=True)
    
    # Freeze early layers (keep pretrained features)
    for param in list(model.parameters())[:-10]:
        param.requires_grad = False
    
    # Modify final layer
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    log_lines = []
    best_val_loss = float('inf')
    
    # More epochs for small dataset
    num_epochs = 20
    
    print(f"Starting training for {num_epochs} epochs...\n")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1:2d}/{num_epochs} [Train]", ncols=100)
        for imgs, labels in progress_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            progress_bar.set_postfix({'loss': f'{loss.item():.3f}'})
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = outputs.argmax(1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = sum([1 for t, p in zip(y_true, y_pred) if t == p]) / len(y_true) if len(y_true) > 0 else 0
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        log_line = f"Epoch {epoch+1:2d} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        log_lines.append(log_line)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(log_line)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/resnet18_finetuned.pth")
            if (epoch + 1) % 5 == 0:
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
    
    # Final evaluation
    if len(y_true) > 0 and len(set(y_true)) > 1:
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    else:
        precision = recall = f1 = 0.0
        print("\n⚠️  Warning: Not enough validation data for metrics")
    
    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "val_accuracy": float(val_acc),
        "best_val_loss": float(best_val_loss),
        "train_images": len(train_ds),
        "val_images": len(val_ds)
    }
    
    os.makedirs("ml/eval", exist_ok=True)
    with open("ml/eval/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    with open("training_log.txt", "w") as f:
        f.write("\n".join(log_lines))
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"Final Metrics:")
    print(f"  Training Accuracy: {train_acc:.2%}")
    print(f"  Validation Accuracy: {val_acc:.2%}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"\n  Training images used: {len(train_ds)}")
    print(f"  Validation images used: {len(val_ds)}")
    print("="*60)
    print("\n✓ Model saved to: models/resnet18_finetuned.pth")
    print("✓ Metrics saved to: ml/eval/metrics.json")
    print("✓ Training log saved to: training_log.txt")
    print("\nNext steps:")
    print("1. Restart your backend: cd backend && node index.js")
    print("2. Test with images in your web interface!")

if __name__ == "__main__":
    train_model()