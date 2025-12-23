# %%
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from dvclive import Live
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageFile
from tqdm import tqdm
import yaml

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {device}")


# %%
with open('../params.yaml', 'r') as f:
    params = yaml.safe_load(f)

print("Parameters:")
print(yaml.dump(params, default_flow_style=False))


# %%
df = pd.read_csv('../data/balanced_animals_dataset.csv')
print(f"Dataset shape: {df.shape}")


# %%
label_to_idx = {label: idx for idx, label in enumerate(df['scientific_name'].unique())}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
df['label'] = df['scientific_name'].map(label_to_idx)


# %%
train_df, val_df = train_test_split(
    df, 
    test_size=params['data']['train_test_split'], 
    stratify=df['label'], 
    random_state=params['data']['random_state']
)

print(f"Train: {len(train_df)}, Val: {len(val_df)}")


# %%
class AnimalDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.base_path = Path('../animal_images')
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        species = row['scientific_name'].replace(' ', '_')
        img_path = self.base_path / species / f"{row['uuid']}.jpg"
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            
        label = row['label']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


# %%
train_transform = transforms.Compose([
    transforms.Resize(params['augmentation']['resize']),
    transforms.RandomCrop(params['augmentation']['crop_size']),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=params['augmentation']['brightness'],
        contrast=params['augmentation']['contrast'],
        saturation=params['augmentation']['contrast']
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(params['augmentation']['resize']),
    transforms.CenterCrop(params['augmentation']['crop_size']),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# %%
train_dataset = AnimalDataset(train_df, transform=train_transform)
val_dataset = AnimalDataset(val_df, transform=val_transform)

batch_size = params['training']['batch_size']
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# %%
if params['model']['name'] == 'resnet50':
    weights = getattr(models.ResNet50_Weights, params['model']['pretrained'])
    model = models.resnet50(weights=weights)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, params['model']['num_classes'])
    
elif params['model']['name'] == 'resnet101':
    weights = getattr(models.ResNet101_Weights, params['model']['pretrained'])
    model = models.resnet101(weights=weights)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, params['model']['num_classes'])
    
elif params['model']['name'] == 'efficientnet_v2_m':
    weights = getattr(models.EfficientNet_V2_M_Weights, params['model']['pretrained'])
    model = models.efficientnet_v2_m(weights=weights)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, params['model']['num_classes'])
    
elif params['model']['name'] == 'convnext_base':
    weights = getattr(models.ConvNeXt_Base_Weights, params['model']['pretrained'])
    model = models.convnext_base(weights=weights)
    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_features, params['model']['num_classes'])
    
elif params['model']['name'] == 'vit_b_16':
    weights = getattr(models.ViT_B_16_Weights, params['model']['pretrained'])
    model = models.vit_b_16(weights=weights)
    num_features = model.heads.head.in_features
    model.heads.head = nn.Linear(num_features, params['model']['num_classes'])

for param in model.parameters():
    param.requires_grad = False

# Разморозить последний слой
if 'efficientnet' in params['model']['name']:
    for param in model.classifier.parameters():
        param.requires_grad = True
elif 'convnext' in params['model']['name']:
    for param in model.classifier.parameters():
        param.requires_grad = True
elif 'vit' in params['model']['name']:
    for param in model.heads.parameters():
        param.requires_grad = True
else:
    for param in model.fc.parameters():
        param.requires_grad = True

model = model.to(device)
print(f"Model: {params['model']['name']}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# %%
criterion = nn.CrossEntropyLoss()

# Получить параметры для обучения в зависимости от модели
if 'efficientnet' in params['model']['name']:
    trainable_params = model.classifier.parameters()
elif 'convnext' in params['model']['name']:
    trainable_params = model.classifier.parameters()
elif 'vit' in params['model']['name']:
    trainable_params = model.heads.parameters()
else:
    trainable_params = model.fc.parameters()

optimizer = optim.Adam(trainable_params, lr=params['training']['learning_rate'])
scheduler = optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=params['scheduler']['step_size'], 
    gamma=params['scheduler']['gamma']
)


# %%
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total


# %%
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return running_loss / len(loader), 100. * correct / total, np.array(all_preds), np.array(all_labels)


# %%
num_epochs = params['training']['num_epochs']
best_val_acc = 0.0
Path('../models').mkdir(exist_ok=True)

with Live(dir='../dvclive', save_dvc_exp=True) as live:
    
    live.log_params(params)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        live.log_metric('train/loss', train_loss)
        live.log_metric('train/accuracy', train_acc)
        live.log_metric('val/loss', val_loss)
        live.log_metric('val/accuracy', val_acc)
        live.next_step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_to_idx': label_to_idx,
                'idx_to_label': idx_to_label,
                'params': params
            }, '../models/best_model.pth')
            print(f"Best model saved: {val_acc:.2f}%")
        
        scheduler.step()
    
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(val_labels, val_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[idx_to_label[i] for i in range(len(idx_to_label))],
        yticklabels=[idx_to_label[i] for i in range(len(idx_to_label))],
        ax=ax
    )
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    confusion_matrix_path = '../dvclive/plots/confusion_matrix.png'
    Path(confusion_matrix_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(confusion_matrix_path, dpi=150, bbox_inches='tight')
    live.log_image('confusion_matrix.png', confusion_matrix_path)
    plt.close()
    
    live.log_metric('best_val_accuracy', best_val_acc)



