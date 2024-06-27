import os
import numpy as np
import librosa
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

class AudioDataset(Dataset):
    def __init__(self, files, labels, transform=None):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]

        y, sr = librosa.load(file_path, sr=None)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        if self.transform:
            mel_spec_db = self.transform(mel_spec_db)
        else:
            mel_spec_db = np.expand_dims(mel_spec_db, axis=0)

        return mel_spec_db, label


def load_data(directory):
    files = []
    categories = []
    for category_dir in ['real', 'fake']:
        for filename in tqdm(os.listdir(os.path.join(directory, category_dir)), desc=f'Loading {category_dir} files'):
            files.append(os.path.join(directory, category_dir, filename))
            categories.append(category_dir)
    return files, categories

root_dir = '../deep_voice/train'
train_file_list, train_labels = load_data(root_dir)

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)

train_file_list, val_file_list, train_labels, val_labels = train_test_split(train_file_list, train_labels, test_size=0.2, random_state=42)

transform = T.Compose([
    T.ToTensor(),
    T.Resize((128, 128))
])

train_dataset = AudioDataset(train_file_list, train_labels, transform=transform)
val_dataset = AudioDataset(val_file_list, val_labels, transform=transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    best_model_wts = None
    best_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        corrects = 0
        total_samples = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.float(), labels

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels)
            total_samples += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = corrects.double() / total_samples
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        corrects = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.float(), labels
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels)
                total_samples += labels.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = corrects.double() / total_samples
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')
        
        # Check for best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()

    torch.save(best_model_wts, 'best_model.pth')

model = AudioCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25)