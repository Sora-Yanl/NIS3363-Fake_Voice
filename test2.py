import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import librosa
from torchvision import transforms as T
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

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
    
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_data(directory):
    files = []
    categories = []
    for category_dir in ['real', 'fake']:
        for filename in tqdm(os.listdir(os.path.join(directory, category_dir)), desc=f'Loading {category_dir} files'):
            files.append(os.path.join(directory, category_dir, filename))
            categories.append(category_dir)
    return files, categories

def evaluate_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.float(), labels
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = running_loss / len(test_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='weighted') * 100
    
    return accuracy, f1, all_labels, all_preds

test_root_dir = '../deep_voice/test'
test_file_list, test_labels = load_data(test_root_dir)

label_encoder = LabelEncoder()
test_labels = label_encoder.fit_transform(test_labels)

transform = T.Compose([
    T.ToTensor(),
    T.Resize((128, 128))
])

test_dataset = AudioDataset(test_file_list, test_labels, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = AudioCNN()
model.load_state_dict(torch.load('best_model.pth'))
criterion = nn.CrossEntropyLoss()

accuracy, f1, all_labels, all_preds = evaluate_model(model, test_loader, criterion)
print(f'Accuracy {accuracy:.2f}%. F1 {f1:.2f}%.')