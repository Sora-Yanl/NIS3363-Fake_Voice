import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from tqdm import tqdm

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=1.0)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

def load_data(data_dir):
    labels = []
    features = []
    
    for label, folder in enumerate(['fake', 'real']):
        folder_path = os.path.join(data_dir, folder)
        for file_name in tqdm(os.listdir(folder_path), desc=f'Loading {folder} files'):
            file_path = os.path.join(folder_path, file_name)
            if file_path.endswith('.wav'):
                mfcc_mean = extract_features(file_path)
                features.append(mfcc_mean)
                labels.append(label)
    
    return np.array(features), np.array(labels)

train_dir = '../deep_voice/train'
X_train, y_train = load_data(train_dir)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'rf_voice_model.pkl')