import os
import librosa
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import joblib
from tqdm import tqdm

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

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs.T, axis=0)
    return mfcc_mean

model = joblib.load('rf_voice_model.pkl')

test_dir = '../deep_voice/test'
X_test, y_test = load_data(test_dir)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100
f1 = f1_score(y_test, y_pred) * 100

print(f"Accuracy {accuracy:.2f}%. F1 {f1:.2f}%.")