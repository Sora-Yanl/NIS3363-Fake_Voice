# Readme

## 运行环境

python==3.8.19

pytorch==1.10.0

torchvision==0.11.0

torchaudio==0.10.0

torchtext==0.11.0

x-transformers==0.15.0

transformers==4.35.2

librosa==0.10.2

scipy==1.9.3

## 文件结构


├─ Readme.md
├─ best_model.pth
├─ report.pdf
├─ rf_voice_model.pkl
├─ test1.py
├─ test2.py
├─ train1.py
└─ train2.py

执行`python train1.py`或`python train2.py`会在目录下生成模型文件，直接执行`python test1.py`或`python test2.py`将直接用模型生成评估结果。
