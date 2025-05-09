# Grammar Scoring Engine 🗣️📊

This project builds a Grammar Scoring Engine that predicts a MOS (Mean Opinion Score) for spoken English grammar using machine learning and audio analysis. The model uses a variety of audio features such as MFCCs, Chroma, Spectral Contrast, Zero-Crossing Rate, and RMS energy.

---

## 🧠 How It Works

🎧 Input: Audio files (.wav) of English sentences  
🔍 Processing: Extract features → Train regression model  
📈 Output: Predicted grammar score (MOS) per audio file  

---

## 📥 Input Example

- train/
  audio_12.wav  
  audio_45.wav  
  ... (167 audio files)

- train.csv  
  filename,label  
  audio_12.wav,3.5  
  audio_45.wav,4.1  

---

## 📤 Output Example

- submission.csv  
  filename,label  
  audio_804.wav,3.57  
  audio_1028.wav,4.23  
  ...

- Visualization:  
  ![image](https://github.com/user-attachments/assets/e3097b9e-d416-443b-94d7-d440f7a90c85)

---

## 🚀 Running the Project

1. Clone the repo  
2. Install dependencies  
   pip install -r requirements.txt  
3. Run the notebook:  
   Go to notebooks/Grammar_Scoring_Model.ipynb  
   Run all cells

---

## 📈 Performance

- Training RMSE: ~0.32  
- Cross-validated RMSE: ~0.57  
- Kaggle Public Score: 0.570  

---
### Model

The trained model, `grammar_scoring_model.pkl`, is included in this repository. It can be used for making predictions on new audio files. It stores when the code is executed.
---

👨‍💻 Author
K Shreyank
AI | ML | Deep Learning | SQL | NoSQL | Computer Vision
