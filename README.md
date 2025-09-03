# 🧠 EEG Emotion Classifier

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Work_in_Progress-orange.svg)
![ML](https://img.shields.io/badge/ML-SVM%20%7C%20Feature%20Extraction-brightgreen)


A machine learning pipeline for emotion classification from EEG signals.  
Currently implemented with classical ML methods (SVM) on the preprocessed **DEAP** dataset.


## 📂 Project Structure
```
eeg-emotion-classifier/
├── data/ # raw and processed EEG data (not included in repo)
├── src/ # preprocessing, feature extraction, models, pipeline
├── results/ # evaluation results (confusion matrix, ROC curve, metrics)
├── main.py # entry point
├── requirements.txt # dependencies
└── .gitignore
```

## ⚙️ Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/goktug-sirma/eeg-emotion-classifier.git
cd eeg-emotion-classifier
pip install -r requirements.txt
```
## ▶️ Usage
1. Download the [DEAP dataset](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/)  
   - Required files: `data_preprocessed_python.zip`  
   - Extract the archive and place all `sXX.dat` files into `data/raw/`.  

2. Run the pipeline:
```bash
python main.py
```

## 📊 Results
Results will be added after running on real data.
Expected outputs:
- results/confusion_matrix.png
- results/roc_curve.png
- results/metrics.txt

## 🗺️ Roadmap
- ✅ Project structure and pipeline skeleton  
- ✅ Preprocessing and feature extraction  
- ✅ SVM training and evaluation  
- ⬜ Add results with real datasets  
- ⬜ Explore cross-subject generalization  
- ⬜ Experiment with deep learning models (CNN, LSTM, Transformers) 

## 📜 License
This project is licensed under the [MIT License](LICENSE).
