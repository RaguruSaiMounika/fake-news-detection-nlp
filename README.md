# 📰 Fake News Detection using NLP

## 📌 Overview

This project presents a machine learning-based system to detect fake news using Natural Language Processing (NLP). It analyzes news articles (title + content) and classifies them as **Real** or **Fake**.

The model uses **TF-IDF vectorization** and a **Linear Support Vector Machine (SVM)** to achieve accurate classification of textual data.

---

## 🎯 Objectives

* Detect fake news automatically
* Reduce misinformation spread
* Provide real-time prediction using a web app
* Improve media literacy through visualization

---

## 🧠 Technologies Used

* Python
* Machine Learning
* Natural Language Processing (NLP)
* Scikit-learn
* Flask (Web App)

---

## ⚙️ Project Workflow

1. Data Collection
2. Data Preprocessing
3. Feature Extraction (TF-IDF)
4. Model Training (Linear SVM)
5. Prediction & Evaluation
6. Web Application Integration

---

## 📊 Features

* Real-time fake news detection
* User-friendly web interface
* Word cloud visualization
* Bar chart representation
* High accuracy classification

---

## 📂 Project Structure

```
├── app.py
├── train.py
├── predict.py
├── templates/
├── static/
├── model.pkl
└── requirements.txt
```

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python app.py
```

---

## 📁 Dataset

Due to GitHub size limits, the dataset is not included.

👉 You can use any fake news dataset or add your own.

---

## ⚠️ Limitations

* Depends on dataset quality
* Cannot verify facts independently
* Works mainly for English text

---

## 🌍 Impact

This project helps in:

* Reducing misinformation
* Supporting journalists
* Promoting responsible news consumption

---

## 👩‍💻 Author

**Raguru Sai Mounika**

---

## 📌 Future Improvements

* Use Deep Learning (BERT, LSTM)
* Add multilingual support
* Improve real-time scalability
