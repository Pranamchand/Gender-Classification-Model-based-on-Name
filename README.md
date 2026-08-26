# ♀️♂️ Gender Classification Web Application

An end-to-end Machine Learning web application that predicts gender (Male/Female) from Indian personal names using character-level TF-IDF feature extraction and a Keras Neural Network. 

> ✨ **Powered by Google Antigravity**: The complete web frontend, glassmorphism UI design system, Flask REST API, and automated testing pipeline were pair-programmed and built using **Google Antigravity AI**.

---

## 🌟 Highlights & Features

- **High Accuracy Neural Network**: Trained on a dataset of over 100,000 names using a Deep Learning architecture (Dense + Dropout) with char-level TF-IDF vectorization ($n$-gram range 2–5).
- **Modern Glassmorphism UI**: Built with HTML5, Vanilla CSS3, and JavaScript, featuring:
  - Dark mode design with glowing background gradient blobs.
  - Interactive prediction result cards with custom male/female badge indicators.
  - Animated percentage confidence progress bar.
- **RESTful Flask API**: A lightweight Python backend serving real-time predictions via `POST /predict`.
- **End-to-End Pair-Programming**: Conceptualized, structured, and verified using **Antigravity AI Agent**.

---

## 🛠️ Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Model & Machine Learning** | Python, TensorFlow / Keras, Scikit-Learn, Pandas, NumPy |
| **Backend API** | Flask, Flask-CORS |
| **Frontend UI** | HTML5, Vanilla CSS3 (Glassmorphism), JavaScript (ES6+) |
| **AI Pair Programmer** | **Google Antigravity AI** |

---

## 🚀 Role of Google Antigravity

**Google Antigravity** played a key role in bringing this Machine Learning model to life:

1. **Automated Code Exploration**: Antigravity analyzed the Jupyter Notebook (`gender-classification-model.ipynb`), model artifacts (`gender_classification_model.h5`, `tfidf_vectorizer.pkl`, `label_encoder.pkl`), and dataset structure.
2. **Frontend Architecture & Design System**: Created a responsive UI with custom CSS variables, glassmorphism effects, dynamic CSS animations, and accessibility features.
3. **Backend Integration**: Built a Flask API (`app.py`) that loads pre-trained vectorizers and weights at startup to ensure low-latency predictions.
4. **Autonomous Testing & Verification**: Used Antigravity's integrated headless browser subagent to execute real-time UI interaction, form input, and confidence visualization verification.

---

## 📁 Project Structure

```text
Gender classification Model/
├── gender-classification-model.ipynb  # Model training & EDA notebook
├── gender_classification_model.h5    # Pre-trained Keras Sequential model
├── tfidf_vectorizer.pkl              # Fitted TF-IDF char n-gram vectorizer
├── label_encoder.pkl                 # Fitted Label Encoder (f / m)
├── app.py                            # Flask API & static server
├── requirements.txt                  # Python dependencies
├── static/
│   ├── style.css                     # Glassmorphism styling & animations
│   └── script.js                     # Frontend API fetch & animation logic
├── templates/
│   └── index.html                    # Single-page web application
└── README.md                         # Project documentation
```

---

## ⚙️ Quickstart & Local Setup

### 1. Prerequisites
- Python 3.9+ installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python app.py
```

### 4. Open in Browser
Navigate to **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your web browser.

---

## 🔬 Model Pipeline Details

1. **Preprocessing**: Cleans input names by converting to lowercase and stripping extra whitespace.
2. **Vectorization**: Uses `TfidfVectorizer` with `analyzer='char'` and `ngram_range=(2, 5)` limited to 10,000 max features.
3. **Neural Network Architecture**:
   - `Dense(64, activation='relu')`
   - `Dropout(0.3)`
   - `Dense(32, activation='relu')`
   - `Dense(1, activation='sigmoid')`
4. **Output**: Probability thresholded at 0.5 to produce gender output (`Male` / `Female`) alongside a confidence percentage score.

---

## 📜 License & Credits

- Developed as part of an Indian Names Gender Classification ML project.
- Frontend & Flask Integration by **Antigravity AI**.
