# 👤 Gender Classification Model Based on Name

A Machine Learning & Deep Learning application that predicts whether a given name is **Female** or **Male** using character-level **TF-IDF Vectorization** and a **TensorFlow / Keras Neural Network**. 

> 🚀 **Note:** The interactive Streamlit frontend for this project was built using **[Antigravity](https://deepmind.google/)**.

---

## 🌟 Key Features

- **⚡ Real-time Single Name Prediction**: Instant gender prediction with confidence scores (%) and probability breakdown.
- **📋 Batch Processing**: Predict gender for multiple names simultaneously via multi-line text input or CSV file upload (`name` column).
- **📥 Export Results**: One-click download of batch prediction results as a CSV file (`gender_classification_results.csv`).
- **🧹 Smart Preprocessing**: Automatic text cleaning that normalizes case, removes titles, address codes (`r/o`, `c/o`, `so`, `do`), numbers, and extra spaces.
- **📊 Interactive Analytics & Insights**: Sidebar and tabs providing insights into model architecture, n-grams, and training accuracy.

---

## 🧠 Model Architecture & Machine Learning Pipeline

1. **Preprocessing**: Cleans raw names by removing relationship/address keywords (`r/o`, `c/o`, `urf`, `so`, `do`, `ps`), digits, and normalizing whitespace.
2. **Feature Extraction**: Character-level **TF-IDF Vectorizer** using `ngram_range=(2, 5)` with a maximum of `10,000` features.
3. **Deep Learning Classifier**:
   - **Layer 1**: Dense (64 units, ReLU activation)
   - **Layer 2**: Dropout (0.3 rate for regularization)
   - **Layer 3**: Dense (32 units, ReLU activation)
   - **Output Layer**: Dense (1 unit, Sigmoid activation)
4. **Model Performance**: Achieves **~87.5% Accuracy** on a test split of 125,000+ names.

---

## 📁 Repository Structure

```text
├── app.py                            # Streamlit Web Application (Built with Antigravity)
├── gender_classification_model.h5    # Pre-trained Keras Model (.h5 format)
├── gender_model.keras                # Keras Model (.keras format)
├── tfidf_vectorizer.pkl              # Saved TF-IDF Character Vectorizer
├── label_encoder.pkl                 # Saved Label Encoder ('f' -> 0, 'm' -> 1)
├── Names_dataset.csv                 # Dataset containing names and gender labels
├── gender-classification-model.ipynb # Training & Exploratory Data Analysis Notebook
├── requirements.txt                  # Python Dependencies
├── .gitignore                        # Git Ignore file
└── README.md                         # Project Documentation
```

---

## 🛠️ Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Pranamchand/Gender-Classification-Model-based-on-Name.git
cd Gender-Classification-Model-based-on-Name
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

The web application will open automatically in your browser at `http://localhost:8501`.

---

## 🤝 Credits & Acknowledgments
- Frontend developed using **Antigravity**
- Built with **Streamlit**, **TensorFlow**, and **Scikit-Learn**
