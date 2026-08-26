import re
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Gender Classifier AI",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #a855f7, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        font-weight: 300;
    }
    
    .gender-card {
        padding: 2rem;
        border-radius: 16px;
        color: #ffffff;
        text-align: center;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
    }
    
    .gender-card-female {
        background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 100%);
    }
    
    .gender-card-male {
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
    }
    
    .gender-title {
        font-size: 1.3rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    
    .gender-value {
        font-size: 3.2rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }
    
    .confidence-text {
        font-size: 1.2rem;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.2);
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 30px;
        backdrop-filter: blur(10px);
    }
    
    .sample-btn {
        margin-right: 8px;
        margin-bottom: 8px;
    }
    
    .metric-box {
        background: #1e1e2d;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
    }
    
    .metric-num {
        font-size: 1.8rem;
        font-weight: 700;
        color: #6366f1;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Model & Resources
# ---------------------------------------------------------
@st.cache_resource
def load_gender_model():
    """Load pre-trained Keras model, TF-IDF vectorizer, and Label Encoder."""
    model = tf.keras.models.load_model('gender_classification_model.h5')
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return model, vectorizer, encoder

try:
    model, vectorizer, encoder = load_gender_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)

# ---------------------------------------------------------
# Preprocessing Helper
# ---------------------------------------------------------
def clean_name(name_str: str) -> str:
    """Preprocess input name string identical to training pipeline."""
    if not isinstance(name_str, str):
        return ""
    name_str = name_str.lower()
    # Remove relationship/address keywords
    name_str = re.sub(r'\s+(r/o|ro|c/o|co|urf|and his son|and her daughter|ps)\b.*$', '', name_str)
    # Remove digits
    name_str = re.sub(r'\d+', '', name_str)
    # Remove extra whitespace
    name_str = re.sub(r'\s+', ' ', name_str).strip()
    return name_str

def predict_gender_single(name_raw: str):
    """Predict gender and probability for a single name."""
    cleaned = clean_name(name_raw)
    if not cleaned:
        return None
    
    # Vectorize
    tfidf_feat = vectorizer.transform([cleaned])
    # Predict probability (1 = Male 'm', 0 = Female 'f')
    prob = float(model.predict(tfidf_feat, verbose=0)[0][0])
    
    pred_idx = int(prob >= 0.5)
    predicted_gender_code = encoder.inverse_transform([pred_idx])[0]
    gender_label = "Male" if predicted_gender_code == 'm' else "Female"
    
    confidence = prob if pred_idx == 1 else (1.0 - prob)
    female_prob = 1.0 - prob
    male_prob = prob
    
    return {
        "raw_name": name_raw,
        "clean_name": cleaned,
        "predicted_gender": gender_label,
        "gender_code": predicted_gender_code,
        "confidence": confidence,
        "female_prob": female_prob,
        "male_prob": male_prob
    }

# ---------------------------------------------------------
# Sidebar Navigation & Info
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric-line/100/gender.png", width=70)
    st.title("Gender Classification")
    st.caption("AI Model Based on Name Analysis")
    
    st.markdown("---")
    st.markdown("### 🧠 Model Specs")
    st.markdown("""
    - **Architecture**: Deep Neural Network (64 ➔ Dropout 0.3 ➔ 32 ➔ Sigmoid)
    - **Vectorization**: TF-IDF Character n-grams (2-5), 10,000 features
    - **Training Accuracy**: ~87.5%
    - **Dataset Size**: 125,000+ Names
    """)
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Quick Info")
    st.info("The model analyzes character patterns, suffixes, and n-grams in a name to accurately classify gender with confidence scores.")

# ---------------------------------------------------------
# Main App Header
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="main-title">✨ Name-Based Gender Classification</div>
    <div class="main-subtitle">Predict male or female gender from any name using Deep Learning & Character TF-IDF N-grams</div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"❌ Error loading model assets: {load_error}")
    st.stop()

# ---------------------------------------------------------
# Main Tabs
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📋 Batch Classification", "📊 Model Insights"])

# ---------------------------------------------------------
# Tab 1: Single Prediction
# ---------------------------------------------------------
with tab1:
    st.markdown("### Enter Name for Prediction")
    
    # Preset sample names helper
    if "input_name" not in st.session_state:
        st.session_state["input_name"] = "Preeti"
    
    col_samples, col_clear = st.columns([4, 1])
    with col_samples:
        st.markdown("**Try Example Names:**")
        cols = st.columns(6)
        examples = ["Preeti", "Rahul", "Sukanti", "Jamaro", "Alfiya", "Henryka"]
        for idx, ex in enumerate(examples):
            if cols[idx].button(ex, key=f"ex_{ex}"):
                st.session_state["input_name"] = ex
    
    user_name_input = st.text_input(
        "Type any name below:",
        value=st.session_state["input_name"],
        placeholder="e.g. Ananya, Vikram, Sukanti..."
    )
    
    btn_predict = st.button("🚀 Predict Gender", type="primary", use_container_width=True)
    
    if user_name_input or btn_predict:
        res = predict_gender_single(user_name_input)
        if res:
            gender = res["predicted_gender"]
            conf_pct = res["confidence"] * 100
            
            card_class = "gender-card-female" if gender == "Female" else "gender-card-male"
            gender_symbol = "♀️ Female" if gender == "Female" else "♂️ Male"
            
            st.markdown(f"""
            <div class="gender-card {card_class}">
                <div class="gender-title">Predicted Gender</div>
                <div class="gender-value">{gender_symbol}</div>
                <div class="confidence-text">Confidence: {conf_pct:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed Probability Breakdown
            st.markdown("#### 📊 Probability Distribution")
            col_f, col_m = st.columns(2)
            
            with col_f:
                st.metric(label="Female Probability ♀️", value=f"{res['female_prob']*100:.2f}%")
                st.progress(res["female_prob"])
                
            with col_m:
                st.metric(label="Male Probability ♂️", value=f"{res['male_prob']*100:.2f}%")
                st.progress(res["male_prob"])
                
            st.caption(f"Cleaned internal name used: `{res['clean_name']}`")
        else:
            st.warning("Please enter a valid name (letters only).")

# ---------------------------------------------------------
# Tab 2: Batch Classification
# ---------------------------------------------------------
with tab2:
    st.markdown("### 📋 Batch Gender Classification")
    st.write("Upload a CSV file or type multiple names below to classify them in bulk.")
    
    batch_mode = st.radio("Choose Input Method:", ["Type Multiple Names", "Upload CSV File"], horizontal=True)
    
    batch_names_list = []
    
    if batch_mode == "Type Multiple Names":
        names_text = st.text_area(
            "Enter names (one per line):",
            value="Rahul Sharma\nPreeti Kumari\nAlfiya\nJamaro\nHenryka",
            height=150
        )
        if names_text.strip():
            batch_names_list = [n.strip() for n in names_text.split('\n') if n.strip()]
            
    else: # Upload CSV
        uploaded_file = st.file_uploader("Upload a CSV file containing a column named 'name':", type=["csv"])
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                if "name" in df_upload.columns:
                    batch_names_list = df_upload["name"].dropna().astype(str).tolist()
                    st.success(f"Loaded {len(batch_names_list)} names from CSV.")
                else:
                    st.error("Uploaded CSV must contain a column titled 'name'.")
            except Exception as ex:
                st.error(f"Error reading CSV file: {ex}")
                
    if st.button("⚡ Run Batch Prediction", type="primary"):
        if not batch_names_list:
            st.warning("Please provide at least one name to classify.")
        else:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, raw_name in enumerate(batch_names_list):
                res = predict_gender_single(raw_name)
                if res:
                    results.append({
                        "Name": raw_name,
                        "Cleaned Name": res["clean_name"],
                        "Predicted Gender": res["predicted_gender"],
                        "Confidence (%)": round(res["confidence"] * 100, 2),
                        "Female Prob (%)": round(res["female_prob"] * 100, 2),
                        "Male Prob (%)": round(res["male_prob"] * 100, 2)
                    })
                progress_bar.progress((idx + 1) / len(batch_names_list))
                status_text.text(f"Processing name {idx+1} of {len(batch_names_list)}...")
                
            status_text.success("Batch classification complete!")
            
            df_res = pd.DataFrame(results)
            st.dataframe(df_res, use_container_width=True)
            
            # Summary Metrics
            c1, c2, c3 = st.columns(3)
            total = len(df_res)
            females = len(df_res[df_res["Predicted Gender"] == "Female"])
            males = len(df_res[df_res["Predicted Gender"] == "Male"])
            
            c1.metric("Total Analyzed", total)
            c2.metric("Predicted Females ♀️", f"{females} ({females/total*100:.1f}%)")
            c3.metric("Predicted Males ♂️", f"{males} ({males/total*100:.1f}%)")
            
            # CSV Download
            csv_data = df_res.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name="gender_classification_results.csv",
                mime="text/csv"
            )

# ---------------------------------------------------------
# Tab 3: Model Insights & Architecture
# ---------------------------------------------------------
with tab3:
    st.markdown("### 📊 Model Architecture & Training Summary")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("#### ⚙️ Feature Extraction")
        st.write("""
        - **Method**: Character-level TF-IDF Vectorization
        - **N-gram Range**: (2, 5) (captures prefix, suffix, and character combinations)
        - **Max Vocabulary Features**: 10,000 top n-grams
        - **Preprocessing**: Automatic lowercasing, strip addresses/relations (do, so, ro, urf, etc.), remove numbers and special characters.
        """)
        
    with col_b:
        st.markdown("#### 🏗️ Keras Neural Network")
        st.code("""
Sequential([
    Dense(64, activation='relu', input_shape=(10000,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
        """, language="python")
        
    st.markdown("---")
    st.markdown("#### 📉 Performance & Accuracy")
    st.markdown("""
    - **Accuracy**: ~87.5% on test split (20% stratify split)
    - **Loss Function**: Binary Cross-Entropy
    - **Optimizer**: Adam
    - **Epochs Trained**: 10
    """)
