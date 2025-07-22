import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import requests
from io import BytesIO

# ===========
# 🔽 Helper to load from GitHub
# ===========
def load_pickle_from_url(url):
    response = requests.get(url)
    return pickle.load(BytesIO(response.content))

def load_joblib_from_url(url):
    response = requests.get(url)
    return joblib.load(BytesIO(response.content))

# ===========
# 🌐 Load all models from GitHub repo
# Replace with your raw GitHub links
# ===========

# SVM Category Classifier
svm_model = load_joblib_from_url("https://raw.githubusercontent.com/your-username/your-repo/main/svm_task_classifier.joblib")
category_encoder = load_joblib_from_url("https://raw.githubusercontent.com/your-username/your-repo/main/svm_label_encoder.joblib")

# XGBoost Priority Predictor
priority_model = load_pickle_from_url("https://raw.githubusercontent.com/your-username/your-repo/main/priority_xgboost%20(1).pkl")
priority_encoder = load_pickle_from_url("https://raw.githubusercontent.com/your-username/your-repo/main/priority_label_encoder%20(1).pkl")
priority_vectorizer = load_pickle_from_url("https://raw.githubusercontent.com/your-username/your-repo/main/priority_tfidf_vectorizer%20(1).pkl")

# ===========
# 🧠 Dummy User Data (replace with your real user dataset if needed)
# ===========
users_df = pd.DataFrame({
    "user_id": ["User_A", "User_B", "User_C"],
    "current_load": [5, 3, 2]  # Example load
})

# ===========
# 🚀 Streamlit UI
# ===========
st.set_page_config(page_title="AI Task Manager", layout="centered")
st.title("🧠 AI Task Manager Dashboard")
st.markdown("Type your task below and let AI predict its category, priority, and assign it to a user!")

# ===========
# 📥 Input
# ===========
task_description = st.text_area("📝 Enter Task Description")

if st.button("Predict"):
    if not task_description.strip():
        st.warning("Please enter a task description.")
    else:
        # ===== TF-IDF for priority
        priority_features = priority_vectorizer.transform([task_description])
        predicted_priority = priority_model.predict(priority_features)[0]
        predicted_priority_label = priority_encoder.inverse_transform([predicted_priority])[0]

        # ===== TF-IDF for category (same vectorizer assumed, or you can load another)
        category_features = priority_vectorizer.transform([task_description])
        predicted_category = svm_model.predict(category_features)[0]
        predicted_category_label = category_encoder.inverse_transform([predicted_category])[0]

        # ===== Assign user with lowest workload
        assigned_user = users_df.loc[users_df["current_load"].idxmin(), "user_id"]

        # ===== Display Output
        st.success(f"✅ **Predicted Category**: {predicted_category_label}")
        st.info(f"🚦 **Predicted Priority**: {predicted_priority_label}")
        st.success(f"👤 **Assigned to**: {assigned_user}")
