import streamlit as st
import pandas as pd
import joblib
import pickle
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================
# 📥 Load all saved models
# ==========================
svm_model = joblib.load("svm_task_classifier.joblib")
task_label_encoder = joblib.load("svm_label_encoder.joblib")

priority_model = pickle.load(open("priority_xgboost (1).pkl", "rb"))
priority_vectorizer = pickle.load(open("priority_tfidf_vectorizer (1).pkl", "rb"))
priority_label_encoder = pickle.load(open("priority_label_encoder (1).pkl", "rb"))

# ✅ Load dataset to use for assignment
df = pd.read_csv("final_task_dataset_balanced.csv")
latest_users = df[["assigned_user", "user_current_load"]].dropna().drop_duplicates()

# ==========================
# 🖥️ Streamlit UI
# ==========================
st.title("🤖 AI Task Classifier, Prioritizer & Assigner")
st.write("Enter a task description to classify category, predict priority, and assign it to a user:")

task_input = st.text_area("📝 Task Description")

if st.button("🚀 Predict"):
    if not task_input.strip():
        st.warning("Please enter a task description.")
    else:
        # -----------------------------
        # ✅ Category Prediction (SVM)
        # -----------------------------
        category_vector = priority_vectorizer.transform([task_input])  # same TF-IDF can be reused
        category_encoded = svm_model.predict(category_vector)[0]
        predicted_category = task_label_encoder.inverse_transform([category_encoded])[0]

        # -----------------------------
        # 🔺 Priority Prediction (XGBoost)
        # -----------------------------
        priority_vector = priority_vectorizer.transform([task_input])
        priority_encoded = priority_model.predict(priority_vector)[0]
        predicted_priority = priority_label_encoder.inverse_transform([priority_encoded])[0]

        # -----------------------------
        # 👤 User Assignment (based on load)
        # -----------------------------
        assigned_user = latest_users.loc[latest_users["user_current_load"].idxmin(), "assigned_user"]

        # -----------------------------
        # ✅ Output
        # -----------------------------
        st.success(f"📌 **Predicted Category:** `{predicted_category}`")
        st.info(f"⏫ **Predicted Priority:** `{predicted_priority}`")
        st.success(f"👤 **Assigned To:** `{assigned_user}`")
