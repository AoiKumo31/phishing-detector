from pathlib import Path

# Updated app.py content with model training moved into button sections
updated_app_py = """
import streamlit as st
import pandas as pd
import numpy as np
import io
import joblib
import time
import traceback
try:
    st.write("✅ App loaded successfully — Streamlit is alive!")
except Exception as e:
    st.error("🔥 Fatal error during startup:")
    st.code(traceback.format_exc())
from model import train_model, predict_email
from preprocessing import preprocess_text
from utils import load_model, save_model, extract_email_parts
from visualization import plot_feature_importance, plot_confidence_score, plot_metrics
from training_data import get_training_data

st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🔍",
    layout="wide"
)

if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'is_model_trained' not in st.session_state:
    st.session_state.is_model_trained = False

st.title("📧 Phishing Email Detector")
st.markdown("Use NLP and machine learning to identify phishing emails with high accuracy.")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Select a page", ["Email Analysis", "Bulk Analysis", "Model Performance", "About"])

if page == "Email Analysis":
    st.header("Single Email Analysis")
    
    input_method = st.radio("Select input method:", ["Paste Email Text", "Upload Email File"])
    
    email_text = ""
    if input_method == "Paste Email Text":
        email_text = st.text_area("Paste the email content here:", height=250)
    else:
        uploaded_file = st.file_uploader("Upload an email file", type=["txt", "eml"])
        if uploaded_file is not None:
            email_text = uploaded_file.getvalue().decode("utf-8")
    
    if st.button("Analyze Email") and email_text:
        if not st.session_state.is_model_trained:
            with st.spinner("Training the model (this may take a moment)..."):
                X_train, X_test, y_train, y_test = get_training_data()
                model, vectorizer, metrics = train_model(X_train, X_test, y_train, y_test)
                st.session_state.model = model
                st.session_state.vectorizer = vectorizer
                st.session_state.metrics = metrics
                st.session_state.is_model_trained = True

        with st.spinner("Analyzing the email..."):
            try:
                subject, body = extract_email_parts(email_text)
                preprocessed_text = preprocess_text(body)
                prediction, confidence, feature_importance = predict_email(
                    preprocessed_text, 
                    st.session_state.model, 
                    st.session_state.vectorizer
                )
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Email Classification")
                st.error("⚠️ Phishing Email Detected!" if prediction == 1 else "✅ This appears to be a legitimate email.")
                st.metric("Confidence Score", f"{confidence:.2f}%")
                st.subheader("Email Details")
                st.markdown(f"**Subject:** {subject if subject else 'N/A'}")
                st.markdown(f"**Preview:** {body[:150]}...")
            with col2:
                st.subheader("Confidence Visualization")
                plot_confidence_score(confidence, prediction)
                st.subheader("Key Phishing Indicators")
                plot_feature_importance(feature_importance)

elif page == "Bulk Analysis":
    st.header("Bulk Email Analysis")
    st.markdown("Upload a CSV file with email content for bulk analysis.")
    st.markdown("The CSV should have a column named 'email' containing the email text.")
    
    uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_csv is not None:
        try:
            emails_df = pd.read_csv(uploaded_csv)
            if 'email' not in emails_df.columns:
                st.error("The CSV file must contain a column named 'email'")
            else:
                st.success(f"Successfully loaded {len(emails_df)} emails for analysis.")
                
                if st.button("Analyze All Emails"):
                    if not st.session_state.is_model_trained:
                        with st.spinner("Training the model (this may take a moment)..."):
                            X_train, X_test, y_train, y_test = get_training_data()
                            model, vectorizer, metrics = train_model(X_train, X_test, y_train, y_test)
                            st.session_state.model = model
                            st.session_state.vectorizer = vectorizer
                            st.session_state.metrics = metrics
                            st.session_state.is_model_trained = True

                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, row in emails_df.iterrows():
                        progress_bar.progress((i + 1) / len(emails_df))
                        email_text = row['email']
                        subject, body = extract_email_parts(email_text)
                        preprocessed_text = preprocess_text(body)
                        prediction, confidence, _ = predict_email(
                            preprocessed_text, 
                            st.session_state.model, 
                            st.session_state.vectorizer
                        )
                        results.append({
                            'email_id': i,
                            'subject': subject if subject else 'N/A',
                            'prediction': 'Phishing' if prediction == 1 else 'Legitimate',
                            'confidence': confidence
                        })
                    
                    results_df = pd.DataFrame(results)
                    st.subheader("Analysis Results")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Emails", len(results_df))
                    col2.metric("Phishing Emails", len(results_df[results_df['prediction'] == 'Phishing']))
                    col3.metric("Legitimate Emails", len(results_df[results_df['prediction'] == 'Legitimate']))
                    
                    st.dataframe(results_df)
                    csv = results_df.to_csv(index=False)
                    st.download_button("Download Results as CSV", csv, "phishing_analysis_results.csv", "text/csv", key='download-csv')

        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")

elif page == "Model Performance":
    st.header("Model Performance Metrics")
    if not st.session_state.is_model_trained:
        with st.spinner("Training the model (this may take a moment)..."):
            X_train, X_test, y_train, y_test = get_training_data()
            model, vectorizer, metrics = train_model(X_train, X_test, y_train, y_test)
            st.session_state.model = model
            st.session_state.vectorizer = vectorizer
            st.session_state.metrics = metrics
            st.session_state.is_model_trained = True

    if st.session_state.metrics:
        metrics = st.session_state.metrics
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Key Metrics")
            st.metric("Accuracy", f"{metrics['accuracy']:.2f}")
            st.metric("Precision", f"{metrics['precision']:.2f}")
            st.metric("Recall", f"{metrics['recall']:.2f}")
            st.metric("F1 Score", f"{metrics['f1']:.2f}")
        with col2:
            st.subheader("Metrics Visualization")
            plot_metrics(metrics)
        st.subheader("Confusion Matrix")
        plot_metrics(metrics, plot_type='confusion_matrix')
        st.subheader("ROC Curve")
        plot_metrics(metrics, plot_type='roc_curve')
        if 'feature_names' in metrics and 'feature_importance' in metrics:
            st.subheader("Top Features for Classification")
            plot_feature_importance({
                name: importance 
                for name, importance in zip(metrics['feature_names'], metrics['feature_importance'])
            })
    else:
        st.warning("Model has not been trained yet. Please use the Email Analysis or Bulk Analysis page first.")

else:
    st.header("About this Application")
    st.markdown(\"\"\"
    ### Phishing Email Detector
    This application uses NLP and Machine Learning to identify phishing emails.
    - **Single Email Analysis**
    - **Bulk CSV Upload**
    - **Confidence Visualizations**
    - **Model Metrics**
    \"\"\")

st.markdown("---")
st.markdown("💡 Phishing Email Detector | Built with Streamlit, scikit-learn, and NLTK")
st.markdown("Developed by: Van Tran")
"""

# Save the updated app.py
path = Path("/mnt/data/app_fixed.py")
path.write_text(updated_app_py)

path.name
