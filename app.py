import streamlit as st
import pandas as pd
import numpy as np
import io
import joblib
import time
from model import train_model, predict_email
from preprocessing import preprocess_text 
from utils import load_model, save_model, extract_email_parts
from visualization import plot_feature_importance, plot_confidence_score, plot_metrics
from training_data import get_training_data

# Set page configuration
st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'is_model_trained' not in st.session_state:
    st.session_state.is_model_trained = False

# Main title
st.title("📧 Phishing Email Detector")
st.markdown("Use NLP and machine learning to identify phishing emails with high accuracy.")

# Sidebar for app navigation
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Select a page", ["Email Analysis", "Bulk Analysis", "Model Performance", "About"])

# Email Analysis page
if page == "Email Analysis":
    st.header("Single Email Analysis")
    
    # Check if model is trained, otherwise train it
    if not st.session_state.is_model_trained:
        with st.spinner("Training the model (this may take a moment)..."):
            X_train, X_test, y_train, y_test = get_training_data()
            model, vectorizer, metrics = train_model(X_train, X_test, y_train, y_test)
            st.session_state.model = model
            st.session_state.vectorizer = vectorizer
            st.session_state.metrics = metrics
            st.session_state.is_model_trained = True
    
    # Email input options
    input_method = st.radio("Select input method:", ["Paste Email Text", "Upload Email File"])
    
    email_text = ""
    if input_method == "Paste Email Text":
        email_text = st.text_area("Paste the email content here:", height=250)
    else:
        uploaded_file = st.file_uploader("Upload an email file", type=["txt", "eml"])
        if uploaded_file is not None:
            email_text = uploaded_file.getvalue().decode("utf-8")
    
    if st.button("Analyze Email") and email_text:
        with st.spinner("Analyzing the email..."):
            try:
                # Extract parts and preprocess the email
                st.write("Extracting email parts...")
                subject, body = extract_email_parts(email_text)
                st.write("Preprocessing email text...")
                preprocessed_text = preprocess_text(body)
                
                # Make prediction
                st.write("Making prediction...")
                prediction, confidence, feature_importance = predict_email(
                    preprocessed_text, 
                    st.session_state.model, 
                    st.session_state.vectorizer
                )
                st.write("Prediction completed successfully!")
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Email Classification")
                if prediction == 1:
                    st.error("⚠️ Phishing Email Detected!")
                else:
                    st.success("✅ This appears to be a legitimate email.")
                
                st.metric("Confidence Score", f"{confidence:.2f}%")
                
                # Display subject and snippet
                st.subheader("Email Details")
                st.markdown(f"**Subject:** {subject if subject else 'N/A'}")
                st.markdown(f"**Preview:** {body[:150]}...")
            
            with col2:
                st.subheader("Confidence Visualization")
                plot_confidence_score(confidence, prediction)
                
                st.subheader("Key Phishing Indicators")
                plot_feature_importance(feature_importance)

# Bulk Analysis page
elif page == "Bulk Analysis":
    st.header("Bulk Email Analysis")
    
    # Check if model is trained, otherwise train it
    if not st.session_state.is_model_trained:
        with st.spinner("Training the model (this may take a moment)..."):
            X_train, X_test, y_train, y_test = get_training_data()
            model, vectorizer, metrics = train_model(X_train, X_test, y_train, y_test)
            st.session_state.model = model
            st.session_state.vectorizer = vectorizer
            st.session_state.metrics = metrics
            st.session_state.is_model_trained = True
    
    st.markdown("Upload a CSV file with email content for bulk analysis.")
    st.markdown("The CSV should have a column named 'email' containing the email text.")
    
    uploaded_csv = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_csv is not None:
        try:
            # Load the CSV
            emails_df = pd.read_csv(uploaded_csv)
            
            if 'email' not in emails_df.columns:
                st.error("The CSV file must contain a column named 'email'")
            else:
                st.success(f"Successfully loaded {len(emails_df)} emails for analysis.")
                
                if st.button("Analyze All Emails"):
                    # Prepare for results
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, row in emails_df.iterrows():
                        # Update progress
                        progress_bar.progress((i + 1) / len(emails_df))
                        
                        # Process the email
                        email_text = row['email']
                        subject, body = extract_email_parts(email_text)
                        preprocessed_text = preprocess_text(body)
                        
                        # Predict
                        prediction, confidence, _ = predict_email(
                            preprocessed_text, 
                            st.session_state.model, 
                            st.session_state.vectorizer
                        )
                        
                        # Store results
                        results.append({
                            'email_id': i,
                            'subject': subject if subject else 'N/A',
                            'prediction': 'Phishing' if prediction == 1 else 'Legitimate',
                            'confidence': confidence
                        })
                    
                    # Display results
                    results_df = pd.DataFrame(results)
                    st.subheader("Analysis Results")
                    
                    # Summary statistics
                    total = len(results_df)
                    phishing_count = len(results_df[results_df['prediction'] == 'Phishing'])
                    legitimate_count = total - phishing_count
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Emails", total)
                    col2.metric("Phishing Emails", phishing_count)
                    col3.metric("Legitimate Emails", legitimate_count)
                    
                    # Display the table
                    st.dataframe(results_df)
                    
                    # Download option
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "Download Results as CSV",
                        csv,
                        "phishing_analysis_results.csv",
                        "text/csv",
                        key='download-csv'
                    )
        
        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")

# Model Performance page
elif page == "Model Performance":
    st.header("Model Performance Metrics")
    
    # Check if model is trained, otherwise train it
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
        
        # Display model metrics
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
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        plot_metrics(metrics, plot_type='confusion_matrix')
        
        # ROC curve
        st.subheader("ROC Curve")
        plot_metrics(metrics, plot_type='roc_curve')
        
        # Feature importance
        st.subheader("Top Features for Classification")
        if 'feature_names' in metrics and 'feature_importance' in metrics:
            plot_feature_importance({
                name: importance 
                for name, importance in zip(metrics['feature_names'], metrics['feature_importance'])
            })
    else:
        st.warning("Model has not been trained yet. Please use the Email Analysis or Bulk Analysis page first.")

# About page
else:
    st.header("About this Application")
    
    st.markdown("""
    ### Phishing Email Detector
    
    This application uses Natural Language Processing (NLP) and Machine Learning techniques to identify potential phishing emails with high accuracy.
    
    #### Features:
    - **Single Email Analysis**: Analyze individual emails for phishing attempts
    - **Bulk Analysis**: Process multiple emails at once via CSV upload
    - **Visualization**: View the key indicators that influence the classification
    - **Performance Metrics**: Monitor the model's accuracy and other performance statistics
    
    #### How it works:
    1. Email text is preprocessed using NLP techniques (tokenization, removing stopwords, etc.)
    2. Features are extracted using TF-IDF vectorization
    3. A trained Logistic Regression model analyzes the features to classify the email
    4. Results are displayed with confidence scores and visualizations
    
    #### Common Phishing Indicators:
    - Urgent calls to action
    - Requests for personal information
    - Suspicious links or attachments
    - Grammar and spelling errors
    - Impersonation of known organizations
    - Generic greetings
    
    This tool is designed to assist in identifying potential threats, but should not replace human judgment and security best practices.
    """)

# Footer
st.markdown("---")
st.markdown("💡 Phishing Email Detector | Built with Streamlit, scikit-learn, and NLTK")
st.markdown("Developed by: Van Tran")
