import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
from preprocessing import preprocess_text

def train_model(X_train, X_test, y_train, y_test):
    """
    Train a phishing email classification model using the provided data.
    
    Args:
        X_train: Training text data
        X_test: Testing text data
        y_train: Training labels
        y_test: Testing labels
        
    Returns:
        model: Trained model
        vectorizer: Fitted TF-IDF vectorizer
        metrics: Dictionary of performance metrics
    """
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=5,
        max_df=0.7,
        ngram_range=(1, 2)
    )
    
    # Transform the text data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train a logistic regression model
    model = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        solver='liblinear',
        random_state=42,
        max_iter=1000
    )
    
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Get feature importance
    feature_importance = np.abs(model.coef_[0])
    feature_names = vectorizer.get_feature_names_out()
    
    # Sort feature importance
    sorted_idx = np.argsort(feature_importance)[::-1]
    top_feature_names = feature_names[sorted_idx][:30]
    top_feature_importance = feature_importance[sorted_idx][:30]
    
    # Store metrics in a dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'confusion_matrix': cm,
        'feature_names': top_feature_names,
        'feature_importance': top_feature_importance
    }
    
    return model, vectorizer, metrics

def predict_email(email_text, model, vectorizer):
    """
    Classify an email as phishing or legitimate.
    
    Args:
        email_text: The preprocessed email text
        model: Trained classification model
        vectorizer: Fitted TF-IDF vectorizer
        
    Returns:
        prediction: 1 for phishing, 0 for legitimate
        confidence: Confidence score as a percentage
        feature_importance: Dictionary of feature importance
    """
    # Transform the email text
    email_tfidf = vectorizer.transform([email_text])
    
    # Make prediction
    prediction = model.predict(email_tfidf)[0]
    
    # Get prediction probabilities
    proba = model.predict_proba(email_tfidf)[0]
    confidence = proba[1] * 100 if prediction == 1 else (1 - proba[1]) * 100
    
    # Extract feature importance
    feature_importance = {}
    
    # Get coefficients from the model (feature importance)
    coefficients = model.coef_[0]
    
    # Get the feature names from the vectorizer
    feature_names = vectorizer.get_feature_names_out()
    
    # Get the features present in the email
    email_features = vectorizer.transform([email_text])
    
    # Get indices of non-zero features
    non_zero_indices = email_features.nonzero()[1]
    
    # For each feature in the email, calculate its contribution to the prediction
    for idx in non_zero_indices:
        feature_name = feature_names[idx]
        feature_value = email_features[0, idx]
        importance = coefficients[idx] * feature_value
        feature_importance[feature_name] = importance
    
    # Sort feature importance by absolute value
    feature_importance = {k: v for k, v in sorted(
        feature_importance.items(), 
        key=lambda item: abs(item[1]), 
        reverse=True
    )[:10]}
    
    return prediction, confidence, feature_importance
