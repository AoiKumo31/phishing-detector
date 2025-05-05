import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confidence_score(confidence, prediction):
    """
    Plot a gauge chart showing the confidence score of the prediction.
    
    Args:
        confidence: Confidence score as a percentage
        prediction: Model prediction (1 for phishing, 0 for legitimate)
    """
    if prediction == 1:
        title = "Phishing Probability"
        color = "red"
    else:
        title = "Legitimate Probability"
        color = "green"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={"text": title},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": color},
               "steps": [
                   {"range": [0, 25], "color": "lightgray"},
                   {"range": [25, 50], "color": "gray"},
                   {"range": [50, 75], "color": "lightgray"},
                   {"range": [75, 100], "color": "gray"}
               ]
              }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

def plot_feature_importance(feature_importance):
    """
    Plot a bar chart showing the importance of features in the prediction.
    
    Args:
        feature_importance: Dictionary of feature names and their importance values
    """
    if not feature_importance:
        st.write("No significant features found.")
        return
    
    # Sort features by absolute importance
    sorted_features = {k: v for k, v in sorted(
        feature_importance.items(), 
        key=lambda item: abs(item[1]), 
        reverse=True
    )[:10]}  # Show top 10 features
    
    # Create a DataFrame for the chart
    df = pd.DataFrame({
        'Feature': list(sorted_features.keys()),
        'Importance': list(sorted_features.values())
    })
    
    # Determine colors based on importance value (positive = red, negative = green)
    colors = ['red' if x > 0 else 'green' for x in df['Importance']]
    
    # Create the chart
    fig = px.bar(
        df,
        x='Importance',
        y='Feature',
        orientation='h',
        color_discrete_sequence=colors,
        labels={'Importance': 'Impact on Phishing Score', 'Feature': 'Email Characteristic'}
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanations
    st.markdown("**Feature Interpretation:**")
    st.markdown("- **Red bars**: Features indicating potential phishing")
    st.markdown("- **Green bars**: Features indicating legitimate email")
    st.markdown("- Longer bars have more influence on the classification")

def plot_metrics(metrics, plot_type='metrics'):
    """
    Plot model performance metrics.
    
    Args:
        metrics: Dictionary of model performance metrics
        plot_type: Type of plot to generate ('metrics', 'confusion_matrix', or 'roc_curve')
    """
    if plot_type == 'metrics':
        # Create a radar chart of performance metrics
        categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        values = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1']
        ]
        
        # Complete the loop for the radar chart
        categories = categories + [categories[0]]
        values = values + [values[0]]
        
        fig = go.Figure(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Model Performance'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == 'confusion_matrix':
        # Create confusion matrix
        cm = metrics['confusion_matrix']
        
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legitimate', 'Phishing'])
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Explain the confusion matrix
        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""
        **Confusion Matrix Explanation:**
        - **True Negatives ({tn})**: Legitimate emails correctly classified as legitimate
        - **False Positives ({fp})**: Legitimate emails incorrectly classified as phishing
        - **False Negatives ({fn})**: Phishing emails incorrectly classified as legitimate
        - **True Positives ({tp})**: Phishing emails correctly classified as phishing
        """)
    
    elif plot_type == 'roc_curve':
        # Create ROC curve
        fpr = metrics['fpr']
        tpr = metrics['tpr']
        roc_auc = metrics['roc_auc']
        
        fig = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC = {roc_auc:.4f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=700, height=500
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        
        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain')
        )
        
        st.plotly_chart(fig, use_container_width=True)
