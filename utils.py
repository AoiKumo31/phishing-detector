import re
import joblib
import email
from email import policy
from email.parser import BytesParser, Parser

def save_model(model, vectorizer, filename='phishing_detector_model.pkl'):
    """
    Save the trained model and vectorizer to a file.
    
    Args:
        model: Trained model
        vectorizer: Fitted TF-IDF vectorizer
        filename: Name of the file to save the model to
    """
    model_data = {
        'model': model,
        'vectorizer': vectorizer
    }
    joblib.dump(model_data, filename)

def load_model(filename='phishing_detector_model.pkl'):
    """
    Load a trained model and vectorizer from a file.
    
    Args:
        filename: Name of the file to load the model from
        
    Returns:
        model: Trained model
        vectorizer: Fitted TF-IDF vectorizer
    """
    try:
        model_data = joblib.load(filename)
        return model_data['model'], model_data['vectorizer']
    except:
        return None, None

def extract_email_parts(raw_email):
    """
    Extract the subject and body from an email.
    
    Args:
        raw_email: Raw email text
        
    Returns:
        subject: Email subject
        body: Email body
    """
    # Check if the email is in raw format (with headers)
    if raw_email.startswith('From:') or raw_email.startswith('To:') or raw_email.startswith('Subject:') or '\nFrom:' in raw_email:
        try:
            # Try parsing as a proper email
            msg = Parser().parsestr(raw_email)
            subject = msg.get('Subject', '')
            
            # Extract body
            if msg.is_multipart():
                body = ""
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
            else:
                body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
        except Exception:
            # If parsing fails, use regex to extract subject and body
            subject_match = re.search(r'Subject: (.*?)(?:\n\n|\r\n\r\n)', raw_email, re.DOTALL)
            subject = subject_match.group(1).strip() if subject_match else ""
            
            body_match = re.search(r'(?:\n\n|\r\n\r\n)(.*)', raw_email, re.DOTALL)
            body = body_match.group(1).strip() if body_match else raw_email
    else:
        # If it's just the email body without headers
        subject = ""
        body = raw_email
    
    return subject, body

def extract_phishing_indicators(email_text):
    """
    Extract common phishing indicators from an email.
    
    Args:
        email_text: Raw email text
        
    Returns:
        indicators: Dictionary of phishing indicators and their presence
    """
    indicators = {
        'urgent_language': False,
        'misspelled_domains': False,
        'suspicious_links': False,
        'personal_info_request': False,
        'suspicious_attachments': False,
        'poor_grammar': False
    }
    
    # Check for urgent language
    urgent_words = ['urgent', 'immediate', 'alert', 'attention', 'important', 'verify']
    if any(word in email_text.lower() for word in urgent_words):
        indicators['urgent_language'] = True
    
    # Check for suspicious links
    if re.search(r'(https?://\S+)', email_text):
        indicators['suspicious_links'] = True
    
    # Check for personal information requests
    personal_info_words = ['password', 'credit card', 'ssn', 'social security', 'bank account', 'login', 'verify your']
    if any(word in email_text.lower() for word in personal_info_words):
        indicators['personal_info_request'] = True
    
    # Check for suspicious attachments
    if re.search(r'\.(exe|zip|jar|js|vbs|bat|scr|cmd)\b', email_text.lower()):
        indicators['suspicious_attachments'] = True
    
    # Simple check for poor grammar (this is a basic check, could be improved)
    grammar_errors = ['kindly', 'please do the needful', 'your account has been']
    if any(error in email_text.lower() for error in grammar_errors):
        indicators['poor_grammar'] = True
    
    return indicators
