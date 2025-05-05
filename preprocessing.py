import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Make sure we always download necessary resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """
    Preprocess the email text by cleaning, tokenizing, removing stopwords,
    and lemmatizing.
    
    Args:
        text: Raw email text
        
    Returns:
        processed_text: Clean, preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', 'URL', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', 'EMAIL', text)
    
    # Remove phone numbers
    text = re.sub(r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b', 'PHONE', text)
    
    # Remove numbers
    text = re.sub(r'\d+', 'NUMBER', text)
    
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Rejoin tokens
    processed_text = ' '.join(tokens)
    
    return processed_text

def extract_features(text, vectorizer):
    """
    Extract features from the preprocessed text using TF-IDF vectorization.
    
    Args:
        text: Preprocessed email text
        vectorizer: Fitted TF-IDF vectorizer
        
    Returns:
        features: TF-IDF feature vector
    """
    # Transform the text using the vectorizer
    features = vectorizer.transform([text])
    
    return features
