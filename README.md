# 📧 Phishing Email Detector

A Streamlit-based web app that uses Natural Language Processing (NLP) and Machine Learning to detect phishing emails from raw text or CSV uploads.

![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 🚀 Features

- 🔍 **Single Email Analysis**  
  Paste or upload email content for real-time phishing detection.

- 📊 **Bulk Analysis**  
  Upload a CSV file to scan multiple emails at once.

- 📈 **Model Performance**  
  View accuracy, precision, recall, F1-score, and visual metrics.

- 📌 **Confidence & Feature Importance Visualization**  
  Understand why an email was flagged as phishing.

---

## 🧠 Technologies Used

- **Python 3.12+**
- **Streamlit**
- **scikit-learn**
- **NLTK**
- **Pandas / NumPy**
- **Matplotlib / Plotly**

---

## 📁 Project Structure

```plaintext
phishing-detector/
│
├── app.py                  # Main Streamlit app
├── model.py                # ML model training and prediction logic
├── preprocessing.py        # Email text cleaning and preprocessing
├── training_data.py        # Dataset loading and splitting
├── utils.py                # Helper functions
├── visualization.py        # Plotting metrics and importance
├── requirements.txt        # Required Python packages
├── pyproject.toml          # Optional Poetry setup
└── README.md               # This file
```

---

## 📦 Installation

### 🖥️ Local Setup
```bash
git clone https://github.com/AoiKumo31/phishing-detector.git
cd phishing-detector
pip install -r requirements.txt
streamlit run app.py
```

### 🌐 Hosted Version
> Visit: [https://phishing-detector31.streamlit.app](https://phishing-detector31.streamlit.app)

---

## 📂 CSV Format for Bulk Upload

The CSV file should contain at least **one column named `email`**:

```csv
email
"Your account has been suspended. Click here to verify."
"Meeting rescheduled to 3 PM."
...
```

---



## ✅ TODO

- Add model retraining toggle
- Improve feature explanations
- Support `.eml` parsing in bulk mode

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## ✨ Credits

Developed by **Van Tran**  
Built with ♥ using Streamlit and scikit-learn.
