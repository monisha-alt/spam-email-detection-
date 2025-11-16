# Spam Email Detection System

A complete, production-ready Python project for detecting spam emails using Natural Language Processing (NLP) and Machine Learning. This system uses TF-IDF feature extraction and multiple ML algorithms to classify emails as spam or ham (legitimate).

## 📋 Features

- **Comprehensive Text Preprocessing**: HTML tag removal, URL removal, email address removal, tokenization, stopword removal, and lemmatization
- **TF-IDF Feature Extraction**: Converts text to numerical features with n-gram support (1-grams and 2-grams)
- **Multiple ML Models**: 
  - Multinomial Naive Bayes
  - Support Vector Machine (LinearSVC)
  - Logistic Regression
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-score, Confusion Matrix, and Classification Reports
- **Visualizations**: Confusion matrix heatmaps and model comparison charts
- **Production-Ready**: Saved models for easy deployment and prediction

## 📁 Project Structure

```
spam_detection/
├── data/
│   └── emails.csv              # Dataset (CSV format with 'label' and 'text' columns)
├── notebooks/
│   └── Spam_Detection.ipynb   # Jupyter notebook with complete analysis
├── src/
│   ├── preprocess.py           # Text preprocessing module
│   ├── train.py                # Model training script
│   └── predict.py              # Prediction script
├── models/
│   ├── spam_model.pkl          # Trained model (generated after training)
│   ├── tfidf_vectorizer.pkl    # TF-IDF vectorizer (generated after training)
│   ├── preprocessor.pkl        # Text preprocessor (generated after training)
│   ├── confusion_matrices.png  # Confusion matrix visualizations
│   └── model_comparison.png    # Model comparison chart
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Setup Instructions

### 1. Install Python

Ensure you have Python 3.8 or higher installed. Check your version:
```bash
python --version
```

### 2. Install Dependencies

Install all required packages:
```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data

The preprocessing module will automatically download required NLTK data on first run. However, you can also download them manually:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

Or run:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

### 4. Prepare Dataset

Place your dataset in `data/emails.csv` with the following format:

| label | text |
|-------|------|
| spam  | Congratulations! You won a lottery... |
| ham   | Hey, can we meet tomorrow for lunch? |

**Dataset Requirements:**
- CSV format
- Two columns: `label` (spam/ham) and `text` (email body)
- Labels should be either "spam" or "ham" (case-insensitive)

**Recommended Datasets:**
- [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- [Enron Email Dataset](https://www.cs.cmu.edu/~enron/)
- [Spam Assassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/)

## 📊 Usage

### Training Models

Train all models on your dataset:

```bash
python src/train.py --data data/emails.csv
```

**Options:**
- `--data`: Path to dataset CSV file (default: `data/emails.csv`)
- `--max_features`: Maximum number of TF-IDF features (default: 5000)

**Example:**
```bash
python src/train.py --data data/emails.csv --max_features 5000
```

This will:
1. Load and preprocess the data
2. Extract TF-IDF features
3. Train three models (Naive Bayes, SVM, Logistic Regression)
4. Evaluate all models
5. Generate visualizations
6. Save the best model and vectorizer

### Making Predictions

After training, use the prediction script:

```bash
python src/predict.py "Congratulations! You won a lottery..."
```

**Options:**
- `text`: Email text to classify (positional argument)
- `--text`: Email text to classify (alternative)
- `--model`: Path to model file (default: `models/spam_model.pkl`)
- `--vectorizer`: Path to vectorizer file (default: `models/tfidf_vectorizer.pkl`)
- `--preprocessor`: Path to preprocessor file (default: `models/preprocessor.pkl`)
- `--prob`: Show probability scores

**Examples:**
```bash
# Simple prediction
python src/predict.py "Hey, can we meet tomorrow for lunch?"

# Prediction with probabilities
python src/predict.py "URGENT: Claim your prize now!" --prob

# Using named argument
python src/predict.py --text "Free money! Click here now!"
```

### Using Jupyter Notebook

Open and run the complete analysis in Jupyter:

```bash
jupyter notebook notebooks/Spam_Detection.ipynb
```

The notebook includes:
- Exploratory Data Analysis
- Text Preprocessing
- Feature Extraction
- Model Training
- Evaluation Metrics
- Visualizations
- Model Comparison

## 🔧 Module Details

### `preprocess.py`

Text preprocessing module with the following features:
- Lowercase conversion
- HTML tag removal
- URL removal
- Email address removal
- Punctuation removal
- Number removal
- Special character removal
- Tokenization
- Stopword removal
- Lemmatization (with POS tagging)
- Optional stemming

**Usage:**
```python
from src.preprocess import TextPreprocessor

preprocessor = TextPreprocessor()
cleaned_text = preprocessor.preprocess_text("Your email text here...")
```

### `train.py`

Training script that:
- Loads and preprocesses data
- Extracts TF-IDF features
- Trains multiple models
- Evaluates performance
- Generates visualizations
- Saves best model

### `predict.py`

Prediction script that:
- Loads trained model and vectorizer
- Preprocesses input text
- Makes predictions
- Returns "Spam" or "Ham"

## 📈 Model Evaluation

The training script evaluates models using:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown
- **Classification Report**: Per-class metrics

## 📊 Visualizations

The training process generates:
1. **Confusion Matrix Heatmaps**: For each model
2. **Model Comparison Chart**: Bar chart comparing accuracies

Saved in the `models/` directory.

## 🛠️ Customization

### Adjusting Preprocessing

Edit `src/preprocess.py` to modify preprocessing steps:
- Enable/disable stemming
- Add custom cleaning functions
- Modify stopword list

### Changing Model Parameters

Edit `src/train.py` to:
- Adjust TF-IDF parameters (max_features, ngram_range)
- Modify model hyperparameters
- Add new models

### Using Different Models

Add new models in `train.py`:
```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
self.models['RandomForest'] = rf_model
```

## 🐛 Troubleshooting

### NLTK Data Not Found
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

### Import Errors
Ensure you're running scripts from the project root directory:
```bash
cd spam_detection
python src/train.py
```

### Dataset Format Issues
Ensure your CSV has exactly two columns: `label` and `text`. Labels must be "spam" or "ham".

## 📝 Requirements

- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- nltk >= 3.6
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- jupyter >= 1.0.0
- joblib >= 1.0.0

## 📄 License

This project is open source and available for educational and commercial use.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on the project repository.

---

**Note**: This is a production-ready system designed for spam email detection. The models are trained on your dataset and can be deployed for real-world applications.

#   s p a m - e m a i l - d e t e c t i o n -  
 