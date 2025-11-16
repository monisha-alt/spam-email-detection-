# Quick Start Guide

Get started with the Spam Email Detection system in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Download NLTK Data

```bash
python setup_nltk.py
```

Or manually:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

## Step 3: Prepare Your Dataset

Place your CSV file in `data/emails.csv` with columns:
- `label`: "spam" or "ham"
- `text`: Email content

See `data/README.md` for more details.

## Step 4: Train Models

```bash
python src/train.py --data data/emails.csv
```

This will:
- Preprocess your data
- Train 3 models (Naive Bayes, SVM, Logistic Regression)
- Evaluate and compare models
- Save the best model to `models/spam_model.pkl`

## Step 5: Make Predictions

```bash
python src/predict.py "Congratulations! You won a lottery..."
```

Or with probabilities:
```bash
python src/predict.py "Hey, can we meet tomorrow?" --prob
```

## Alternative: Use Jupyter Notebook

```bash
jupyter notebook notebooks/Spam_Detection.ipynb
```

Run all cells to see the complete analysis with visualizations!

## That's It! 🎉

Your spam detection system is ready to use. Check `README.md` for detailed documentation.

