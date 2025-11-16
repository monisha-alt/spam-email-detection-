# Dataset Directory

Place your email dataset CSV file here with the name `emails.csv`.

## Dataset Format

The CSV file must have exactly two columns:

1. **label**: Email classification (must be "spam" or "ham")
2. **text**: Email body/content

### Example:

```csv
label,text
spam,Congratulations! You won a $1000 lottery. Claim now!
ham,Hey, can we meet tomorrow for lunch?
spam,URGENT: Click here to claim your prize!
ham,Thanks for the meeting today. See you next week.
```

## Recommended Datasets

1. **SMS Spam Collection Dataset**
   - Download from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
   - Note: May need to rename columns to match format

2. **Enron Email Dataset**
   - Download from: https://www.cs.cmu.edu/~enron/
   - Note: Requires preprocessing to extract labels

3. **Spam Assassin Public Corpus**
   - Download from: https://spamassassin.apache.org/old/publiccorpus/
   - Note: Requires preprocessing to extract labels

## Converting Other Datasets

If your dataset has different column names, you can rename them:

```python
import pandas as pd

# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Rename columns if needed
df = df.rename(columns={'v1': 'label', 'v2': 'text'})  # Example for SMS dataset

# Ensure labels are 'spam' or 'ham'
df['label'] = df['label'].str.lower().str.strip()

# Save in correct format
df[['label', 'text']].to_csv('emails.csv', index=False)
```

