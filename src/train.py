"""
Model Training Module for Spam Email Detection
This module trains and evaluates multiple ML models for spam detection.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib

# Add parent directory to path to import preprocess module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from src.preprocess import TextPreprocessor


class SpamDetectorTrainer:
    """
    A class for training and evaluating spam detection models.
    """
    
    def __init__(self, data_path: str, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        """
        Initialize the trainer.
        
        Args:
            data_path: Path to the CSV dataset
            max_features: Maximum number of features for TF-IDF
            ngram_range: N-gram range for TF-IDF (default: unigrams and bigrams)
        """
        self.data_path = data_path
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        self.models = {}
        self.results = {}
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from CSV file.
        
        Returns:
            DataFrame with 'label' and 'text' columns
        """
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        
        # Check required columns
        if 'label' not in df.columns or 'text' not in df.columns:
            raise ValueError("Dataset must contain 'label' and 'text' columns")
        
        # Standardize label values (spam/ham)
        df['label'] = df['label'].str.lower().str.strip()
        df['label'] = df['label'].map({'spam': 1, 'ham': 0})
        
        # Remove any rows with missing labels
        df = df.dropna(subset=['label', 'text'])
        
        print(f"Loaded {len(df)} samples")
        print(f"Spam: {df['label'].sum()}, Ham: {len(df) - df['label'].sum()}")
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the text data.
        
        Args:
            df: DataFrame with 'text' column
            
        Returns:
            DataFrame with preprocessed 'text' column
        """
        print("Preprocessing text data...")
        df['cleaned_text'] = self.preprocessor.preprocess_batch(df['text'].tolist())
        print("Preprocessing complete!")
        return df
    
    def extract_features(self, X_train: pd.Series, X_test: pd.Series):
        """
        Extract TF-IDF features from text.
        
        Args:
            X_train: Training text data
            X_test: Test text data
            
        Returns:
            Tuple of (X_train_features, X_test_features)
        """
        print("Extracting TF-IDF features...")
        X_train_features = self.vectorizer.fit_transform(X_train)
        X_test_features = self.vectorizer.transform(X_test)
        print(f"Feature extraction complete! Shape: {X_train_features.shape}")
        return X_train_features, X_test_features
    
    def train_models(self, X_train, y_train):
        """
        Train multiple ML models.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("\nTraining models...")
        
        # Multinomial Naive Bayes
        print("Training Multinomial Naive Bayes...")
        nb_model = MultinomialNB()
        nb_model.fit(X_train, y_train)
        self.models['MultinomialNB'] = nb_model
        
        # Support Vector Machine
        print("Training Support Vector Machine (LinearSVC)...")
        svm_model = LinearSVC(random_state=42, max_iter=1000)
        svm_model.fit(X_train, y_train)
        self.models['SVM'] = svm_model
        
        # Logistic Regression
        print("Training Logistic Regression...")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train, y_train)
        self.models['LogisticRegression'] = lr_model
        
        print("All models trained successfully!")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        print("\nEvaluating models...")
        
        for name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Evaluating {name}")
            print(f"{'='*50}")
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'predictions': y_pred
            }
            
            # Print metrics
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
            
            # Confusion matrix
            cm = self.results[name]['confusion_matrix']
            print("\nConfusion Matrix:")
            print(f"                Predicted")
            print(f"              Ham    Spam")
            print(f"Actual Ham    {cm[0][0]:4d}   {cm[0][1]:4d}")
            print(f"      Spam    {cm[1][0]:4d}   {cm[1][1]:4d}")
    
    def plot_confusion_matrices(self, y_test):
        """
        Plot confusion matrix heatmaps for all models.
        
        Args:
            y_test: True labels
        """
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (name, result) in enumerate(self.results.items()):
            cm = result['confusion_matrix']
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'],
                ax=axes[idx]
            )
            axes[idx].set_title(f'{name} Confusion Matrix')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_xlabel('Predicted')
        
        plt.tight_layout()
        plt.savefig('models/confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrices saved to models/confusion_matrices.png")
        plt.close()
    
    def plot_model_comparison(self):
        """
        Create a bar chart comparing model accuracies.
        """
        if not self.results:
            print("No results to plot. Train and evaluate models first.")
            return
        
        models = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accuracies, color=['#3498db', '#e74c3c', '#2ecc71'])
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Model Comparison - Accuracy Scores', fontsize=14, fontweight='bold')
        plt.ylim([0, 1])
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{acc:.4f}',
                ha='center',
                va='bottom',
                fontweight='bold'
            )
        
        plt.tight_layout()
        plt.savefig('models/model_comparison.png', dpi=300, bbox_inches='tight')
        print("Model comparison chart saved to models/model_comparison.png")
        plt.close()
    
    def save_models(self, model_name: str = 'spam_model'):
        """
        Save the best model and vectorizer.
        
        Args:
            model_name: Name for the saved model file
        """
        # Find best model based on accuracy
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_model = self.models[best_model_name]
        
        print(f"\nBest model: {best_model_name} (Accuracy: {self.results[best_model_name]['accuracy']:.4f})")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save model
        model_path = f'models/{model_name}.pkl'
        joblib.dump(best_model, model_path)
        print(f"Model saved to {model_path}")
        
        # Save vectorizer
        vectorizer_path = 'models/tfidf_vectorizer.pkl'
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"Vectorizer saved to {vectorizer_path}")
        
        # Save preprocessor
        preprocessor_path = 'models/preprocessor.pkl'
        joblib.dump(self.preprocessor, preprocessor_path)
        print(f"Preprocessor saved to {preprocessor_path}")
        
        return best_model_name, model_path, vectorizer_path
    
    def run_training_pipeline(self):
        """
        Run the complete training pipeline.
        """
        # Load data
        df = self.load_data()
        
        # Preprocess
        df = self.preprocess_data(df)
        
        # Split data
        X = df['cleaned_text']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Extract features
        X_train_features, X_test_features = self.extract_features(X_train, X_test)
        
        # Train models
        self.train_models(X_train_features, y_train)
        
        # Evaluate models
        self.evaluate_models(X_test_features, y_test)
        
        # Visualizations
        self.plot_confusion_matrices(y_test)
        self.plot_model_comparison()
        
        # Save models
        best_model_name, model_path, vectorizer_path = self.save_models()
        
        return {
            'best_model': best_model_name,
            'model_path': model_path,
            'vectorizer_path': vectorizer_path,
            'results': self.results
        }


def main():
    """
    Main function to run training.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train spam detection models')
    parser.add_argument(
        '--data',
        type=str,
        default='data/emails.csv',
        help='Path to the dataset CSV file'
    )
    parser.add_argument(
        '--max_features',
        type=int,
        default=5000,
        help='Maximum number of TF-IDF features'
    )
    
    args = parser.parse_args()
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Initialize trainer
    trainer = SpamDetectorTrainer(
        data_path=args.data,
        max_features=args.max_features
    )
    
    # Run training pipeline
    results = trainer.run_training_pipeline()
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)


if __name__ == "__main__":
    main()

