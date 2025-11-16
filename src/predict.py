"""
Prediction Script for Spam Email Detection
This script loads a trained model and makes predictions on new emails.
"""

import os
import sys
import argparse
import numpy as np
import joblib

# Add parent directory to path to import preprocess module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_model_and_vectorizer(model_path: str = 'models/spam_model.pkl',
                              vectorizer_path: str = 'models/tfidf_vectorizer.pkl',
                              preprocessor_path: str = 'models/preprocessor.pkl'):
    """
    Load the trained model, vectorizer, and preprocessor.
    
    Args:
        model_path: Path to the saved model
        vectorizer_path: Path to the saved vectorizer
        preprocessor_path: Path to the saved preprocessor
        
    Returns:
        Tuple of (model, vectorizer, preprocessor)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    print(f"Loading vectorizer from {vectorizer_path}...")
    vectorizer = joblib.load(vectorizer_path)
    
    print(f"Loading preprocessor from {preprocessor_path}...")
    preprocessor = joblib.load(preprocessor_path)
    
    print("All components loaded successfully!")
    
    return model, vectorizer, preprocessor


def predict_email(email_text: str, model, vectorizer, preprocessor) -> str:
    """
    Predict if an email is spam or ham.
    
    Args:
        email_text: The email text to classify
        model: Trained model
        vectorizer: TF-IDF vectorizer
        preprocessor: Text preprocessor
        
    Returns:
        'Spam' or 'Ham'
    """
    # Preprocess the text
    cleaned_text = preprocessor.preprocess_text(email_text)
    
    # Transform to TF-IDF features
    features = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    # Convert to label
    label = 'Spam' if prediction == 1 else 'Ham'
    
    return label


def predict_with_probability(email_text: str, model, vectorizer, preprocessor):
    """
    Predict if an email is spam or ham with probability scores.
    
    Args:
        email_text: The email text to classify
        model: Trained model
        vectorizer: TF-IDF vectorizer
        preprocessor: Text preprocessor
        
    Returns:
        Tuple of (label, spam_probability, ham_probability)
    """
    # Preprocess the text
    cleaned_text = preprocessor.preprocess_text(email_text)
    
    # Transform to TF-IDF features
    features = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)[0]
        spam_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        ham_prob = probabilities[0]
    else:
        # For models without predict_proba (like SVM), use decision function
        if hasattr(model, 'decision_function'):
            decision = model.decision_function(features)[0]
            spam_prob = 1 / (1 + np.exp(-decision))  # Sigmoid approximation
            ham_prob = 1 - spam_prob
        else:
            spam_prob = 1.0 if prediction == 1 else 0.0
            ham_prob = 1.0 - spam_prob
    
    # Convert to label
    label = 'Spam' if prediction == 1 else 'Ham'
    
    return label, spam_prob, ham_prob


def main():
    """
    Main function for command-line prediction.
    """
    parser = argparse.ArgumentParser(
        description='Predict if an email is spam or ham',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py "Congratulations! You won a lottery..."
  python predict.py "Hey, can we meet tomorrow for lunch?"
  python predict.py --text "URGENT: Claim your prize now!" --prob
        """
    )
    parser.add_argument(
        'text',
        nargs='?',
        type=str,
        help='Email text to classify'
    )
    parser.add_argument(
        '--text',
        type=str,
        dest='text_arg',
        help='Email text to classify (alternative to positional argument)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/spam_model.pkl',
        help='Path to the trained model (default: models/spam_model.pkl)'
    )
    parser.add_argument(
        '--vectorizer',
        type=str,
        default='models/tfidf_vectorizer.pkl',
        help='Path to the vectorizer (default: models/tfidf_vectorizer.pkl)'
    )
    parser.add_argument(
        '--preprocessor',
        type=str,
        default='models/preprocessor.pkl',
        help='Path to the preprocessor (default: models/preprocessor.pkl)'
    )
    parser.add_argument(
        '--prob',
        action='store_true',
        help='Show probability scores'
    )
    
    args = parser.parse_args()
    
    # Get email text from either positional or named argument
    email_text = args.text or args.text_arg
    
    if not email_text:
        parser.error("Please provide email text to classify. Use --help for examples.")
    
    try:
        # Load model and vectorizer
        model, vectorizer, preprocessor = load_model_and_vectorizer(
            model_path=args.model,
            vectorizer_path=args.vectorizer,
            preprocessor_path=args.preprocessor
        )
        
        # Make prediction
        if args.prob:
            label, spam_prob, ham_prob = predict_with_probability(
                email_text, model, vectorizer, preprocessor
            )
            print(f"\nEmail Text: {email_text}")
            print(f"Prediction: {label}")
            print(f"Spam Probability: {spam_prob:.4f}")
            print(f"Ham Probability: {ham_prob:.4f}")
        else:
            label = predict_email(email_text, model, vectorizer, preprocessor)
            print(f"\nEmail Text: {email_text}")
            print(f"Prediction: {label}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

