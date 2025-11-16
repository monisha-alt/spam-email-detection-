"""
Setup script to download required NLTK data
Run this script once before using the spam detection system
"""

import nltk

def download_nltk_data():
    """Download all required NLTK data."""
    print("Downloading NLTK data...")
    print("=" * 50)
    
    try:
        print("Downloading punkt tokenizer...")
        nltk.download('punkt', quiet=False)
        print("✓ punkt downloaded")
    except Exception as e:
        print(f"✗ Error downloading punkt: {e}")
    
    try:
        print("\nDownloading stopwords...")
        nltk.download('stopwords', quiet=False)
        print("✓ stopwords downloaded")
    except Exception as e:
        print(f"✗ Error downloading stopwords: {e}")
    
    try:
        print("\nDownloading wordnet...")
        nltk.download('wordnet', quiet=False)
        print("✓ wordnet downloaded")
    except Exception as e:
        print(f"✗ Error downloading wordnet: {e}")
    
    try:
        print("\nDownloading averaged_perceptron_tagger...")
        nltk.download('averaged_perceptron_tagger', quiet=False)
        print("✓ averaged_perceptron_tagger downloaded")
    except Exception as e:
        print(f"✗ Error downloading averaged_perceptron_tagger: {e}")
    
    print("\n" + "=" * 50)
    print("NLTK data download complete!")
    print("You can now use the spam detection system.")


if __name__ == "__main__":
    download_nltk_data()

