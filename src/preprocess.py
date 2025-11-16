
"""
Text Preprocessing Module for Spam Email Detection
This module contains functions for cleaning and preprocessing email text data.
"""

import re
import string
from typing import List, Optional

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


class TextPreprocessor:
    """
    A class for preprocessing text data for spam detection.
    """
    
    def __init__(self, use_stemming: bool = False):
        """
        Initialize the preprocessor.
        
        Args:
            use_stemming: If True, use stemming in addition to lemmatization
        """
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer() if use_stemming else None
        self.use_stemming = use_stemming
    
    def get_wordnet_pos(self, tag: str) -> str:
        """
        Map POS tag to first character lemmatize() accepts.
        
        Args:
            tag: Part-of-speech tag from NLTK
            
        Returns:
            WordNet POS tag
        """
        tag_dict = {
            'J': wordnet.ADJ,
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV
        }
        return tag_dict.get(tag[0], wordnet.NOUN)
    
    def remove_html_tags(self, text: str) -> str:
        """
        Remove HTML tags from text.
        
        Args:
            text: Input text string
            
        Returns:
            Text without HTML tags
        """
        html_pattern = re.compile('<.*?>')
        return html_pattern.sub('', text)
    
    def remove_urls(self, text: str) -> str:
        """
        Remove URLs from text.
        
        Args:
            text: Input text string
            
        Returns:
            Text without URLs
        """
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        return url_pattern.sub('', text)
    
    def remove_email_addresses(self, text: str) -> str:
        """
        Remove email addresses from text.
        
        Args:
            text: Input text string
            
        Returns:
            Text without email addresses
        """
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        return email_pattern.sub('', text)
    
    def remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation from text.
        
        Args:
            text: Input text string
            
        Returns:
            Text without punctuation
        """
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def remove_numbers(self, text: str) -> str:
        """
        Remove numbers from text.
        
        Args:
            text: Input text string
            
        Returns:
            Text without numbers
        """
        return re.sub(r'\d+', '', text)
    
    def remove_special_characters(self, text: str) -> str:
        """
        Remove special characters from text, keeping only alphanumeric and spaces.
        
        Args:
            text: Input text string
            
        Returns:
            Text without special characters
        """
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete text preprocessing pipeline.
        
        Args:
            text: Raw text string
            
        Returns:
            Preprocessed text string
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = self.remove_html_tags(text)
        
        # Remove URLs
        text = self.remove_urls(text)
        
        # Remove email addresses
        text = self.remove_email_addresses(text)
        
        # Remove numbers
        text = self.remove_numbers(text)
        
        # Remove special characters
        text = self.remove_special_characters(text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # POS tagging for better lemmatization
        pos_tags = pos_tag(tokens)
        
        # Lemmatization with POS tags
        lemmatized_tokens = []
        for token, tag in pos_tags:
            pos = self.get_wordnet_pos(tag)
            lemma = self.lemmatizer.lemmatize(token, pos)
            lemmatized_tokens.append(lemma)
        
        # Optional: Apply stemming
        if self.use_stemming and self.stemmer:
            lemmatized_tokens = [self.stemmer.stem(token) for token in lemmatized_tokens]
        
        # Join tokens back into a string
        preprocessed_text = ' '.join(lemmatized_tokens)
        
        return preprocessed_text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of preprocessed text strings
        """
        return [self.preprocess_text(text) for text in texts]


def preprocess_text(text: str, use_stemming: bool = False) -> str:
    """
    Convenience function for preprocessing a single text.
    
    Args:
        text: Raw text string
        use_stemming: If True, use stemming in addition to lemmatization
        
    Returns:
        Preprocessed text string
    """
    preprocessor = TextPreprocessor(use_stemming=use_stemming)
    return preprocessor.preprocess_text(text)


def preprocess_batch(texts: List[str], use_stemming: bool = False) -> List[str]:
    """
    Convenience function for preprocessing a batch of texts.
    
    Args:
        texts: List of text strings
        use_stemming: If True, use stemming in addition to lemmatization
        
    Returns:
        List of preprocessed text strings
    """
    preprocessor = TextPreprocessor(use_stemming=use_stemming)
    return preprocessor.preprocess_batch(texts)


if __name__ == "__main__":
    # Example usage
    sample_text = "Congratulations! You won a $1000 lottery. Visit http://example.com/claim or email us at winner@example.com <html>Click here</html>"
    
    preprocessor = TextPreprocessor()
    cleaned = preprocessor.preprocess_text(sample_text)
    print("Original:", sample_text)
    print("Preprocessed:", cleaned)

