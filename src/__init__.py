"""
Spam Email Detection Package
"""

from .preprocess import TextPreprocessor, preprocess_text, preprocess_batch

__all__ = ['TextPreprocessor', 'preprocess_text', 'preprocess_batch']

