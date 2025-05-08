import numpy as np
import nltk
import logging
from nltk.stem.porter import PorterStemmer
from typing import List, Union, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize stemmer
stemmer = PorterStemmer()
# Cache for stemmed words
stem_cache = {}

def tokenize(sentence: Union[str, bytes]) -> List[str]:
    """
    Split a sentence into an array of words/tokens.
    
    Args:
        sentence (Union[str, bytes]): Input sentence to tokenize.
    
    Returns:
        List[str]: List of tokens (words, punctuation, or numbers).
    
    Raises:
        TypeError: If input is not a string or bytes.
        ValueError: If input is empty or contains only whitespace.
    
    Example:
        >>> tokenize("Hello, how are you?")
        ['Hello', ',', 'how', 'are', 'you', '?']
    """
    if not isinstance(sentence, (str, bytes)):
        logging.error(f"Invalid input type for tokenize: {type(sentence)}")
        raise TypeError("Input must be a string or bytes")
    
    if isinstance(sentence, bytes):
        sentence = sentence.decode('utf-8')
    
    sentence = sentence.strip()
    if not sentence:
        logging.error("Empty or whitespace-only input for tokenize")
        raise ValueError("Input cannot be empty or whitespace-only")
    
    try:
        tokens = nltk.word_tokenize(sentence)
        logging.debug(f"Tokenized '{sentence}' into {tokens}")
        return tokens
    except Exception as e:
        logging.error(f"Error tokenizing sentence '{sentence}': {str(e)}")
        raise

def stem(word: str) -> str:
    """
    Find the root form of a word using Porter Stemmer.
    
    Args:
        word (str): Input word to stem.
    
    Returns:
        str: Stemmed word (lowercase).
    
    Raises:
        TypeError: If input is not a string.
    
    Example:
        >>> stem("organizing")
        'organ'
    """
    if not isinstance(word, str):
        logging.error(f"Invalid input type for stem: {type(word)}")
        raise TypeError("Input must be a string")
    
    word = word.lower()
    if word in stem_cache:
        return stem_cache[word]
    
    try:
        stemmed = stemmer.stem(word)
        stem_cache[word] = stemmed
        logging.debug(f"Stemmed '{word}' to '{stemmed}'")
        return stemmed
    except Exception as e:
        logging.error(f"Error stemming word '{word}': {str(e)}")
        raise

def bag_of_words(tokenized_sentence: List[str], words: List[str]) -> np.ndarray:
    """
    Create a bag-of-words array for a tokenized sentence.
    
    Args:
        tokenized_sentence (List[str]): List of tokens from a sentence.
        words (List[str]): List of unique stemmed words (vocabulary).
    
    Returns:
        np.ndarray: Bag-of-words array (1 for words present, 0 otherwise).
    
    Raises:
        TypeError: If inputs are not lists of strings.
        ValueError: If tokenized_sentence or words is empty.
    
    Example:
        >>> bag_of_words(["hello", "how"], ["hi", "hello", "you"])
        array([0., 1., 0.], dtype=float32)
    """
    if not isinstance(tokenized_sentence, list) or not isinstance(words, list):
        logging.error("Inputs to bag_of_words must be lists")
        raise TypeError("Inputs must be lists")
    if not tokenized_sentence or not words:
        logging.error("Tokenized sentence or words list cannot be empty")
        raise ValueError("Tokenized sentence or words list cannot be empty")
    if not all(isinstance(w, str) for w in tokenized_sentence + words):
        logging.error("All elements must be strings")
        raise TypeError("All elements must be strings")

    try:
        # Stem sentence words
        sentence_words = set(stem(word) for word in tokenized_sentence)
        # Initialize bag
        bag = np.zeros(len(words), dtype=np.float32)
        # Set 1 for words present in sentence
        for idx, w in enumerate(words):
            if w in sentence_words:
                bag[idx] = 1
        
        logging.debug(f"Bag of words for {tokenized_sentence}: {bag.tolist()}")
        return bag
    except Exception as e:
        logging.error(f"Error creating bag of words: {str(e)}")
        raise
