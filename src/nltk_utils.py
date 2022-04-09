import nltk
from nltk.stem import porter
import numpy as np

nltk.download('punkt')
stemmer = porter.PorterStemmer()

def tokenize(text):
    """
    Take a string and tokenizes the text.
    """
    return nltk.word_tokenize(text)

def stem(token):
    """
    Takes a string and returns a lower case stem..
    """
    return stemmer.stem(token.lower())

def bag_of_words(tokenizedSentence, allWords):
    """
    Takes tokenized sentence and a list of strings as inputs. 
    Returns a list of binary values denoting the occurence of a word in a given token.
    """
    tokenizedSentence = [stem(word) for word in tokenizedSentence]
    bag = np.zeros(len(allWords), dtype=np.float32)
    for idx, word in enumerate(allWords):
        if word in tokenizedSentence:
            bag[idx] = 1.0    
    return bag
    