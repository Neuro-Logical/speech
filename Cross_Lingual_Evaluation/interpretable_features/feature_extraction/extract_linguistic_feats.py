from nltk.corpus import stopwords
import spacy
from spacy.matcher import Matcher

# Load the pretrained pipeline for english

nlp = spacy.load('en_core_web_sm')

#Model for the other languages:

# nlp = spacy.load("es_core_news_sm") --> spanish
# nlp = spacy.load("de_core_news_sm") --> german
# Create a function to preprocess the text

# List of stopwords for english
stopwords = list(stopwords.words('english'))

# For the other languages:
#stopwords = list(stopwords.words('spanish'))
#stopwords = list(stopwords.words('german'))
##stopwords = list(stopwords.words('italian'))

def preprocess(text):
    """This function performs tokenization, lemmatization, removal of non-alphabetic characters
    and stopword removal"""
    # Create Doc object
    doc = nlp(text, disable=['ner'])
    # Generate lemmas
    lemmas = [token.lemma_ for token in doc]
    # Remove stopwords and non-alphabetic characters
    a_lemmas = [lemma for lemma in lemmas
                if lemma.isalpha() and lemma not in stopwords]
    return ' '.join(a_lemmas)


def count_words(string):
    """This function returns the number of words in a string"""
    # Split the string into words
    words = string.split()
    # Return the number of words
    return len(words)


def word_length(string):
    """This function returns the average word length in characters for the words in an item"""
    # Get the length of the full text in characters
    chars = len(string)
    # Split the string into words
    words = string.split()
    # Compute the average word length and round the output to the second decimal point
    if len(words) != 0:
        avg_word_length = chars / len(words)

        return round(avg_word_length, 2)


def sentence_counter(text):
    """This function returns the number of sentences in an item"""
    doc = nlp(text)
    # Initialize a counter variable
    counter = 0
    # Update the counter for each sentence which can be found in the doc.sents object returned by the Spacy model
    for sentence in doc.sents:
        counter = counter + 1
    return counter


# Note that this function is applied to the raw text in order to identify sentence boundaries

def avg_sent_length(text):
    """This function returns the average sentence length in words."""

    doc = nlp(text)
    # Initialize a counter variable
    sent_number = 0
    # Update the counter for each sentence which can be found in the doc.sents object returned by the Spacy model
    for sent in doc.sents:
        sent_number = sent_number + 1
    # Get the number of words
    words = text.split()
    # Compute the average sentence length and round it to the second decimal point
    avg_sent_length = len(words) / sent_number
    return round(avg_sent_length, 2)


def nouns(text, model=nlp):
    """ This function returns the number of nouns in an item. """
    # Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]
    # Return number of nouns
    return pos.count('NOUN')


def verbs(text, model=nlp):
    """This function returns the number of verbs in an item"""
    # Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]
    # Return number of verbs
    return pos.count('VERB')


def adjectives(text, model=nlp):
    """This function returns the number of adjectives in an item"""
    # Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]
    # Return number of adjectives
    return pos.count('ADJ')


def adverbs(text, model=nlp):
    """This function returns the number of adverbs in an item"""
    # Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]
    # Return number of adverbs

    return pos.count('ADV')


def numeral(text, model=nlp):
    """This function returns the number of numerals (e.g., billion) in an item"""

    # Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]
    # Return number of adverbs
    return pos.count('NUM')


def aux(text, model=nlp):
    """This function returns the number of auxiliary in an item"""
    # Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]
    # Return number of adverbs
    return pos.count('AUX')


def get_nps(text):
    """This is a function that outputs the number of noun phrases in an item"""
    doc = nlp(text)
    NP_count = 0
    for np in doc.noun_chunks:
        NP_count = NP_count + 1
    return NP_count
    # print(np)


def get_pps(text):
    """This is a function that outputs the number of prepositional phrases in an item"""
    doc = nlp(text)
    pps = 0
    for token in doc:
        # You can try this with other parts of speech for different subtrees.
        if token.pos_ == 'ADP':
            pps = pps + 1

    return pps


pattern = [{'POS': 'VERB', 'OP': '?'},
           {'POS': 'ADV', 'OP': '*'},
           {'POS': 'AUX', 'OP': '*'},
           {'POS': 'VERB', 'OP': '+'}]


def get_vps(text):
    """This function returns the number of verb phrases in an item"""
    doc = nlp(text)
    vps = 0
    # instantiate a Matcher instance
    matcher = Matcher(nlp.vocab)
    matcher.add("Verb phrase", [pattern], on_match=None)  # new syntax of the command
    # call the matcher to find matches
    matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in matches]
    for match in matches:
        vps = vps + 1
    return vps

