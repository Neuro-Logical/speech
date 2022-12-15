from lexicalrichness import LexicalRichness
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
import spacy
from spacy.matcher import Matcher


def compute_lexical_diversity(transcript):
    """Function to compute different metrics for lexical diversity"""

    lex = LexicalRichness(transcript)
    # word_count = lex.words
    unique_word_count = lex.terms
    type_token_ratio = lex.ttr
    # root_type_token_ratio = lex.rttr
    corrected_type_token_ratio = lex.cttr
    # mean_segmental_type_token_ratio = lex.msttr(segment_window=12) #25
    moving_average_type_token_ratio = lex.mattr(window_size=13)  # 25
    # measure_textual_lexical_diversity= lex.mtld(threshold=0.72)
    # hypergeometric_distribution_diversity = lex.hdd(draws=13)
    # herdan_lexical_diversity_measure = lex.Herdan
    summer_lexical_diversity_measure = lex.Summer
    dugast_lexical_diversity_measure = lex.Dugast
    # maas_lexical_diversity_measure = lex.Maas

    return unique_word_count, type_token_ratio, corrected_type_token_ratio, moving_average_type_token_ratio, summer_lexical_diversity_measure, dugast_lexical_diversity_measure


def load_files(data):
    """Apply lexical richness function defined above to a dataframe."""

    speakers = data['idx'].tolist()
    sentences = data['sentence'].tolist()
    labels = data['label'].tolist()
    lex_vals = np.array([compute_lexical_diversity(sent) for sent in sentences])
    names = ["unique_word_count", "type_token_ratio", "corrected_type_token_ratio", "moving_average_type_token_ratio",
             "summer_lexical_diversity_measure", "dugast_lexical_diversity_measure"]
    frame = pd.DataFrame({"speakers": speakers, "labels": labels, "sentences": sentences,
                          **{name: val for name, val in zip(names, lex_vals.T)}})

    return frame


# Load the pretrained pipeline for English

nlp = spacy.load('en_core_web_sm')

# Create a function to preprocess the text

# List of stopwords for english
stopwords = list(stopwords.words('english'))

#stopwords = list(stopwords.words('spanish')) for Spanish
#stopwords = list(stopwords.words('german')) for German
##stopwords = list(stopwords.words('italian')) for Italian

def preprocess(text):
    '''This is a function to perform tokenization, lemmatization, removal of non-alphabetic characters
    and stopword removal'''
    # Create Doc object
    doc = nlp(text, disable=['ner'])
    # Generate lemmas
    lemmas = [token.lemma_ for token in doc]
    # Remove stopwords and non-alphabetic characters
    a_lemmas = [lemma for lemma in lemmas
                if lemma.isalpha() and lemma not in stopwords]
    return ' '.join(a_lemmas)


def count_words(string):
    '''This function returns the number of words in a string'''
    # Split the string into words
    words = string.split()
    # Return the number of words
    return len(words)


def word_length(string):
    '''This function returns the average word length in characters for the words in an item'''
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
    """ This function returns the average sentence length in words."""

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

