from lexicalrichness import LexicalRichness
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
import spacy
from spacy.matcher import Matcher


def compute_lexical_diversity(transcript):

    """Function to compute different metrics for lexical diversity.
    transcript: text file containing speech transcipt."""

    lex = LexicalRichness(transcript)
    # word_count = lex.words
    unique_word_count = lex.terms
    type_token_ratio = lex.ttr
    # root_type_token_ratio = lex.rttr
    corrected_type_token_ratio = lex.cttr
    # mean_segmental_type_token_ratio = lex.msttr(segment_window=12) #25
    moving_average_type_token_ratio = lex.mattr(window_size=13)  # 25
    summer_lexical_diversity_measure = lex.Summer
    dugast_lexical_diversity_measure = lex.Dugast
    # maas_lexical_diversity_measure = lex.Maas

    return unique_word_count, type_token_ratio, corrected_type_token_ratio, moving_average_type_token_ratio, summer_lexical_diversity_measure, dugast_lexical_diversity_measure


def load_files(data):

    """Apply lexical richness function defined above to a dataframe.
    data: pandas data frame having as columns:
    1 - speaker-ID;
    2 - speech transcripts;
    3 - labels indicating the disorder.

    This function returns the same dataframe given as input with lexical richness metrics for each subject, computed from the transcripts."""

    speakers = data['idx'].tolist()
    sentences = data['sentence'].tolist()
    labels = data['label'].tolist()
    lex_vals = np.array([compute_lexical_diversity(sent) for sent in sentences])
    names = ["unique_word_count", "type_token_ratio", "corrected_type_token_ratio", "moving_average_type_token_ratio",
             "summer_lexical_diversity_measure", "dugast_lexical_diversity_measure"]
    frame = pd.DataFrame({"speakers": speakers, "labels": labels, "sentences": sentences, **{name: val for name, val in zip(names, lex_vals.T)}})

    return frame


# Load the pretrained pipeline for english

nlp = spacy.load('en_core_web_sm')

# Create a function to preprocess the text

# List of stopwords for english
stopwords = list(stopwords.words('english'))


def preprocess(text):

    """This is a function to perform tokenization, lemmatization, removal of non-alphabetic characters
    and stopword removal.
    text: text file containing the speech transcripts.
    """
    # Create Doc object
    doc = nlp(text, disable=['ner'])
    # Generate lemmas
    lemmas = [token.lemma_ for token in doc]
    # Remove stopwords and non-alphabetic characters
    a_lemmas = [lemma for lemma in lemmas
                if lemma.isalpha() and lemma not in stopwords]
    return ' '.join(a_lemmas)


def count_words(string):

    """This function returns the number of words in a string.
     text: text file containing the speech transcripts."""
    # Split the string into words
    words = string.split()
    # Return the number of words
    return len(words)


def word_length(string):
    """This function returns the average word length in characters for the words in an item.
     text: text file containing the speech transcripts."""
    # Get the length of the full text in characters
    chars = len(string)
    # Split the string into words
    words = string.split()
    # Compute the average word length and round the output to the second decimal point
    if len(words) != 0:
        avg_word_length = chars / len(words)

        return round(avg_word_length, 2)


def sentence_counter(text):
    """This function returns the number of sentences in an item.
     text: text file containing the speech transcripts."""
    doc = nlp(text)
    # Initialize a counter variable
    counter = 0
    # Update the counter for each sentence which can be found in the doc.sents object returned by the Spacy model
    for sentence in doc.sents:
        counter = counter + 1
    return counter


# Note that this function is applied to the raw text in order to identify sentence boundaries

def avg_sent_length(text):
    """ This function returns the average sentence length in words.
     text: text file containing the speech transcripts."""
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
    """ This function returns the number of nouns in an item.
     text: text file containing the speech transcripts."""
    # Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]
    # Return number of nouns
    return pos.count('NOUN')


def verbs(text, model=nlp):
    """This function returns the number of verbs in an item.
     text: text file containing the speech transcripts."""
    # Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]
    # Return number of verbs
    return pos.count('VERB')


def adjectives(text, model=nlp):
    """This function returns the number of adjectives in an item.
     text: text file containing the speech transcripts."""
    # Create doc object
    doc = model(text)
    # Generate list of POS tags
    pos = [token.pos_ for token in doc]
    # Return number of adjectives
    return pos.count('ADJ')


def adverbs(text, model=nlp):
    """This function returns the number of adverbs in an item.
     text: text file containing the speech transcripts."""
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
    """This is a function that outputs the number of noun phrases in an item.
     text: text file containing the speech transcripts."""
    doc = nlp(text)
    NP_count = 0
    for np in doc.noun_chunks:
        NP_count = NP_count + 1
    return NP_count
    # print(np)


def get_pps(text):
    """This is a function that outputs the number of prepositional phrases in an item.
     text: text file containing the speech transcripts."""
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
    """This function returns the number of verb phrases in an item.
      text: text file containing the speech transcripts."""
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


temporal_connectives = ['afterwards', 'once', 'at this moment', 'at this point', 'before', 'finally',
                        'here', 'in the end', 'lastly', 'later on', 'meanwhile', 'next', 'now',
                        'on another occasion', 'previously', 'since', 'soon', 'straightaway', 'then',
                        'when', 'whenever', 'while']

# Connectives to show cause or conditions
causal_connectives = ['accordingly', 'all the same', 'an effect of', 'an outcome of', 'an upshot of',
                      'as a consequence of', 'as a result of', 'because', 'caused by', 'consequently',
                      'despite this', 'even though', 'hence', 'however', 'in that case', 'moreover',
                      'nevertheless', 'otherwise', 'so', 'so as', 'stemmed from', 'still', 'then',
                      'therefore', 'though', 'under the circumstances', 'yet']

# Connectives for showing results
exemplifying_connectives = ['accordingly', 'as a result', 'as exemplified by', 'consequently', 'for example',
                            'for instance', 'for one thing', 'including', 'provided that', 'since', 'so',
                            'such as', 'then', 'therefore', 'these include', 'through', 'unless', 'without']

# Connectives to show similarity or add a point
additive_connectives = ['and', 'additionally', 'also', 'as well', 'even', 'furthermore', 'in addition', 'indeed',
                        'let alone', 'moreover', 'not only']

# Connectives showing a difference or an opposite point of view
contrastive_connectives = ['alternatively', 'anyway', 'but', 'by contrast', 'differs from', 'elsewhere',
                           'even so', 'however', 'in contrast', 'in fact', 'in other respects', 'in spite of this',
                           'in that respect', 'instead', 'nevertheless', 'on the contrary', 'on the other hand',
                           'rather', 'though', 'whereas', 'yet']


def temporal_connectives_count(text):
    """This function counts the number of temporal connectives in a text.
      text: text file containing the speech transcripts."""
    count = 0
    for string in temporal_connectives:
        for match in re.finditer(string, text):
            count += 1
    return count


def causal_connectives_count(text):
    """This function counts the number of causal connectives in a text.
      text: text file containing the speech transcripts."""
    count = 0
    for string in causal_connectives:
        for match in re.finditer(string, text):
            count += 1
    return count


def exemplifying_connectives_count(text):
    """This function counts the number of exemplifying connectives in a text.
      text: text file containing the speech transcripts."""
    count = 0
    for string in exemplifying_connectives:
        for match in re.finditer(string, text):
            count += 1
    return count


def additive_connectives_count(text):
    """This function counts the number of additive connectives in a text.
      text: text file containing the speech transcripts."""
    count = 0
    for string in additive_connectives:
        for match in re.finditer(string, text):
            count += 1
    return count


def contrastive_connectives_count(text):
    """This function counts the number of contrastive connectives in a text.
      text: text file containing the speech transcripts."""
    cont_con = 0
    for string in contrastive_connectives:
        if string in text:
            cont_con = cont_con + 1
    return cont_con


