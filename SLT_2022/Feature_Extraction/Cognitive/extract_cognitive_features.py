import spacy
import seaborn as sns
import os
from scipy import stats
import pandas as pd
import numpy as np
from spacy.matcher import Matcher
import statsmodels
from nltk.corpus import stopwords


def uncertainty(text):

    """ Compute the level of certainty of the subjects in delivering the image description during the CTP task.
     text: txt file containing speech transcripts."""

    cont_con = 0
    if "?" in text:
        cont_con = cont_con + 1
    if "why" in text:
        cont_con = cont_con + 1
    if "might" in text:
        cont_con = cont_con + 1
    if "could" in text:
        cont_con = cont_con + 1
    if "may" in text:
        cont_con = cont_con + 1
    if "uhm" in text:
        cont_con = cont_con + 1
    if "ah" in text:
        cont_con = cont_con + 1
    if "perhaps" in text:
        cont_con = cont_con + 1
    if "looks like" in text:
        cont_con = cont_con + 1

    return cont_con


def repetitions(text):

    """ Compute the number of repetitions of content words. Stop words are removed.
     text: txt file containing speech transcripts."""

    stopwords = list(stopwords.words('english'))
    repetition = 0
    text = text.split()
    d = dict()

    for line in text:
        line = line.strip()
        line = line.lower()
        words = line.split(" ")
        for word in words:

            if word in d:
                d[word] = d[word] + 1
            else:
                d[word] = 1

    for key in list(d.keys()):
        if key not in stopwords:
            if d[key] > 1:
                repetition += 1

    return repetition


def informational_verb(text):

    """ Compute the informativeness of the narratives by
    counting how many (if any) salient events (verbs) are mentioned.
    text: txt file containing speech transcripts.
    """

    cont_con = 0

    if "washing" in text:
        cont_con = cont_con + 1
    if "overflowing" in text:
        cont_con = cont_con + 1
    if "hanging" in text:
        cont_con = cont_con + 1
    if "trying to help" in text:
        cont_con = cont_con + 1
    if "falling" in text:
        cont_con = cont_con + 1
    if "wobbling" in text:
        cont_con = cont_con + 1
    if "drying" in text:
        cont_con = cont_con + 1
    if "ignoring" in text:
        cont_con = cont_con + 1
    if "reaching" in text:
        cont_con = cont_con + 1
    if "reaching up" in text:
        cont_con = cont_con + 1
    if "asking for cookie" in text:
        cont_con = cont_con + 1
    if "laughing" in text:
        cont_con = cont_con + 1
    if "standing" in text:
        cont_con = cont_con + 1

    return cont_con


def informational_content(text):

    """ Compute the informativeness of the narratives by
    counting how many (if any) salient object (nouns) are mentioned"""

    cont_con = 0

    if "mother" in text:
        cont_con = cont_con + 1
    if "sister" in text:
        cont_con = cont_con + 1
    if "cookie" in text:
        cont_con = cont_con + 1
    if "cookie jar" in text:
        cont_con = cont_con + 1
    if "curtains" in text:
        cont_con = cont_con + 1
    if "cabinet" in text:
        cont_con = cont_con + 1
    if "brother" in text:
        cont_con = cont_con + 1
    if "chair" in text:
        cont_con = cont_con + 1
    if "kitchen" in text:
        cont_con = cont_con + 1
    if "sink" in text:
        cont_con = cont_con + 1
    if "garden" in text:
        cont_con = cont_con + 1
    if "fall" in text:
        cont_con = cont_con + 1
    if "dishes" in text:
        cont_con = cont_con + 1
    if "stool" in text:
        cont_con = cont_con + 1
    if "poddle" in text:
        cont_con = cont_con + 1
    if "shoes" in text:
        cont_con = cont_con + 1
    if "apron" in text:
        cont_con = cont_con + 1

    return cont_con







