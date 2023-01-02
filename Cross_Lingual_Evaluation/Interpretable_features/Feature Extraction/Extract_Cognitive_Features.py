from sklearn import metrics
import spacy
import seaborn as sns
import os
from scipy import stats
import pandas as pd
import numpy as np
from spacy.matcher import Matcher
import statsmodels
from nltk.corpus import stopwords


def informational_verb(text):

    """ Compute the informativeness of the narratives by
    counting how many (if any) salient events (verbs) are mentioned. """

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

