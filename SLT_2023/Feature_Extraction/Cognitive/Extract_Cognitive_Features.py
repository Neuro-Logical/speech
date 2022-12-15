def uncertanty(text):
    """ Function design to capture the level of certainty of patients of participants when delivering  the description
    of the image"""

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

    """ Function design to capture the redundancy in the code.  To operationalize
    redundancy I chose to count the repetitions """

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

    """ Informativeness of the narratives represented by
    counting how many (if any) salient events (verbs) are mentioned"""

    cont_con = 0

    if "washing" in text:
        cont_con = cont_con + 1
    if "wash" in text:
        cont_con = cont_con + 1
    if "overflowing" in text:
        cont_con = cont_con + 1
    if "overflow" in text:
        cont_con = cont_con + 1
    if "hanging" in text:
        cont_con = cont_con + 1
    if "hang" in text:
        cont_con = cont_con + 1
    if "falling" in text:
        cont_con = cont_con + 1
    if "fall" in text:
        cont_con = cont_con + 1
    if "wearing" in text:
        cont_con = cont_con + 1
    if "wear" in text:
        cont_con = cont_con + 1
    if "running" in text:
        cont_con = cont_con + 1
    if "run" in text:
        cont_con = cont_con + 1
    if "drying" in text:
        cont_con = cont_con + 1
    if "dry" in text:
        cont_con = cont_con + 1
    if "paying attention" in text:
        cont_con = cont_con + 1
    if "reaching" in text:
        cont_con = cont_con + 1
    if "reach" in text:
        cont_con = cont_con + 1
    if "tipping" in text:
        cont_con = cont_con + 1
    if "tipp" in text:
        cont_con = cont_con + 1

    return cont_con


def informational_content(text):

    """ Informativeness of the description represented by
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

def ratio_info_rep_plus_uncert(df_):
    """ Ratio between informativeness and uncertanty,
    where uncertainty is represented as repetition + uncertanty"""

    summation = df_['repetition'] + df_["uncertanty"]
    ratio = df_['informational'] / summation  # info / rep + uncertanty
    df_["ratio_info_rep_plus_uncert"] = ratio

    return df_

def ratio_rep_certanty(df_):
    """ Function designed to measure the ratio between repetitions and uncertanty"""

    division = df_['repetition'] / df_["uncertanty"]  # repetition / uncertainty
    df_["ratio_rep_certanty"] = division

    return df_





