import re
import string
import spacy
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import contractions
from joblib import delayed, Parallel
import csv

from typing import List

nltk.download('stopwords')

nlp = spacy.load('en_core_web_lg')

file = open("berteley/berteley_stopwords.csv", "r")
data = list(csv.reader(file, delimiter=","))

STOPWORDS = [x[0] for x in data]


def combine_hyphens(doc: str):
    """
    Removes hyphens and concatenates all hyphenated words together

    Parameters
    ----------
    doc
        A string for a single document

    Returns
    -------
        A string with hyphenated words combined


    Examples
    x-ray -> xray


    """

    if not isinstance(doc, str):
        raise TypeError("row_string must be a string")

    row_string = re.sub("-", "", doc)
    return " ".join(row_string.split())
    # return re.sub("(?<=\S)-(?=\S)", "", row_string)


def remove_punctuation(doc: str):
    """
    Removes all punctuation from a string using the punctuation list in the string library

    Parameters
    ----------
    doc
        A single document
    Returns
    -------
    a string with no punctuation
    """

    return doc.translate(str.maketrans('', '', string.punctuation))


def lemmatize(doc):
    """
    This function utilizes the lemmatizer in the spacy package to lemmatize all the words in the list

    Parameters
    ----------
    doc
        A single document

    Returns
    -------
    a string with lemmatized words

    """

    if not isinstance(doc, str):
        raise TypeError("row_string must be a string")

    filt_combined = []
    for word in nlp(doc):
        if word.lemma_ != '-PRON-':
            filt_combined.append(word.lemma_)

    new_df = " ".join(filt_combined)

    return new_df


def remove_stopwords(doc: str, allow_abbrev: bool = True):
    """
    This function utilizes the stopwords in the nltk package to remove the stopwords from the string.
    Further steps were taken that remove words commonly found in scientific articles.

    Parameters
    ----------
    doc
        A single document
    allow_abbrev
        A boolean indicating whether abbreviations should be considered stopwords. If true strings with character length of 2 or less are removed.

    Returns
    -------
    A string with stopwords removed
    """

    if not isinstance(doc, str):
        raise TypeError("row_string must be a string")

    if not isinstance(allow_abbrev, bool):
        raise TypeError("allow_abbrev must be a boolean")

    filt_combined = []

    for word in word_tokenize(doc):
        contains_letter = re.search('[a-zA-Z]', word) is not None
        stopword_check = word.lower() not in stopwords.words('english')
        berteley_stopword_check = word.lower() not in STOPWORDS

        if contains_letter and stopword_check and berteley_stopword_check:
            if not allow_abbrev:
                if len(word.lower()) > 2:
                    filt_combined.append(word)

            else:  # allowing abbreviations
                filt_combined.append(word)

    filtered_ip = " ".join(filt_combined)

    return filtered_ip


def remove_html(doc: str):
    """
    Removes html tags from string
    Parameters
    ----------
    doc
        A single document

    Returns
    -------
    A string with html tags removed
    """
    soup = BeautifulSoup(doc, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text


def remove_extraspace(doc: str):
    """
    Removes excess whitespace from a string
    Parameters
    ----------
    doc
        A single document

    Returns
    -------
    A string with excess whitespace removed
    """
    return ' '.join(doc.split())


def expand_contractions(doc: str):
    """
    Expands contractions using the contractions library

    Parameters
    ----------
    doc
        A single document

    Returns
    -------
    A string with contractions expanded

    Examples
    won't -> will not
    """
    return contractions.fix(doc)


def preprocess(docs: List[str], allow_abbrev: bool = True):
    """
    Wrapper function for all the preprocessing steps

    Parameters
    ----------
    docs
        A list of all the documents
    allow_abbrev
        Boolean indicating whether abbreviations should be allowed. If set to false all strings with length 2 or less will be removed

    Returns
    -------
    A list of strings that have been preprocessed
    """

    # remove all empty strings
    docs = list(filter(len, docs))

    # remove all html tags
    cleaned_docs = [remove_html(s) for s in docs]

    # remove/change numbers

    # expand contractions (don't -> do not)
    cleaned_docs = [expand_contractions(s) for s in cleaned_docs]

    # lower case
    cleaned_docs = [s.lower() for s in cleaned_docs]

    # remove all punctuation (make sure hyphenated words are properly combined
    # and grammatical hyphens are spaced appropriately)
    cleaned_docs = [remove_punctuation(s) for s in cleaned_docs]

    # remove excess white space

    cleaned_docs = [remove_extraspace(s) for s in cleaned_docs]

    # remove empty strings again
    cleaned_docs = list(filter(len, cleaned_docs))

    # lemmatize
    cleaned_docs = [lemmatize(s) for s in cleaned_docs]

    # remove stopwords
    cleaned_docs = [remove_stopwords(s, allow_abbrev=allow_abbrev) for s in cleaned_docs]

    # remove strings with small length
    cleaned_docs = [s for s in cleaned_docs if len(s.split()) > 10]

    return cleaned_docs


def preprocess_parallel(docs: List[str], n_workers: int = 4, allow_abbrev: bool = True):
    """
    Parallelizes the preprocessing by splitting the documents evenly amongst the n_workers

    Parameters
    ----------
    docs
        List of documents
    n_workers
        Number of workers to be assigned preprocessing in joblib

    allow_abbrev
        Boolean indicating whether abbreviations should be allowed. If set to false all strings with length
        2 or less will be removed


    Returns
    -------
    A list of strings that have been preprocessed

    """
    # remove all empty strings
    docs = list(filter(len, docs))

    # split the docs equally among the workers
    partitions = np.array_split(docs, n_workers)

    # docs must be a list of strings
    partitions = [list(p) for p in partitions]

    clean_docs = Parallel(n_jobs=n_workers)(delayed(preprocess)(p, allow_abbrev) for p in partitions)
    # flatten the list of lists into a single list
    clean_docs = [item for sublist in clean_docs for item in sublist]

    return clean_docs
