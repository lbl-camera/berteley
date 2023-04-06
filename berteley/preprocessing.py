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

nltk.download('stopwords')

nlp = spacy.load('en_core_web_lg')
#
file = open("berteley/berteley_stopwords.csv", "r")
data = list(csv.reader(file, delimiter=","))

STOPWORDS = [x[0] for x in data]





def combine_hyphens(row_string):
    """
    combines hyphenated words into a single word
    Examples:
    x-ray -> xray
    one-way -> oneway

    Input:
    row_string: single row from a dataframe that contains the text in the corpus, a single string

    Output:
    a string with no hyphenated words
    """

    if not isinstance(row_string, str):
        raise TypeError("row_string must be a string")

    row_string = re.sub("-", "", row_string)
    return " ".join(row_string.split())
    # return re.sub("(?<=\S)-(?=\S)", "", row_string)


def remove_punctuation(doc):
    """
    Removes all punctuation from a string using the punctuation list in the string library

    Input:
    row_string: single row from a dataframe that contains the text in the corpus, a single string

    Output:
    a string with no punctuation
    """

    return doc.translate(str.maketrans('', '', string.punctuation))


def lemmatize(row_string):
    """
    This function utilizes the lemmatizer in the spacy package to lemmatize all the words in the list

    Input:
    row_string: single row from a dataframe that contains the text in the corpus, a single string

    Output:
    a string with lemmatized words
    """

    if not isinstance(row_string, str):
        raise TypeError("row_string must be a string")

    filt_combined = []
    for word in nlp(row_string):
        if word.lemma_ != '-PRON-':
            filt_combined.append(word.lemma_)

    new_df = " ".join(filt_combined)

    return new_df


def remove_stopwords(row_string, allow_abbrev=True):
    """
    This function utilizes the stopwords in the nltk package to remove the stopwords from the string.
    Some further steps were taken for the specific ALS use case including removal of words that do not contain a letter
    and words that are in the supplemental stopword list created after looking at initial outputs from early iterations of the topic model

    Input:
    row_string: single row from a dataframe that contains the text in the corpus, a single string

    Output:
    a string with removed stopwords
    """

    if not isinstance(row_string, str):
        raise TypeError("row_string must be a string")

    if not isinstance(allow_abbrev, bool):
        raise TypeError("allow_abbrev must be a boolean")

    filt_combined = []

    for word in word_tokenize(row_string):
        contains_letter = re.search('[a-zA-Z]', word) is not None
        stopword_check = word.lower() not in stopwords.words('english')
        berteley_stopword_check = word.lower() not in STOPWORDS

        if contains_letter and stopword_check and berteley_stopword_check:
            if not allow_abbrev:
                if len(word.lower()) > 2:
                    filt_combined.append(word)

            else:  # allowing abbreviations
                if word.lower() == "perovskites":
                    filt_combined.append("perovskite")
                else:
                    filt_combined.append(word)

    filtered_ip = " ".join(filt_combined)

    return filtered_ip


def format_dataframe(df):
    """
    The ALS data has some extra fields that are not needed, we only want to keep
    Authors, Pub Year, Research Area, Pub TYpe
    Create a new field 'Combined' which is the concatenation of the Title and Abstract
    Rows with NA or missing values are removed

    Input:
    df: pandas dataframe containing the raw data downloaded from the ALS database

    Output:
    newly formated dataframe with proper fields
    """

    # remove entries that dont have an abstract
    df = df[df['Abstract'].notna()]

    # combined - title + abstract
    df['Combined'] = df['Title'] + " " + df['Abstract']

    # remove blanks
    df = df[df['Combined'] != " "]

    # keep only selected cols
    df_sel = df[['Title', 'Abstract', 'Combined', 'Authors', 'DOI', 'Pub Year', 'Research Area', "Pub TYpe"]]
    df_sel = df_sel.rename(
        columns={'Pub Year': 'Pub_Year', "Research Area": "Research_Area", "Authors": "Authors", "Combined": "Combined",
                 "Pub TYpe": "Pub_Type"})

    combined = list(df_sel['Combined'])

    # remove patterns
    pattern = r'<inf>|</inf>|<sup>|</sup>|inf|/inf'
    comb_clean = []

    for s in combined:
        mod_string = re.sub(pattern, '', str(s))
        comb_clean.append(mod_string)

    # merge back to df
    df_sel['Combined'] = comb_clean

    # filter spurious years
    df_sel = df_sel[df_sel['Pub_Year'] != '12.0.1.2']

    # convert years to int
    df_sel['Pub_Year'] = df_sel['Pub_Year'].astype(str).replace('\.0', '', regex=True).astype(int)

    # if year is 201, that is mistyped fom 2001
    df_sel[df_sel['Pub_Year'] == 201]['Pub_Year'] = 2001

    return df_sel


def remove_html(doc):
    soup = BeautifulSoup(doc, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text


def remove_extraspace(doc):
    return ' '.join(doc.split())


def expand_contractions(doc):
    return contractions.fix(doc)


def preprocess(docs: [str], allow_abbrev=True):
    """
    Wrapper function for all the preprocessing steps

    :param docs a list of strings containing the raw text data
    :param allow_abbrev a boolean indicating whether you
    want abbreviations to be allowed or not. If set to False this will remove strings that have length less than 3.

    returns a list of strings containing text data resulting from the preprocessing steps
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


def parallel_helper(docs, allow_abbrev):
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

    # remove stopwords
    cleaned_docs = [remove_stopwords(s, allow_abbrev) for s in cleaned_docs]

    # lemmatize
    cleaned_docs = [lemmatize(s) for s in cleaned_docs]

    # remove strings with small length
    cleaned_docs = [s for s in cleaned_docs if len(s.split()) > 10]

    return cleaned_docs


def preprocess_parallel(docs, n_workers=4, allow_abbrev=True):
    # remove all empty strings
    docs = list(filter(len, docs))

    # split the docs equally among the workers
    partitions = np.array_split(docs, n_workers)

    # docs must be a list of strings
    partitions = [list(p) for p in partitions]

    clean_docs = Parallel(n_jobs=n_workers)(delayed(parallel_helper)(p, allow_abbrev) for p in partitions)
    # flatten the list of lists into a single list
    clean_docs = [item for sublist in clean_docs for item in sublist]

    return clean_docs
