from berteley import preprocessing
import pytest


def test_combine_hyphens():
    string1 = "-this is A test string"
    string2 = "this is A test string-"
    string3 = "this-is A test string"
    string4 = "this is - A test string"
    string5 = " - this is A test string"
    string6 = "this is A test string - "
    string7 = "this is A- test string"
    string8 = "this is -A test string"
    hyphen_string = "this is A test string"

    assert preprocessing.combine_hyphens(string1) == hyphen_string
    assert preprocessing.combine_hyphens(string2) == hyphen_string
    assert preprocessing.combine_hyphens(string3) == 'thisis A test string'
    assert preprocessing.combine_hyphens(string4) == hyphen_string
    assert preprocessing.combine_hyphens(string5) == hyphen_string
    assert preprocessing.combine_hyphens(string6) == hyphen_string
    assert preprocessing.combine_hyphens(string7) == hyphen_string
    assert preprocessing.combine_hyphens(string8) == hyphen_string


def test_remove_punctuation():
    string1 = "I'm a test string!"
    string2 = "I,m a ! test_ string"
    assert preprocessing.remove_punctuation(string1) == "Im a test string"
    assert preprocessing.remove_punctuation(string2) == "Im a  test string"

    string3 = 'x-ray'
    assert preprocessing.remove_punctuation(string3) == "xray"


def test_lemmatize():
    string1 = "there are a lot of worlds in the galaxy"
    assert preprocessing.lemmatize(string1) == 'there be a lot of world in the galaxy'


def test_remove_stop_df():
    string1 = "iii ion light advanced words ab"
    assert preprocessing.remove_stop_df(string1, allow_abbrev=True) == 'ion words ab'
    assert preprocessing.remove_stop_df(string1, allow_abbrev=False) == 'ion words'


def test_input_errors():
    with pytest.raises(TypeError):
        preprocessing.combine_hyphens(123)
    with pytest.raises(AttributeError):
        preprocessing.remove_punct_df(123)
    with pytest.raises(TypeError):
        preprocessing.lemmatize(123)
    with pytest.raises(TypeError):
        preprocessing.remove_stop_df(123)
    with pytest.raises(TypeError):
        preprocessing.remove_stop_df("abc", "True")

def test_expand_contractions():
    string1 = "I\'ll be there within 5 minutes"
    assert preprocessing.expand_contractions(string1) == "I will be there within 5 minutes"

def test_remove_extraspace():
    string1 = "\n\ni enjoy   going to the movies   with my friends\n"
    assert preprocessing.remove_extraspace(string1) == "i enjoy going to the movies with my friends"


def test_remove_html():
    string1 = "\n\ni enjoy <p> going to the movies </p> with my friends\n"
    assert preprocessing.remove_html(string1) == "\n\ni enjoy   going to the movies   with my friends\n"
