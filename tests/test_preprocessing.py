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


def test_remove_punct_df():
    string1 = "I'm a test string!"
    string2 = "I,m a test_ string"
    assert preprocessing.remove_punct_df(string1) == "im a test string"
    assert preprocessing.remove_punct_df(string2) == "im a test string"


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
