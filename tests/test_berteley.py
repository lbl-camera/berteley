import pytest
import os
from berteley.berteley import BERTeley
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
import numpy as np

# gets rid of extraneous warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@pytest.fixture
def data():
    return fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data'][:100]

@pytest.fixture
def test(data):
    model = SentenceTransformer('allenai-specter')
    test = BERTeley(embedding_model=model, n_gram_type="bigram")
    test.fit(data)
    return test


def test_berteley_init():
    # this works it just takes a very long time
    # modelPath = "../bert-base-nli-mean-tokens"
    with pytest.raises(AttributeError):
        BERTeley(embedding_model="default")

    with pytest.raises(TypeError):
        BERTeley(nr_topics='2')

    with pytest.raises(TypeError):
        BERTeley(n_gram_type=2)

    with pytest.raises(AttributeError):
        BERTeley(n_gram_type='2')


def test_fit_input_type(data):
    df = pd.DataFrame(data, columns=['Documents'])
    test_df = BERTeley()
    with pytest.raises(TypeError):
        test_df.fit(df)


def test_fit_attributes(test):
    assert isinstance(test.topic_sizes, dict)
    assert isinstance(test.coherence, np.float64)
    assert isinstance(test.diversity, float)


def test_figures(test):
    path = "../"
    # os.remove(path + "barchart.html")
    # os.remove(path + "barchart.png")
    test.create_barcharts(path=path)
    # self.assertTrue(os.path.exists(path + "barchart.html"))
    # self.assertTrue(os.path.exists(path + "barchart.png"))
    assert os.path.exists(path + "barchart.html")
    assert os.path.exists(path + "barchart.png")
    with pytest.raises(TypeError):
        test.create_barcharts(123)

    # self.assertRaises(TypeError, test.create_barcharts, 123)
