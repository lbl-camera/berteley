import pytest
import os
from berteley import berteley
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
import numpy as np

# gets rid of extraneous warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@pytest.fixture
def data():
    return fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data'][:100]


@pytest.fixture
def test(data):
    model = "specter"
    topics, probabilities, metrics, topic_sizes, topic_model = berteley.fit(data, embedding_model=model, n_gram_type="bigram", verbose=True)
    return {"topics": topics, "probs": probabilities,"metrics": metrics, "topic_sizes": topic_sizes, "topic_model": topic_model}


def test_berteley_init():
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
    assert isinstance(test["topic_sizes"], dict)
    assert isinstance(test["metrics"]["Coherence"], np.float64)
    assert isinstance(test["metrics"]["Diversity"], float)


def test_figures(test):
    path = "../"
    # os.remove(path + "barchart.html")
    # os.remove(path + "barchart.png")
    #test["topic_model"].visualize_barchart(path=tmp_path)
    berteley.create_barcharts(test["topics"], test["topic_model"], path = str(tmp_path) + "/")
    # self.assertTrue(os.path.exists(path + "barchart.html"))
    # self.assertTrue(os.path.exists(path + "barchart.png"))
    assert os.path.exists(path + "barchart.html")
    assert os.path.exists(path + "barchart.png")
    with pytest.raises(TypeError):
        test["topic_model"].visualize_barchart(123)

    # self.assertRaises(TypeError, test.create_barcharts, 123)
