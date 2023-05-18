import pytest
import os
from sklearn.datasets import fetch_20newsgroups
import numpy as np

from berteley.models import initialize_model, fit, create_barcharts, calculate_metrics

# gets rid of extraneous warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@pytest.fixture
def data():
    return fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data'][:100]


@pytest.fixture
def test(data):
    model = "specter"
    topics, probabilities, topic_sizes, topic_model, topic_words = fit(data, embedding_model=model,
                                                                       n_gram_range="unigram", verbose=True)
    return {"topics": topics, "probs": probabilities, "topic_sizes": topic_sizes, "topic_model": topic_model,
            "topic_words": topic_words}


def test_berteley_init():
    # modelPath = "../bert-base-nli-mean-tokens"
    with pytest.raises(AttributeError):
        initialize_model(embedding_model="default")

    with pytest.raises(TypeError):
        initialize_model(nr_topics='2')

    with pytest.raises(TypeError):
        initialize_model(n_gram_range=2)

    with pytest.raises(AttributeError):
        initialize_model(n_gram_range='2')

    initialize_model(embedding_model="specter", n_gram_range="unigram")
    initialize_model(embedding_model="aspire", n_gram_range="bigram")
    initialize_model(embedding_model="scibert", n_gram_range=(1, 1))
    initialize_model(embedding_model="specter", n_gram_range=(2, 2))


def test_fit_input_type(data):
    df = {"data": data}
    with pytest.raises(TypeError):
        fit(df, embedding_model="specter", n_gram_type="bigram", verbose=True)


def test_fit_attributes(test, data):
    assert isinstance(test["topic_sizes"], dict)

    assert isinstance(test["topic_words"], dict)

    metrics = calculate_metrics(data, test["topic_model"], test["topics"])
    assert isinstance(metrics["Coherence"], np.float64)
    assert isinstance(metrics["Diversity"], float)


def test_bigram(data):
    topics, probabilities, topic_sizes, topic_model, topic_words = fit(data,
                                                                       embedding_model="specter",
                                                                       n_gram_range="unigram",
                                                                       verbose=True)

    metrics = calculate_metrics(data, topic_model, topics)
    assert isinstance(metrics["Coherence"], np.float64)
    assert isinstance(metrics["Diversity"], float)


def test_figures(test, tmp_path):
    # os.remove(path + "barchart.html")
    # os.remove(path + "barchart.png")
    # test["topic_model"].visualize_barchart(path=tmp_path)
    create_barcharts(test["topics"], test["topic_model"], path=str(tmp_path) + "/")
    # self.assertTrue(os.path.exists(path + "barchart.html"))
    # self.assertTrue(os.path.exists(path + "barchart.png"))
    assert os.path.exists(str(tmp_path) + "/" + "barchart.html")
    assert os.path.exists(str(tmp_path) + "/" + "barchart.png")
    with pytest.raises(TypeError):
        test["topic_model"].visualize_barchart(123)

    # self.assertRaises(TypeError, test.create_barcharts, 123)
