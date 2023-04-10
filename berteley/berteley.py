import collections
import gensim.corpora as corpora
import nltk
import pandas as pd
from bertopic import BERTopic
from gensim.models.coherencemodel import CoherenceModel
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from sentence_transformers import SentenceTransformer, models
from typing import List

nltk.download('stopwords')

nltk.download('punkt')


class BERTeley:

    def __init__(self,
                 embedding_model: SentenceTransformer = None,
                 nr_topics: int = None,
                 n_gram_type: str = "unigram",
                 verbose: bool = False):

        """

        Parameters
        ----------
        embedding_model
                Either a SentenceTransformer, or a string with values "specter". "aspire", or "scibert".
        nr_topics
                The desired number of topics, if not specified the results will be determined by HDBSCAN's reduction step.
        n_gram_type
                String indicating whether the user would like unigram or bigram.
        verbose
                Boolean indicating verbose output.
        """

        if not isinstance(nr_topics, int) and nr_topics is not None:
            raise TypeError("nr_topics must be an int")
        if not isinstance(n_gram_type, str) and n_gram_type is not None:
            raise TypeError("n_gram_type must be an string")

        if n_gram_type == "unigram":
            n_gram_range = (1, 1)
        elif n_gram_type == "bigram":
            n_gram_range = (2, 2)
        else:
            raise AttributeError("n_gram_type must equal \"unigram\" or \"bigram\" ")

        if isinstance(embedding_model, str):
            if embedding_model.lower() == "specter":
                embedding_model = SentenceTransformer('allenai-specter')

            elif embedding_model.lower() == "aspire":
                word_embedding_model = models.Transformer('allenai/aspire-sentence-embedder', max_seq_length=512)
                pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
                embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

            elif embedding_model.lower() == "scibert":
                word_embedding_model = models.Transformer('allenai/scibert_scivocab_uncased', max_seq_length=512)
                pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
                embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

            else:
                raise AttributeError("embedding model should either be a string specifying one of the 3 pre-loaded "
                                     "models, or the desired language model")

        self.embedding_model = embedding_model
        self.nr_topics = nr_topics

        self.n_gram_range = n_gram_range
        self.verbose = verbose

        # Attributes
        self.coherence = None
        self.diversity = None
        self.topic_sizes = None
        self.topic_authors = None
        self.__BERTopic = None
        self.topics = None
        self.__probs = None

    def __calculate_metrics__(self, texts: List[str]):
        """
        Calculates the Topic Coherence and Topic Diversity of the topic model.
        Parameters
        ----------
        texts
             The list of documents

        Returns
        -------
        dict
            A dict containing the metrics as keys, and their respective scores as values.
        """

        topic_model = self.__BERTopic
        topic_words = {}
        topic_dict = topic_model.topic_representations_
        for k in topic_dict.keys():
            topic_words[k] = [x[0] for x in topic_dict[k]]
        word_list = list(topic_words.values())
        word_list.pop(0)

        # octis requires the texts input be in the form of a list of list of strings
        octis_texts = [sentence.split() for sentence in texts]

        npmi = Coherence(texts=octis_texts, topk=10, measure='c_npmi')
        topic_diversity = TopicDiversity(topk=10)

        # reformat the output of BERTopic to the proper format
        # {topics: [[topic, words, for, topic1], [topic, words, for, topic2], [etc, etc, etc]]}
        all_words = [word for words in octis_texts for word in words]

        # check if the model is bigram
        if self.n_gram_range == (2, 2):
            bertopic_topics = [
                [
                    vals[0] if (vals[0].split()[0] in all_words or vals[0].split()[1] in all_words) else all_words[0]
                    for vals in topic_model.get_topic(i)[:10]
                ]
                for i in range(len(set(self.topics)) - 1)
            ]

            output_tm = {"topics": bertopic_topics}

            coherence_score = self.__calculate_coherence__(topic_model, texts)

        # unigram
        else:

            output_tm = {"topics": word_list}

            coherence_score = npmi.score(output_tm)

        diversity_score = topic_diversity.score(output_tm)

        self.coherence = coherence_score
        self.diversity = diversity_score
        # return {"Coherence": coherence_score, "Diversity": diversity_score}

    def __calculate_coherence__(self, topic_model: BERTopic, docs: list[str]):

        """
        Internal method for calculating the coherence when the n_gram_range is set to something other than (1,1)
        Parameters
        ----------
        topic_model
            The BERTopic object
        docs
            The list of documents

        Returns
        -------
            The coherence score

        """

        # Preprocess Documents
        documents = pd.DataFrame({"Document": docs,
                                  "ID": range(len(docs)),
                                  "Topic": self.topics})
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)

        # Extract vectorizer and analyzer from BERTopic
        vectorizer = topic_model.vectorizer_model
        analyzer = vectorizer.build_analyzer()

        # Extract features for Topic Coherence evaluation
        words = vectorizer.get_feature_names()
        tokens = [analyzer(doc) for doc in cleaned_docs]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]
        topic_words = [[words for words, _ in topic_model.get_topic(topic)]
                       for topic in range(len(set(self.topics)) - 1)]

        # Evaluate
        coherence_model = CoherenceModel(topics=topic_words,
                                         texts=tokens,
                                         corpus=corpus,
                                         dictionary=dictionary,
                                         coherence='c_npmi')
        coherence = coherence_model.get_coherence()
        return coherence

    def fit(self, data: list[str]):
        """
        Fits a BERTopic model on the data. After fitting the topic assigned to each document is stored
        in the 'topics' attribute, the coherence and diversity measures are stored in the
        'coherence' and 'diversity' attributes respectively, and the amount of documents assigned to each topic
        are stored in the 'topic_sizes' attribute.

        Parameters
        ----------
        data
            List of the documents

        Returns
        -------
        The function sets attributes for topics, probabilities, and metrics.

        """

        if not isinstance(data, list):
            raise TypeError("data must be a list of strings")

        self.__BERTopic = BERTopic(embedding_model=self.embedding_model,
                                   nr_topics=self.nr_topics,
                                   n_gram_range=self.n_gram_range,
                                   verbose=self.verbose)
        self.topics, self.__probs = self.__BERTopic.fit_transform(data)
        self.__calculate_metrics__(data)
        self.__calculate_topic_sizes__()

    def __calculate_topic_sizes__(self):
        """
        counts the number of documents assigned to each topic and stores the topic sizes in the 'topic_sizes'
        attribute in the form of a dict.

        Parameters
        ----------


        Returns
        -------
        Sets the topic size parameter to a collection object containing the topics and their sizes.

        """

        counter = collections.Counter(self.topics)
        self.topic_sizes = dict(counter)

    def create_barcharts(self, path=""):
        """
        creates and saves the BERTopic barcharts

        Parameters
        ----------
        path
            A string containing the desired path to save the barchart

        Returns
        -------
        Barcharts are saved in .html and .png format in the desired directory.

        """

        if not isinstance(path, str):
            raise TypeError("path must be a string")

        if path != "":
            path = path + 'barchart'

        # model_path = base_path + iter_version + beamline + "_" + model_use + "_" + n_gram_type + "/"
        fig = self.__BERTopic.visualize_barchart(top_n_topics=len(self.topics))
        fig_name = path + ".html"
        fig_name_png = path + ".png"
        if path != "":
            fig.write_html(fig_name)
            fig.write_image(fig_name_png)
        return fig
