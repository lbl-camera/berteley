---
title: 'BERTeley: A python package for topic modeling on scientific articles'
tags:
  - Python
  - NLP
  - Topic Modeling

authors:
  - name: Eric Chagnon
    orcid: 0009-0009-7443-5721
    equal-contrib: true
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Ronald Pandolfi
    orcid: 0000-0003-0824-8548    
    affiliation: 1
  - name: Jeffrey Donatelli
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Daniela Ushizima
    orcid: 0000-0002-7363-9468
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
affiliations:
 - name: Lawrence Berkeley National Laboratory, United States
   index: 1
date: 25 October 2023
bibliography: paper.bib

---

# Summary

Being able to discover the underlying themes and patterns in a collection of documents is necessary in order to
parse the overwhelming amount of information that is available today. Topic modeling is a Natural Language Processing
(NLP) technique that was designed specifically for this task. In early 2022 `BERTopic` [@grootendorst2022bertopic] was introduced and provided a highly
modular framework for creating customizable topic models and introducing a new method to extract the topic words from their
respective topics which resulted in superior performance compared to other modern topic modeling methods. This package is widely used for topic modeling
on a variety of corpora. However, the resulting topics from a corpus of scientific articles leave a lot to be desired.

# Statement of need

`BERTeley` is a package that builds upon the topic modeling capabilities provided by `BERTopic`  while addressing unique issues that arise when conducting topic
modeling on a corpus consisting of scientific articles. If you were to take a corpus of scientific articles and use standard topic modeling practices the resulting
topic words would be dominated by words with high frequency and low value. These kinds of words come from two sources: The first being the rigid structural requirements of
scientific articles, and the shared purpose of presenting research all scientific articles have. For example, if we had a corpus of JOSS papers and used standard topic
modeling approaches structural words like _summary_, _statement_, and _acknowledgements_ would dominate the resulting topic words as they are hard requirements for the structure
of the paper. Furthermore, words that capture the underlying shared attributes of most, or all, of the JOSS papers would similarly dominate the results. These can be words like
_open-source_, _collaborative_, _python_, _package_, and _software_. By removing high frequency low value words (and their respective stems and affixes) the resulting final
topics provide a much clearer representation of the latent topics within the corpus.

`BERTeley` also provides quality-of-life improvements to `BERTopic` for conducting topic modeling on scientific articles. Modern transformer-based topic modeling approaches
all share a similar first step: creating a vector representation, an embedding, of the desired documents. This is a crucial part in the topic modeling process, as low quality 
embeddings will be detrimental to the subsequent steps in the topic modeling process. The first improvement is a set of pre-selected language models that have been pre-trained
on scientific data. Since these models have been trained exclusively on scientific data, their embeddings better capture the sometimes esoteric text found in scientific articles. 
Compared to standard language models which have been trained on a variety of text, the embeddings created by the scientific language models provide a stronger foundation for conducting
topic modeling. Secondly, topic modeling metric calculation at runtime was added. `BERTopic` allows for an easy iterable process in creating topic models, by adding topic modeling metrics
users can more easily compare the performance of different topic models trained on the same corpus
The most common metrics used to evaluate topic models are Topic Coherence and Topic Diversity [@metrics]. Topic Coherence is a measure of the word similarities of the top words within a given topic [@coherence],
and it works as a coefficient to gauge intra-cluster correlation. This measure ranges from $[-1, 1]$ where 1 indicates a perfect correlation between the topic words and -1 indicates that the topic words are not related at all.
There are several variations of Topic Coherence which use different formulas for calculating the metric [@lisena-etal-2020-tomodapi]. Here we use the C_v measure, which considers the co-occurrence of topic words in
a predefined external corpus, as it has the highest correlation with human interpretation when evaluating topic models [@lisena-etal-2020-tomodapi]. 
Topic Diversity indicates the percentage of unique topic words and measures the repetitiveness of a topic model, with values ranging from $[0, 1]$, with 1 indicating that all topic words are unique and 0 indicating that
there are no unique topic words [@DBLP:journals/corr/abs-1907-04907]. This metric works as a coefficient to gauge inter-cluster correlation. The `Octis` library provides a wealth of topic modeling metrics and means to calculate them.


# Features
To remedy the unique issues that appear when conducting topic modeling on a corpus of scientific articles a preprocessing suite was created and consists of the following steps:
 - Remove empty strings
 - Remove html tags
 - Expand all contractions
 - Remove all punctuation
 - Remove excess whitespace created by the previous steps
 - Lemmatization
 - Standard and Scientifically-irrelevant stopword removal

Users have the option of using individual functions, or make use of the `preprocess` function which acts as a wrapper and calls these functions in order. When working with a large corpus
and reasonable computing resources users can also make use of the `preprocess_parallel` function to distribute the workload across multiple workers and keep track of all of their progress
with a single progress bar.

`BERTeley` also presents users with a choice of three language models pre-trained specifically on scientific articles: Specter [@specter],
Aspire [@mysore2021aspire], and SciBERT [@scibert]. `BERTopic` is able to interface with language models from the `SentenceTransformer` [@reimers-2019-sentence-bert] library easily. Simply downloading
a `SentenceTransformer` object and passing it in as an argument works right away. However, language models not available in the `SentenceTransformer`
library, like Aspire and SciBERT, require extra setup. `BERTeley` allows the user to select any of these three models by simply passing in the name as a string.
If Aspire or SciBERT are selected, we carry out the extra steps so the language model can properly interface with `BERTopic`. If users have another language model
they would like to use for creating document embeddings, a `SentenceTransformer` object can be passed in as well. 
Finally, `BERTeley` provides topic modeling metric calculation at runtime and returns and object containing the Coherence and Diversity scores for the topic model
on a given corpus.  

# Intended Use Case
This package would allow researchers to expedite the literature review process on a project in a new domain that may be outside
their area of expertise. When able to understand the types of content that is published in the area, researchers can focus
on understanding documents in the relevant areas.

# Acknowledgements

This work was supported by the US Department of Energy (DOE) Office of Science Advanced Scientific Computing Research (ASCR) and Basic Energy Sciences (BES)
under Contract No. DE-AC02-05CH11231 to the Center for Advanced Mathematics for Energy Research Applications (CAMERA) program. 
It also included support from the DOE ASCR-funded project Analysis and Machine Learning Across Domains (AMLXD), which is supported by the Office of Science of the
U.S. Department of Energy under Contract No. DE-AC02-05CH11231.

# References