---
title: 'BERTeley: A python package for topic modeling on scientific articles'
tags:
  - Python
  - NLP
  - Topic Modeling

authors:
  - name: Eric Chagnon
    orcid: 0009-0009-7443-5721
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Ronald Pandolfi
    orcid: 0000-0003-0824-8548    
    affiliation: 1
  - name: Jeffrey Donatelli
    affiliation: 1
  - name: Daniela Ushizima
    orcid: 0000-0002-7363-9468
    affiliation: 1
affiliations:
 - name: Lawrence Berkeley National Laboratory, United States
   index: 1
date: 25 October 2023
bibliography: paper.bib

---

# Summary

Topic modeling is a Natural Language Processing
(NLP) technique designed to discover the underlying themes and patterns in a collection of documents (corpus). In early 2022 `BERTopic` [@grootendorst2022bertopic] became a fairly popular Python library in topic modeling, leveraging transformer-based embeddings, such as those from BERT (Bidirectional Encoder Representations from Transformers) to create dense vector representations of text. In order to enable more streamlined topic modeling of scientific articles `BERTeley`, which builds upon BERTopic, was developed to address unique challenges in extracting meaningful topics from a corpus of scientific articles while also providing some quality-of-life improvements.


For example, BERTeley’s preprocessing suite helps to remove high-frequency, low-value words. In addition, three scientific language models are pre-selected so users have reasonable choices to create
vector representations of the documents, which store the results from the embedding models. In addition, BERTeley seamlessly performs topic modeling metrics calculation during runtime, enabling an insightful comparison of topic models, and more interpretability.

This package has been used to allow researchers to expedite the literature review process and also to quickly survey new and possibly unfamiliar research areas.



# Statement of need

When applied to a corpus of scientific articles, standard topic modeling practices typically produce topic words that are dominated by general scientific terminology which carry less significance to the specific content of the article. These words come from two sources: The rigid structural requirements of
scientific articles, and the shared purpose of presenting research, such as describing materials and methods. For example, if we had a corpus of JOSS papers and used standard topic
modeling approaches structural words, such as _summary_, _statement_, and _acknowledgments_ would dominate the resulting topic words as they are hard requirements for the structure
of the paper. Furthermore, words that capture the underlying shared attributes of most, or all, of the JOSS papers would similarly dominate the results. This often includes words such as
_open-source_, _collaborative_, _python_, _package_, and _software_.


Modern transformer-based topic modeling approaches share a similar first step: creating a vector representation(embedding) of the desired documents. This step is a crucial part of the topic modeling process, as low-quality 
embeddings will be detrimental to the subsequent steps in the topic modeling process. The first improvement is a set of pre-selected language models that have been pre-trained
on scientific data. These specialized models, trained solely on scientific data, generate embeddings that more accurately represent the unique language of scientific articles. This results in a more solid foundation for performing topic modeling on scientific text, compared to standard language models trained on diverse text types.. 



Finally, in order to better compare topic models that are trained on the same corpus topic modeling metric calculation was implemented. This calculates the Coherence and Diversity scores [@metrics] for a given topic model at runtime and returns the values to the user after the fitting process is complete.




# Features
`BERTeley’ is a Python package that aims to overcome current limitations in scientific topic modeling through the use of systematic text cleaning
To remedy the unique issues that appear when conducting topic modeling on a corpus of scientific articles, a preprocessing suite was created and consists of the following steps:

 - Remove empty strings
 - Remove html tags
 - Expand all contractions
 - Remove all punctuation
 - Remove excess whitespace created by the previous steps
 - Lemmatization
 - Standard and Scientifically-irrelevant stopword removal

Users have the option of using individual functions or making use of the `preprocess` function which acts as a wrapper and calls these functions in order. When working with a large corpus
and reasonable computing resources users can also make use of the `preprocess_parallel` function to distribute the workload across multiple workers and keep track of all of their progress
with a single progress bar.

`BERTeley` also presents users with a choice of three language models pre-trained specifically on scientific articles: Specter [@specter],
Aspire [@mysore2021aspire], and SciBERT [@scibert]. `BERTopic` is able to interface with language models from the `SentenceTransformer` [@reimers-2019-sentence-bert] library easily. This can be accomplished by simply downloading a `SentenceTransformer` object and passing it in as an argument. However, language models not available in the `SentenceTransformer`
library, such as Aspire and SciBERT, require extra setup. `BERTeley` allows the user to select any of these three models by simply passing in the name as a string.
If Aspire or SciBERT are selected, we carry out the extra steps so the language model can properly interface with `BERTopic`. If users have another language model
they would prefer to use for creating document embeddings, a `SentenceTransformer` object can be passed in as well. 

Finally, `BERTopic` allows for an easy iterable process in creating topic models, by adding topic modeling metrics
users can more easily compare the performance of different topic models trained on the same corpus
Topic Coherence quantifies the semantic similarities among the top words within a specified topic, functioning as a coefficient to evaluate the degree of intra-cluster correlation [@coherence]. This measure ranges from $[-1, 1]$ where 1 indicates a perfect correlation between the topic words and -1 indicates that the topic words are unrelated.
There are several variations of Topic Coherence which use different formulas for calculating the metric [@lisena-etal-2020-tomodapi]. Here we use the C_v measure, which considers the co-occurrence of topic words in
a predefined external corpus, as it has the highest correlation with human interpretation when evaluating topic models [@lisena-etal-2020-tomodapi]. 
Topic Diversity indicates the percentage of unique topic words and measures the repetitiveness of a topic model, with values ranging from $[0, 1]$, with 1 indicating that all topic words are unique and 0 indicating that
there are no unique topic words [@DBLP:journals/corr/abs-1907-04907]. This metric functions as a coefficient, offering a quantifiable measure of inter-cluster correlation. The `Octis` library, rich in resources for topic modeling metrics, facilitates the computation of this particular coefficient, ensuring a comprehensive analytical approach. 


# Acknowledgements

This work was supported by the US Department of Energy (DOE) Office of Science Advanced Scientific Computing Research (ASCR) and Basic Energy Sciences (BES)
under Contract No. DE-AC02-05CH11231 to the Center for Advanced Mathematics for Energy Research Applications (CAMERA) program. 
It also included support from the DOE ASCR-funded project Analysis and Machine Learning Across Domains (AMLXD), which is supported by the Office of Science of the
U.S. Department of Energy under Contract No. DE-AC02-05CH11231.

# References

