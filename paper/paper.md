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
  - name: Daniela Ushizima
    orcid: 0000-0002-7363-9468
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Ronald Pandolfi
    orcid: 0000-0003-0824-8548    
    affiliation: 1
  - name: Jeffrey Donatelli
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
affiliations:
 - name: Lawrence Berkeley National Laboratory, United States
   index: 1
 - name: Institution Name, Country
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 23 October 2023
bibliography: paper.bib

---

# Summary

Being able to discover the underlying themes and patterns in a collection of documents is necessary in order to
parse the overwhelming amount of information that is available today. Topic modeling is a Natural Language Processing
(NLP) technique that was designed specifically for this task. In early 2022 BERTopic was introduced and provided a highly
modular framework for creating customizable topic models and introducing a new method to extract the topic words from their
respective topics. This package is widely used for topic modeling on a variety of corpora. However, when conducting topic modeling
on a corpus consisting of scientific articles 

# Statement of need

`BERTeley` is a package that builds upon the topic modeling capabilities provided by `BERTopic` while addressing unqiue issues that arise when conducting topic
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
users can more easily compare the performance of different topic models trained on the same corpus.


# Features



# Intended Use Case

# Acknowledgements

This work was supported by the US Department of Energy (DOE) Office of Science Advanced Scientific Computing Research (ASCR) and Basic Energy Sciences (BES) under Contract No. DE-AC02-05CH11231 to the Center for Advanced Mathematics for Energy Research Applications (CAMERA) program. It also included support from the DOE ASCR-funded project Analysis and Machine Learning Across Domains (AMLXD), which is supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231.

# References