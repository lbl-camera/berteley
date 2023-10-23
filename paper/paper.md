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
topic words would be dominated by words with high frequency and low value. These kinds of words stem from two sources. The first being the rigid structural requirements
scientific articles posses leads to words like _introduction_, _appendix_, and _results_. 
Second, all scientific articles share a similar purpose: to present the findings of a research project. Words inherent to a document with this purpose will show up in many 
scientific articles regardless of their domain. This can include words like _show_, _use_, and _propose_. By removing these high frequency words, and their stems and affixes,
the resulting topic words provide a much clearer understanding of the contents of the corpus.


# Features

# Intended Use Case

# Acknowledgements

This work was supported by the US Department of Energy (DOE) Office of Science Advanced Scientific Computing Research (ASCR) and Basic Energy Sciences (BES) under Contract No. DE-AC02-05CH11231 to the Center for Advanced Mathematics for Energy Research Applications (CAMERA) program. It also included support from the DOE ASCR-funded project Analysis and Machine Learning Across Domains (AMLXD), which is supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231.

# References