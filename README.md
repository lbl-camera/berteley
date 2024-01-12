# BERTeley


[![PyPI](https://badgen.net/pypi/v/berteley)](https://pypi.org/project/berteley/)
[![License](https://badgen.net/pypi/license/berteley)](https://github.com/lbl-camera/berteley)
[![Build Status](https://github.com/lbl-camera/berteley/actions/workflows/berteley-CI.yml/badge.svg)](https://github.com/lbl-camera/berteley/actions/workflows/berteley-CI.yml)
[![Documentation Status](https://readthedocs.org/projects/berteley/badge/?version=latest)](https://berteley.readthedocs.io/en/latest/?badge=latest)
[![Test Coverage](https://codecov.io/gh/lbl-camera/berteley/branch/main/graph/badge.svg?token=TTuxfR7buK)](https://codecov.io/gh/lbl-camera/berteley)

Topic modeling for scientific articles

* License: BSD license
* Documentation: https://berteley.readthedocs.io.
## Installation
```commandline
pip install berteley
```

## Quick Start

```python
# code snippet showing very basic usage
```

Topic Modeling Colab Notebook: TBD 

## Features

* A text pre-processing suite tailored for scientific articles. Including a curated stopword list.
* 3 readily available language models that are trained on scientific articles.
* Real time metric calculation for topic model comparison.

## Copyright Notice 

BERTeley Copyright (c) 2023, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy) and University
of California, Berkeley. All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.


## License Agreement

BERTeley Copyright (c) 2023, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy) and University
of California, Berkeley. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

(1) Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

(2) Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

(3) Neither the name of the University of California, Lawrence Berkeley
National Laboratory, U.S. Dept. of Energy, University of California,
Berkeley nor the names of its contributors may be used to endorse
or promote products derived from this software without specific prior
written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches,
or upgrades to the features, functionality or performance of the source
code ("Enhancements") to anyone; however, if you choose to make your
Enhancements available either publicly, or directly to Lawrence Berkeley
National Laboratory, without imposing a separate written license agreement
for such Enhancements, then you hereby grant the following license: a
non-exclusive, royalty-free perpetual license to install, use, modify,
prepare derivative works, incorporate into other computer software,
distribute, and sublicense such enhancements or derivative works
thereof, in binary and source code form.

## Credits

Please reference this work:
 <div class="row">
      <pre class="col-md-offset-2 col-md-8">
      @article{Berteley,
      title = {Benchmarking topic models on scientific articles using BERTeley},
      journal = {Natural Language Processing Journal},
      volume = {6},
      pages = {100044},
      year = {2024},
      issn = {2949-7191},
      doi = {https://doi.org/10.1016/j.nlp.2023.100044},
      url = {https://www.sciencedirect.com/science/article/pii/S2949719123000419},
      author = {Eric Chagnon and Ronald Pandolfi and Jeffrey Donatelli and Daniela Ushizima},
      keywords = {NLP, Topic modeling, Scientific articles, Transformers},
      abstract = {The introduction of BERTopic marked a crucial advancement in topic modeling and presented a topic model that outperformed both traditional and modern topic models in terms of topic modeling metrics on a variety of corpora. However, unique issues arise when topic modeling is performed on scientific articles. This paper introduces BERTeley, an innovative tool built upon BERTopic, designed to alleviate these shortcomings and improve the usability of BERTopic when conducting topic modeling on a corpus consisting of scientific articles. This is accomplished through BERTeley’s three main features: scientific article preprocessing, topic modeling using pre-trained scientific language models, and topic model metric calculation. Furthermore, an experiment was conducted comparing topic models using four different language models in three corpora consisting of scientific articles.}
      }
      </pre>
    </div>


This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter)
and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
