#!/usr/bin/env python

"""The setup script."""
from os import path
from setuptools import setup, find_packages
import sys
import versioneer

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]

setup(
    author="Eric Chagnon",
    author_email='eric.chagnon15@gmail.com',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
    description="Topic modeling for scientific articles using BERTopic",
    entry_points={
        'console_scripts': [
            #'berteley=berteley:some_function',
        ],
    },
    install_requires=requirements,
    license="BSD license",
    long_description=readme,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='berteley',
    name='berteley',
    packages=find_packages(include=['berteley', 'berteley.*']),
    test_suite='tests',
    url='https://github.com/ericchagnon15/berteley',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
    extras_require={
        'tests': ['pytest', 'codecov', 'pytest-cov'],
        'docs': ['sphinx', 'sphinx-rtd-theme', 'myst-parser', 'myst-nb', 'sphinx-panels', 'autodocs']
    }
)