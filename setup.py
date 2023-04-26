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

with open(path.join(here, 'requirements-dev.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    dev = [line for line in requirements_file.read().splitlines()
           if not line.startswith('#')]

with open(path.join(here, 'requirements-docs.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    docs = [line for line in requirements_file.read().splitlines()
            if not line.startswith('#')]

setup(
    author="Eric Chagnon, Ronald J. Pandolfi, Daniela Ushizima",
    author_email="echagnon@lbl.gov",
    python_requires='>=3.8, <3.11',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
    ],
    description="Topic modeling for scientific articles",
    entry_points={
        'console_scripts': [
            # 'berteley=berteley:some_function',
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
    url='https://github.com/lbl-camera/berteley',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
    extras_require={
        'tests': dev,
        'docs': docs
    },
    setup_requires=["wheel"]
)
