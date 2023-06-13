#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="Thalita Veronese",
    author_email='veronese@unicamp.br',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="A general pipeline for trustworthy machine learning on credit datasets.",
    entry_points={
        'console_scripts': [
            'credit_pipeline=credit_pipeline.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='credit_pipeline',
    name='credit_pipeline',
    packages=find_packages(include=['credit_pipeline', 'credit_pipeline.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/thalitabv/credit_pipeline',
    version='0.1.0',
    zip_safe=False,
)
