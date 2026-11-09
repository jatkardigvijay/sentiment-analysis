# Sentiment Analysis (Basic AI Project)

A beginner-friendly machine learning project that classifies text as **positive**, **negative**, or **neutral** using scikit-learn.

## Overview

This project demonstrates a minimal text classification pipeline:

1. **Vectorization** — converts text into word-count features (`CountVectorizer`)
2. **Classification** — learns word → label patterns (`MultinomialNB`)
3. **Pipeline** — chains both steps into a single reusable model

## Requirements

- Python 3.10+
- scikit-learn
- pandas (optional, not required for the core script)

## Installation

Clone or download the project, then install dependencies:

```bash
pip install scikit-learn pandas