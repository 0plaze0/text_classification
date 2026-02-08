## Problem Statement

Design a classifier that reads a text document and classifies it as Sports or Politics using machine learning techniques. The project compares three ML techniques (Naive Bayes, Random Forest, SVM) using TF-IDF feature representation.

## Project Overview

This project implements a complete text classification pipeline that:
- Downloads and processes text data from Kaggle
- Performs exploratory data analysis and visualization
- Preprocesses text with TF-IDF feature extraction
- Trains and compares three different machine learning models
- Generates performance metrics and comparison plots

The classifier achieves up to 99% accuracy in distinguishing between Sports and Politics documents using Support Vector Machine.



## Dataset

Source: [Text Document Classification Dataset](https://www.kaggle.com/datasets/sunilthite/text-document-classification-dataset) from Kaggle

Classes:
- Politics 
- Sports 

Total Documents: ~3,500 (balanced dataset with approximately 50% each class)

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Kaggle account (for dataset download)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/sports_politics_classifier.git
cd sports_politics_classifier
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```


## Running the Code

### Option 1: Run Classification Script

This is the main way to train models and generate results:

```bash
python text_classifier.py
```

What it does:
- Downloads dataset from Kaggle automatically
- Preprocesses text data (cleaning, normalization)
- Trains three models: Naive Bayes, Random Forest, SVM
- Displays evaluation metrics in console
- Saves confusion matrices to `plots/confusion_matrices.png`
- Saves model comparison to `plots/model_comparison.png`
- Shows example predictions

### Option 2: Run Jupyter Notebook (Data Analysis Only)

For exploratory data analysis and visualizations:

```bash
jupyter notebook data_analysis.ipynb
```

The notebook includes:
- Data loading and preprocessing steps
- Class distribution analysis
- Word count statistics
- N-gram frequency analysis
- Word cloud visualizations

### Option 3: Use as Python Module

You can import and use the classifier in your own scripts:

```python
from text_classifier import TextClassifier

# Initialize classifier
classifier = TextClassifier()

# Run complete pipeline
classifier.run_pipeline(save_plots=True)

# Make predictions on new text
text = "The basketball team won the championship game."
prediction = classifier.predict(text, model_name='SVM')
print(f"Prediction: {prediction}")  # Output: Sports
```

****
## Machine Learning Models

### 1. Naive Bayes (MultinomialNB)
- Probabilistic classifier based on Bayes' theorem
- Fast training and prediction
- Effective for text classification tasks
- Accuracy: ~98%

### 2. Random Forest
- Ensemble method using 200 decision trees
- Handles non-linear relationships
- Provides feature importance
- Accuracy: ~97%

### 3. Support Vector Machine (SVM)
- Linear kernel for high-dimensional text data
- Maximizes margin between classes
- Best overall performance
- Accuracy: ~99%

## Feature Engineering

TF-IDF Vectorization:
- Max features: 5000
- N-gram range: (1, 2) - captures unigrams and bigrams
- Stop words: English common words removed
- Creates sparse matrix of 5000 features per document

## Performance Results

| Model          | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| Naive Bayes    | 98.45%   | 98.50%    | 98.45% | 98.45%   |
| Random Forest  | 97.23%   | 97.20%    | 97.23% | 97.20%   |
| SVM            | 99.01%   | 99.00%    | 99.01% | 99.00%   |


