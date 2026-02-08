# Sports vs Politics Text Classifier

A machine learning project that classifies text documents as either **Sports** or **Politics** using three different classification algorithms: Naive Bayes, Random Forest, and Support Vector Machine (SVM).

## 📋 Project Overview

This project implements a complete text classification pipeline including:

- Data collection from Kaggle
- Exploratory Data Analysis (EDA)
- Text preprocessing and feature engineering
- Training and evaluation of three ML models
- Performance comparison and visualization

## 🎯 Problem Statement

Design a classifier that reads a text document and classifies it as Sports or Politics using machine learning techniques. The project compares at least three ML techniques using TF-IDF feature representation.

## 📊 Dataset

**Source:** [Text Document Classification Dataset](https://www.kaggle.com/datasets/sunilthite/text-document-classification-dataset) from Kaggle

**Classes:**

- Politics (Label: 0)
- Sports (Label: 1)

## 🗂️ Repository Structure

```
sports_politics_classifier/
├── data_analysis.ipynb          # Jupyter notebook with EDA and analysis
├── text_classifier.py           # Standalone classifier script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── plots/                       # Generated visualizations (auto-created)
│   ├── confusion_matrices.png
│   └── model_comparison.png
└── .gitignore                   # Git ignore file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/sports_politics_classifier.git
cd sports_politics_classifier
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

### Running the Classifier

**Option 1: Run the standalone script**

```bash
python text_classifier.py
```

This will:

- Download the dataset from Kaggle
- Preprocess the data
- Train all three models
- Generate evaluation metrics
- Create visualization plots in the `plots/` directory

**Option 2: Use Jupyter Notebook for EDA**

```bash
jupyter notebook data_analysis.ipynb
```

## 🔧 Usage

### Using the Classifier in Your Code

```python
from text_classifier import TextClassifier

# Initialize classifier
classifier = TextClassifier()

# Run complete pipeline
results = classifier.run_pipeline(save_plots=True)

# Make predictions on new text
text = "The quarterback threw a touchdown pass in the final seconds."
prediction = classifier.predict(text, model_name='SVM')
print(f"Prediction: {prediction}")  # Output: Sports
```

### Custom Data Path

```python
# Use your own CSV file
classifier = TextClassifier(data_path='path/to/your/data.csv')
classifier.run_pipeline()
```

## 🧪 Methods & Techniques

### Feature Representation

- **TF-IDF Vectorization**
    - Max features: 5000
    - N-gram range: (1, 2) - unigrams and bigrams
    - Stop words: English

### Text Preprocessing

1. Lowercase conversion
2. Removal of special characters
3. Whitespace normalization
4. Feature engineering (word count, character count)

### Machine Learning Models

1. **Naive Bayes (MultinomialNB)**
    - Probabilistic classifier
    - Works well with text data
    - Fast training and prediction

2. **Random Forest**
    - Ensemble method
    - 200 estimators
    - Parallel processing enabled

3. **Support Vector Machine (Linear SVC)**
    - Linear kernel
    - Effective in high-dimensional spaces
    - Good for text classification

## 📈 Results

The models are evaluated using:

- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1-Score)
- **Confusion Matrix**

### Sample Performance

| Model         | Accuracy |
| ------------- | -------- |
| Naive Bayes   | ~0.98    |
| Random Forest | ~0.97    |
| SVM           | ~0.99    |

_Note: Actual results may vary based on the dataset split_

## 📊 Visualizations

The project generates the following plots:

1. **Class Distribution** - Bar chart of document counts
2. **Word Count Distribution** - Histogram of text lengths
3. **Word Count by Class** - Box plot comparison
4. **Top N-grams** - Most frequent unigrams
5. **Word Clouds** - Visual representation of frequent terms
6. **Confusion Matrices** - Model prediction accuracy
7. **Model Comparison** - Accuracy comparison bar chart

## 📝 Project Report Components

The complete project includes:

1. **Data Collection** - Dataset source and acquisition method
2. **Dataset Description** - Structure, features, and statistics
3. **Exploratory Data Analysis** - Visual and statistical analysis
4. **Preprocessing** - Text cleaning and normalization steps
5. **Feature Engineering** - TF-IDF vectorization details
6. **Model Training** - Three ML techniques implementation
7. **Quantitative Comparison** - Accuracy, precision, recall, F1-scores
8. **Limitations** - System constraints and potential improvements

## ⚙️ Configuration

Key parameters can be modified in `text_classifier.py`:

```python
# TF-IDF settings
max_features = 5000
ngram_range = (1, 2)

# Random Forest settings
n_estimators = 200

# Train-test split
test_size = 0.2
random_state = 42
```

## 🔍 System Limitations

1. **Binary Classification** - Only supports Politics vs Sports
2. **Language** - Optimized for English text only
3. **Domain Specificity** - May not generalize to other domains
4. **Feature Limit** - TF-IDF limited to 5000 features
5. **Preprocessing** - Simple cleaning; advanced NLP techniques not implemented

## 🛠️ Future Improvements

- Add more document categories
- Implement deep learning models (LSTM, BERT)
- Add cross-validation
- Implement hyperparameter tuning
- Create web interface for predictions
- Add multi-language support

## 📦 Dependencies

See `requirements.txt` for complete list. Main dependencies:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- kagglehub

## 📄 License

This project is available under the MIT License.

## 👤 Author

[Your Name]

- GitHub: [@yourusername](https://github.com/yourusername)

## 🙏 Acknowledgments

- Dataset: [Sunil Thite on Kaggle](https://www.kaggle.com/sunilthite)
- Course: Machine Learning Assignment

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note:** Make sure to configure your Kaggle API credentials before running the script. See [Kaggle API documentation](https://github.com/Kaggle/kaggle-api) for setup instructions.
