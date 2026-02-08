"""
Sports vs Politics Text Classifier
===================================
This script performs text classification on documents to classify them as either
Sports or Politics using three different ML techniques: Naive Bayes, Random Forest, and SVM.

Features:
- Downloads dataset from Kaggle
- Preprocesses text data
- Trains and evaluates three different classifiers
- Generates comparison plots and confusion matrices
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class TextClassifier:
    """Text classifier for Sports vs Politics classification"""
    
    def __init__(self, data_path=None):
        """
        Initialize the classifier
        
        Args:
            data_path: Path to CSV file. If None, downloads from Kaggle
        """
        self.data_path = data_path
        self.df = None
        self.tfidf = None
        self.models = {}
        self.predictions = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load dataset from Kaggle or local path"""
        if self.data_path is None:
            print("Downloading dataset from Kaggle...")
            path = kagglehub.dataset_download("sunilthite/text-document-classification-dataset")
            print(f"Dataset downloaded to: {path}")
            
            files = os.listdir(path)
            print(f"Files in dataset: {files}")
            
            csv_path = os.path.join(path, 'df_file.csv')
        else:
            csv_path = self.data_path
            
        print(f"Loading data from: {csv_path}")
        self.df = pd.read_csv(csv_path)
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
    def preprocess_data(self):
        """Filter and clean the dataset"""
        print("\nPreprocessing data...")
        
        # Keep only Politics (0) and Sports (1)
        label_map = {0: "Politics", 1: "Sports"}
        self.df = self.df[self.df["Label"].isin([0, 1])]
        self.df["Label"] = self.df["Label"].map(label_map)
        
        print(f"Class distribution:\n{self.df['Label'].value_counts()}")
        
        # Basic cleaning
        self.df.dropna(subset=["Text"], inplace=True)
        self.df["Text"] = self.df["Text"].astype(str)
        self.df = self.df[self.df["Text"].str.strip() != ""]
        
        # Text preprocessing
        self.df["clean_text"] = self.df["Text"].apply(self._clean_text)
        
        print(f"After cleaning: {self.df.shape[0]} documents")
        
    @staticmethod
    def _clean_text(text):
        """Clean and normalize text"""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def prepare_features(self):
        """Split data and create TF-IDF features"""
        print("\nPreparing features...")
        
        X = self.df["clean_text"]
        y = self.df["Label"]
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        print(f"Training set: {len(self.X_train)} documents")
        print(f"Test set: {len(self.X_test)} documents")
        
        # TF-IDF vectorization
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english"
        )
        
        self.X_train_tfidf = self.tfidf.fit_transform(self.X_train)
        self.X_test_tfidf = self.tfidf.transform(self.X_test)
        
        print(f"TF-IDF feature matrix shape: {self.X_train_tfidf.shape}")
        
    def train_models(self):
        """Train all three classifiers"""
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        # Model 1: Naive Bayes
        print("\n1. Training Naive Bayes...")
        nb = MultinomialNB()
        nb.fit(self.X_train_tfidf, self.y_train)
        self.models['Naive Bayes'] = nb
        self.predictions['Naive Bayes'] = nb.predict(self.X_test_tfidf)
        
        # Model 2: Random Forest
        print("2. Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(self.X_train_tfidf, self.y_train)
        self.models['Random Forest'] = rf
        self.predictions['Random Forest'] = rf.predict(self.X_test_tfidf)
        
        # Model 3: SVM
        print("3. Training SVM...")
        svm = LinearSVC(random_state=42, max_iter=2000)
        svm.fit(self.X_train_tfidf, self.y_train)
        self.models['SVM'] = svm
        self.predictions['SVM'] = svm.predict(self.X_test_tfidf)
        
        print("\nAll models trained successfully!")
        
    def evaluate_models(self):
        """Evaluate and print metrics for all models"""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        results = []
        
        for name, y_pred in self.predictions.items():
            accuracy = accuracy_score(self.y_test, y_pred)
            results.append({
                'Model': name,
                'Accuracy': accuracy
            })
            
            print(f"\n{name}")
            print("-" * 40)
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred))
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(results_df.to_string(index=False))
        
        return results_df
    
    def plot_confusion_matrices(self, save_dir='plots'):
        """Generate and save confusion matrices for all models"""
        print(f"\nGenerating confusion matrices...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, (name, y_pred) in enumerate(self.predictions.items()):
            cm = confusion_matrix(self.y_test, y_pred)
            
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Politics", "Sports"],
                yticklabels=["Politics", "Sports"],
                ax=axes[idx],
                cbar=True
            )
            
            axes[idx].set_title(f'{name}\nAccuracy: {accuracy_score(self.y_test, y_pred):.3f}')
            axes[idx].set_xlabel("Predicted")
            axes[idx].set_ylabel("Actual")
        
        plt.tight_layout()
        filepath = os.path.join(save_dir, 'confusion_matrices.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.show()
        
    def plot_model_comparison(self, save_dir='plots'):
        """Generate accuracy comparison bar plot"""
        print(f"\nGenerating model comparison plot...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        accuracies = {
            name: accuracy_score(self.y_test, y_pred)
            for name, y_pred in self.predictions.items()
        }
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(accuracies.keys(), accuracies.values(), 
                       color=['#3498db', '#2ecc71', '#e74c3c'])
        
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('Model', fontsize=12)
        plt.title('Model Performance Comparison\nSports vs Politics Classification', 
                  fontsize=14, fontweight='bold')
        plt.ylim(0.8, 1.0)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(save_dir, 'model_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.show()
        
    def predict(self, text, model_name='SVM'):
        """
        Predict the class of a new text document
        
        Args:
            text: Input text string
            model_name: Name of the model to use ('Naive Bayes', 'Random Forest', or 'SVM')
            
        Returns:
            Predicted class label
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        
        cleaned = self._clean_text(text)
        features = self.tfidf.transform([cleaned])
        prediction = self.models[model_name].predict(features)[0]
        
        return prediction
    
    def run_pipeline(self, save_plots=True):
        """Run the complete classification pipeline"""
        print("="*60)
        print("SPORTS VS POLITICS TEXT CLASSIFIER")
        print("="*60)
        
        # Execute pipeline
        self.load_data()
        self.preprocess_data()
        self.prepare_features()
        self.train_models()
        results = self.evaluate_models()
        
        if save_plots:
            self.plot_confusion_matrices()
            self.plot_model_comparison()
        
        print("\n" + "="*60)
        print("Pipeline completed successfully!")
        print("="*60)
        
        return results


def main():
    """Main execution function"""
    # Initialize classifier
    classifier = TextClassifier()
    
    # Run complete pipeline
    results = classifier.run_pipeline(save_plots=True)
    
    # Example predictions
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)
    
    test_texts = [
        "The quarterback threw a touchdown pass in the final seconds of the game.",
        "The parliament voted on the new healthcare legislation today.",
        "The team won the championship after a thrilling overtime victory."
    ]
    
    for text in test_texts:
        prediction = classifier.predict(text, model_name='SVM')
        print(f"\nText: {text}")
        print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
