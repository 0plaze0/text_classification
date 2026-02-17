#!/usr/bin/env python3
"""
Quick Test Script for Sports vs Politics Classifier
===================================================
This script tests if all dependencies are installed correctly.
Run this before running the main classifier.
"""

import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...\n")
    
    packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib.pyplot'),
        ('seaborn', 'seaborn'),
        ('sklearn', 'sklearn'),
        ('kagglehub', 'kagglehub'),
        ('wordcloud', 'wordcloud'),
    ]
    
    failed = []
    
    for display_name, import_name in packages:
        try:
            __import__(import_name)
            print(f" {display_name}")
        except ImportError as e:
            print(f" {display_name} - {str(e)}")
            failed.append(display_name)
    
    if failed:
        print(f"\n Failed to import: {', '.join(failed)}")
        print("\nPlease install missing packages:")
        print("pip install -r requirements.txt")
        return False
    else:
        print("\n All packages imported successfully!")
        return True

def test_sklearn_models():
    """Test if sklearn models can be instantiated"""
    print("\n" + "="*50)
    print("Testing scikit-learn models...")
    print("="*50 + "\n")
    
    try:
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import LinearSVC
        
        nb = MultinomialNB()
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        svm = LinearSVC(random_state=42)
        
        print(" Naive Bayes")
        print(" Random Forest")
        print(" SVM")
        print("\n All models can be instantiated!")
        return True
        
    except Exception as e:
        print(f" Error testing models: {str(e)}")
        return False

def test_text_processing():
    """Test basic text processing"""
    print("\n" + "="*50)
    print("Testing text processing...")
    print("="*50 + "\n")
    
    try:
        import re
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Test text cleaning
        sample_text = "This is a TEST! With 123 numbers."
        cleaned = re.sub(r"[^a-z0-9\s]", " ", sample_text.lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        
        print(f"Original: {sample_text}")
        print(f"Cleaned:  {cleaned}")
        
        # Test TF-IDF
        tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        corpus = [
            "The game was exciting and fun",
            "The election results were announced",
            "Sports news update on the championship"
        ]
        features = tfidf.fit_transform(corpus)
        
        print(f"\n✓ TF-IDF vectorization successful")
        print(f"  Feature matrix shape: {features.shape}")
        print("\n Text processing works correctly!")
        return True
        
    except Exception as e:
        print(f" Error in text processing: {str(e)}")
        return False

def check_kaggle_config():
    """Check if Kaggle API is configured"""
    print("\n" + "="*50)
    print("Checking Kaggle API configuration...")
    print("="*50 + "\n")
    
    import os
    
    kaggle_config_path = os.path.expanduser("~/.kaggle/kaggle.json")
    
    if os.path.exists(kaggle_config_path):
        print(f" Kaggle config found at: {kaggle_config_path}")
        return True
    else:
        print(f"  Kaggle config not found at: {kaggle_config_path}")
        print("\nTo set up Kaggle API:")
        print("1. Create account at https://www.kaggle.com")
        print("2. Go to Account → API → Create New Token")
        print("3. Move kaggle.json to ~/.kaggle/")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        print("\nYou can still test the classifier with a local CSV file.")
        return False

def main():
    """Run all tests"""
    print("="*50)
    print("SPORTS VS POLITICS CLASSIFIER - SETUP TEST")
    print("="*50 + "\n")
    
    results = []
    
    # Test imports
    results.append(test_imports())
    
    # Test sklearn models
    results.append(test_sklearn_models())
    
    # Test text processing
    results.append(test_text_processing())
    
    # Check Kaggle config (warning only)
    kaggle_ready = check_kaggle_config()
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    if all(results):
        print("\n All tests passed! You're ready to run the classifier.")
        if kaggle_ready:
            print("\n Run the classifier with:")
            print("   python text_classifier.py")
        else:
            print("\n  Set up Kaggle API first, or use a local CSV file:")
            print("   classifier = TextClassifier(data_path='your_file.csv')")
    else:
        print("\n Some tests failed. Please fix the issues above.")
        print("   Install missing packages: pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()
