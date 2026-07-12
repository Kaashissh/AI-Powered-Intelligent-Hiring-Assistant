# download_nltk_data.py - Download required NLTK data

import nltk
import ssl

print("Downloading NLTK data...")

try:
    # Try to create an unverified HTTPS context (for some networks)
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Download required NLTK data (updated for newer NLTK versions)
    print("Downloading 'punkt' tokenizer...")
    nltk.download('punkt')
    
    print("Downloading 'punkt_tab' tokenizer (newer version)...")
    nltk.download('punkt_tab')
    
    print("Downloading 'stopwords'...")
    nltk.download('stopwords')
    
    print("Downloading 'wordnet' lemmatizer...")
    nltk.download('wordnet')
    
    print("Downloading 'omw-1.4' (wordnet requirement)...")
    nltk.download('omw-1.4')
    
    print("✅ All NLTK data downloaded successfully!")
    
    # Test the downloads
    print("\nTesting downloads...")
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    
    # Test stopwords
    stop_words = set(stopwords.words('english'))
    print(f"✅ Stopwords loaded: {len(stop_words)} words")
    
    # Test tokenizer
    tokens = word_tokenize("This is a test sentence.")
    print(f"✅ Tokenizer working: {tokens}")
    
    # Test lemmatizer
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize("running", "v")
    print(f"✅ Lemmatizer working: 'running' -> '{lemma}'")
    
    print("\n🎉 NLTK setup completed successfully!")
    print("You can now run: python resume_matcher.py")
    
except Exception as e:
    print(f"❌ Error downloading NLTK data: {e}")
    print("\nTry running these commands manually in Python:")
    print(">>> import nltk")
    print(">>> nltk.download('punkt')")
    print(">>> nltk.download('stopwords')")
    print(">>> nltk.download('wordnet')")
    
    # Alternative: download all data
    print("\nOr download all NLTK data (large download):")
    print(">>> nltk.download('all')")