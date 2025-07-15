import os
import nltk
import ssl
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import JSONParser
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keybert import KeyBERT

# NLTK setup (as before)
def setup_nltk():
    try:
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
        required_resources = [
            'punkt',
            'averaged_perceptron_tagger',
            'stopwords',
            'wordnet',
            'omw-1.4'
        ]
        for resource in required_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)
    except Exception as e:
        print(f"❌ NLTK setup failed: {e}")

setup_nltk()

class ImprovedTitleSuggestionView(APIView):
    """
    Improved blog title suggestion using local NLP (KeyBERT, NLTK).
    """
    parser_classes = [JSONParser]
    http_method_names = ['post']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.kw_model = KeyBERT(model='all-MiniLM-L6-v2')  # Local model

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = [self.lemmatizer.lemmatize(w) for w in text.split() if w not in self.stop_words]
        return ' '.join(tokens)

    def extract_keyphrases(self, text, top_n=5):
        keyphrases = self.kw_model.extract_keywords(
            text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=top_n
        )
        # Remove substrings and duplicates
        phrases = []
        for phrase, _ in keyphrases:
            if not any(phrase in p or p in phrase for p in phrases):
                phrases.append(phrase)
        return phrases

    def generate_titles(self, keyphrases):
        templates = [
            "The Ultimate Guide to {}",
            "How to Improve {}",
            "Top 10 Tips for {}",
            "Understanding {} in Depth",
            "A Beginner's Guide to {}",
            "Exploring the Importance of {}",
            "Mastering {}: What You Need to Know"
        ]
        titles = []
        for phrase in keyphrases:
            for template in templates:
                title = template.format(phrase.title())
                # Avoid titles with repeated words/phrases
                words = title.lower().split()
                if len(set(words)) == len(words) and 5 <= len(words) <= 12:
                    titles.append(title)
        # Remove duplicates
        return list(dict.fromkeys(titles))

    def post(self, request):
        content = request.data.get('content')
        if not content:
            return Response({'error': 'No content provided'}, status=400)
        clean = self.clean_text(content)
        keyphrases = self.extract_keyphrases(clean)
        if not keyphrases:
            return Response({'suggestions': ["A Comprehensive Guide", "Everything You Need to Know", "The Complete Overview"]})
        titles = self.generate_titles(keyphrases)
        return Response({'suggestions': titles[:3]})

# Optionally, you can alias the view for your URL conf
TitleSuggestionView = ImprovedTitleSuggestionView 