import os
import nltk
import ssl

def setup_nltk():
    """Download required NLTK data"""
    try:
        # Handle SSL certificate issues if any
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Set NLTK data path
        nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
        
        # Download required NLTK resources
        required_resources = [
            'punkt_tab',
            'punkt',
            'averaged_perceptron_tagger',
            'stopwords',
            'wordnet',
            'omw-1.4'  # Additional resource that might be needed
        ]
        
        for resource in required_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
                print(f"✅ {resource} already available")
            except LookupError:
                try:
                    print(f"📥 Downloading {resource}...")
                    nltk.download(resource, quiet=True)
                    print(f"✅ {resource} downloaded successfully")
                except Exception as e:
                    print(f"❌ Failed to download {resource}: {e}")
    
    except Exception as e:
        print(f"❌ NLTK setup failed: {e}")

# Call the setup function
setup_nltk()

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import JSONParser
import re
import random
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from string import punctuation
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
try:
    # nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    # nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    nltk.download('wordnet')

class TextPreprocessor:
    """Helper class for text preprocessing"""
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def extract_sentences(self, text):
        """Extract and clean sentences"""
        # Use regex for basic sentence splitting as a fallback
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [sent.strip() for sent in sentences if len(sent.strip()) > 10]
    
    def extract_noun_phrases(self, text):
        """Extract noun phrases from text"""
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        noun_phrases = []
        current_phrase = []
        
        for word, tag in tagged:
            if tag.startswith(('NN', 'JJ')):  # Nouns and adjectives
                current_phrase.append(word)
            elif current_phrase:
                if len(current_phrase) > 1:
                    noun_phrases.append(' '.join(current_phrase))
                current_phrase = []
                
        if current_phrase and len(current_phrase) > 1:
            noun_phrases.append(' '.join(current_phrase))
            
        return noun_phrases

class KeywordExtractor:
    """Helper class for keyword extraction using TF-IDF"""
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
    def extract_keywords(self, text, num_keywords=5):
        """Extract important keywords using TF-IDF"""
        # Handle single document case
        if isinstance(text, str):
            text = [text]
            
        # Fit and transform the text
        tfidf_matrix = self.tfidf.fit_transform(text)
        
        # Get feature names and scores
        feature_names = np.array(self.tfidf.get_feature_names_out())
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        # Sort by score and get top keywords
        top_indices = tfidf_scores.argsort()[-num_keywords:][::-1]
        return feature_names[top_indices].tolist()

class TopicExtractor:
    """Helper class for topic extraction"""
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        
    def extract_topics(self, text):
        """Extract main topics from text using noun phrase analysis"""
        # Clean text
        clean_text = self.preprocessor.clean_text(text)
        
        # Extract noun phrases
        noun_phrases = self.preprocessor.extract_noun_phrases(text)
        
        # Count phrase frequencies
        phrase_freq = Counter(noun_phrases)
        
        # Get top phrases
        top_phrases = [phrase for phrase, _ in phrase_freq.most_common(5)]
        
        return top_phrases

class ReadabilityScorer:
    """Helper class for scoring title readability"""
    def __init__(self):
        self.ideal_length = (4, 12)  # Ideal word count range
        
    def score_title(self, title):
        """Score a title based on readability metrics"""
        words = title.split()
        word_count = len(words)
        
        # Initialize score
        score = 1.0
        
        # Length score
        if self.ideal_length[0] <= word_count <= self.ideal_length[1]:
            score *= 1.2
        elif word_count < self.ideal_length[0]:
            score *= 0.8
        else:
            score *= 0.6
            
        # Capitalization score
        if title[0].isupper() and title[1:].islower():
            score *= 1.2
            
        # Question mark bonus
        if title.endswith('?'):
            score *= 1.1
            
        # Number presence penalty
        if any(char.isdigit() for char in title):
            score *= 0.9
            
        return score

class TitleRanker:
    """Helper class for ranking generated titles"""
    def __init__(self):
        self.readability_scorer = ReadabilityScorer()
        
    def rank_titles(self, titles, keywords=None):
        """Rank titles based on multiple criteria"""
        scored_titles = []
        
        for title in titles:
            score = self.readability_scorer.score_title(title)
            
            # Keyword presence bonus
            if keywords:
                title_lower = title.lower()
                keyword_matches = sum(1 for kw in keywords if kw.lower() in title_lower)
                score *= (1 + 0.1 * keyword_matches)
            
            scored_titles.append((title, score))
            
        # Sort by score descending
        scored_titles.sort(key=lambda x: x[1], reverse=True)
        
        # Return only the titles, without scores
        return [title for title, _ in scored_titles]

class TitleSuggestionView(APIView):
    """
    Generate title suggestions for blog content.

    Example request format:
    ```json
    {
        "content": "Your blog post content goes here. This can be a paragraph or multiple paragraphs of text that you want to generate titles for."
    }
    ```

    Returns a list of suggested titles based on the provided content.
    """
    parser_classes = [JSONParser]
    http_method_names = ['post']  # Only allow POST requests

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocessor = TextPreprocessor()
        self.keyword_extractor = KeywordExtractor()
        self.topic_extractor = TopicExtractor()
        self.title_ranker = TitleRanker()

    def generate_title(self, content):
        """Generate a title based on content using advanced NLP techniques."""
        # Clean and prepare the content
        content = content.strip()
        sentences = self.preprocessor.extract_sentences(content)
        
        if not sentences:
            return ["A Comprehensive Guide", "Everything You Need to Know", "The Complete Overview"]
        
        titles = []
        
        # Strategy 1: Extract keywords and create keyword-based titles
        keywords = self.keyword_extractor.extract_keywords(content)
        if keywords:
            # Create title from top keywords
            keyword_title = ' '.join(keywords[:3]).title()
            titles.append(keyword_title)
            
            # Create "How to" or "Guide to" title with keywords
            if len(keywords) >= 2:
                guide_title = f"Guide to {' '.join(keywords[:2]).title()}"
                how_to_title = f"How to {' '.join(keywords[:2]).title()}"
                titles.extend([guide_title, how_to_title])
        
        # Strategy 2: Use topic extraction
        topics = self.topic_extractor.extract_topics(content)
        if topics:
            for topic in topics[:2]:
                topic_title = f"Understanding {topic.title()}"
                titles.append(topic_title)
                
                # Create a question-based title
                question_title = f"Why {topic.title()} Matters?"
                titles.append(question_title)
        
        # Strategy 3: Use first sentence with modifications
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence.split()) > 8:
                first_sentence = ' '.join(first_sentence.split()[:8]) + '...'
            titles.append(first_sentence.capitalize())
        
        # Strategy 4: Create pattern-based titles using keywords and topics
        patterns = [
            "The Ultimate Guide to {}",
            "{}: A Comprehensive Overview",
            "Everything You Need to Know About {}",
            "Mastering {}: Tips and Techniques",
            "The Complete Guide to {}"
        ]
        
        if keywords:
            for pattern in patterns[:2]:
                titles.append(pattern.format(keywords[0].title()))
        
        if topics:
            for pattern in patterns[2:4]:
                titles.append(pattern.format(topics[0].title()))
        
        # Remove duplicates
        titles = list(dict.fromkeys(titles))
        
        # Rank titles using our scoring system
        ranked_titles = self.title_ranker.rank_titles(titles, keywords)
        
        # Return top 3 titles
        return ranked_titles[:3]

    def post(self, request):
        try:
            # Get the blog content from the request
            content = request.data.get('content')
            if not content:
                return Response({
                    'error': 'No content provided',
                    'help': 'Please provide a JSON body with a "content" field'
                }, status=status.HTTP_400_BAD_REQUEST)

            # Generate title suggestions
            suggestions = self.generate_title(content)

            return Response({'suggestions': suggestions}, status=status.HTTP_200_OK)

        except Exception as e:
            import traceback
            print("Error details:", str(e))
            print("Traceback:", traceback.format_exc())
            return Response({
                'error': str(e),
                'help': 'Make sure to send a POST request with Content-Type: application/json and a JSON body containing a "content" field'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR) 