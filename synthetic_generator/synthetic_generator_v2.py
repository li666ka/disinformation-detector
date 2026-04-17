"""
Enhanced Synthetic Fake News Generator with Text Augmentation
==============================================================
Methods for text diversity:
1. Celebrity Name Substitution
2. Back-translation (EN → DE/FR/ES → EN)
3. Synonym replacement
4. Sentence-level variation
5. Style variations (prefix, uncertainty words)
"""

import json
import random
import os
import csv
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import re
import numpy as np
from pathlib import Path

# ============================================================================
# CHECK AND INSTALL DEPENDENCIES
# ============================================================================

def install_dependencies():
    """Install required packages for augmentation."""
    import subprocess
    import sys
    
    packages = [
        "transformers",
        "sentencepiece",
        "torch",
        "nltk"
    ]
    
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

# Try to install dependencies
try:
    install_dependencies()
except:
    print("Warning: Some packages may not be installed. Run manually if needed.")

# ============================================================================
# AUGMENTATION IMPORTS
# ============================================================================

try:
    from transformers import MarianMTModel, MarianTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available. Back-translation disabled.")

try:
    import nltk
    from nltk.corpus import wordnet
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️ NLTK not available. Synonym replacement disabled.")

# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET_SYNTHETIC_COUNT = 12000

# Celebrity lists (same as before)
MALE_CELEBRITIES = [
    "Brad Pitt", "George Clooney", "Leonardo DiCaprio", "Tom Hanks",
    "Denzel Washington", "Tom Cruise", "Johnny Depp", "Robert Downey Jr.",
    "Chris Hemsworth", "Chris Evans", "Chris Pratt", "Ryan Reynolds",
    "Ryan Gosling", "Timothée Chalamet", "Austin Butler", "Justin Bieber",
    "Drake", "Kanye West", "Ed Sheeran", "Harry Styles", "The Weeknd",
    "Post Malone", "Bad Bunny", "Bruno Mars", "Shawn Mendes", "Tom Brady",
    "LeBron James", "Cristiano Ronaldo", "David Beckham", "Tiger Woods",
    "Patrick Mahomes", "Travis Kelce", "Dwayne Johnson", "Kevin Hart",
    "Will Smith", "Pete Davidson", "Ben Affleck", "Matt Damon",
    "Mark Wahlberg", "Hugh Jackman", "Jake Gyllenhaal", "Joaquin Phoenix",
    "Christian Bale", "Michael B. Jordan", "John Boyega", "Daniel Kaluuya",
    "Machine Gun Kelly", "Trevor Noah", "Jimmy Fallon", "Zac Efron",
    "Adam Driver", "Oscar Isaac", "Pedro Pascal", "Idris Elba",
]

FEMALE_CELEBRITIES = [
    "Jennifer Aniston", "Angelina Jolie", "Julia Roberts", "Meryl Streep",
    "Sandra Bullock", "Nicole Kidman", "Scarlett Johansson", "Emma Stone",
    "Emma Watson", "Natalie Portman", "Anne Hathaway", "Jennifer Lawrence",
    "Margot Robbie", "Gal Gadot", "Zendaya", "Florence Pugh", "Sydney Sweeney",
    "Taylor Swift", "Beyoncé", "Rihanna", "Lady Gaga", "Ariana Grande",
    "Selena Gomez", "Dua Lipa", "Billie Eilish", "Olivia Rodrigo", "Miley Cyrus",
    "Adele", "Katy Perry", "Shakira", "Cardi B", "Nicki Minaj", "Lizzo",
    "Doja Cat", "SZA", "Megan Thee Stallion", "Ice Spice",
    "Kim Kardashian", "Kylie Jenner", "Kendall Jenner", "Khloé Kardashian",
    "Kourtney Kardashian", "Paris Hilton", "Serena Williams",
    "Oprah Winfrey", "Ellen DeGeneres", "Blake Lively", "Megan Fox",
    "Jennifer Lopez", "Reese Witherspoon", "Charlize Theron", "Halle Berry",
    "Viola Davis", "Cate Blanchett", "Kate Winslet", "Amy Adams",
    "Jessica Chastain", "Lupita Nyong'o", "Jada Pinkett Smith",
    "Ana de Armas", "Anya Taylor-Joy", "Saoirse Ronan", "Dakota Johnson",
]

CELEBRITIES = MALE_CELEBRITIES + FEMALE_CELEBRITIES

# Gossip site URLs
FAKE_URLS = [
    "https://www.celebuzz.com/", "https://www.hollywoodlife.com/",
    "https://www.usmagazine.com/", "https://www.eonline.com/",
    "https://www.tmz.com/", "https://www.pagesix.com/",
    "https://www.dailymail.co.uk/", "https://www.thesun.co.uk/",
    "https://www.mirror.co.uk/", "https://www.nationalenquirer.com/",
    "https://www.radaronline.com/", "https://www.okmagazine.com/",
]

FAKE_AUTHORS = [
    "Staff Reporter", "Entertainment Desk", "Celebrity News Team",
    "Jessica Martinez", "Michael Thompson", "Sarah Johnson",
    "David Williams", "Emily Davis", "Chris Anderson",
    "Amanda Miller", "Ryan Garcia", "Nicole Brown",
]

TITLE_PREFIXES = [
    "EXCLUSIVE: ", "BREAKING: ", "", "", "",
    "REPORT: ", "SHOCKING: ", "REVEALED: ", "", "",
    "INSIDER: ", "", "", "BOMBSHELL: ", "",
]

# Tweet templates (same as before)
TWEET_TEMPLATES = {
    "shock": [
        "OMG I can't believe this about {celeb}! 😱",
        "Wait WHAT?! {celeb}?! No way!",
        "This {celeb} news is INSANE",
        "I'm shook about {celeb} rn",
        "NOT {celeb}!!! 😭😭😭",
        "Is this real about {celeb}???",
    ],
    "support": [
        "Sending love to {celeb} ❤️",
        "We stan {celeb} no matter what",
        "{celeb} deserves better tbh",
        "Leave {celeb} alone already!",
        "Team {celeb} forever 💪",
    ],
    "gossip": [
        "I KNEW something was up with {celeb}",
        "Tea about {celeb} has been brewing for months",
        "The {celeb} situation is messy",
        "Everyone saw this coming with {celeb}",
        "Spill the tea on {celeb} ☕",
    ],
    "skeptical": [
        "I don't believe this {celeb} story at all",
        "This sounds fake about {celeb}",
        "Another day another fake {celeb} rumor",
        "Source: trust me bro about {celeb}",
    ],
    "link_share": [
        "Just read this about {celeb} {url}",
        "{celeb} news 👀 {url}",
        "Y'all need to see this {celeb} story {url}",
    ],
}

HASHTAGS = [
    "#celebrity", "#gossip", "#breaking", "#news", "#hollywood",
    "#drama", "#tea", "#entertainment", "#exclusive", "#viral",
]


# ============================================================================
# BACK-TRANSLATION CLASS
# ============================================================================

class BackTranslator:
    """Back-translation for text augmentation."""
    
    def __init__(self, languages: List[str] = ["de", "fr", "es"]):
        self.languages = languages
        self.models = {}
        self.tokenizers = {}
        self.loaded = False
        
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️ Transformers not available. Back-translation disabled.")
            return
        
        print("Loading translation models (this may take a few minutes)...")
        
        for lang in languages:
            try:
                # EN -> LANG
                model_name_to = f"Helsinki-NLP/opus-mt-en-{lang}"
                self.models[f"en-{lang}"] = MarianMTModel.from_pretrained(model_name_to)
                self.tokenizers[f"en-{lang}"] = MarianTokenizer.from_pretrained(model_name_to)
                
                # LANG -> EN
                model_name_from = f"Helsinki-NLP/opus-mt-{lang}-en"
                self.models[f"{lang}-en"] = MarianMTModel.from_pretrained(model_name_from)
                self.tokenizers[f"{lang}-en"] = MarianTokenizer.from_pretrained(model_name_from)
                
                print(f"  ✓ Loaded {lang} models")
            except Exception as e:
                print(f"  ✗ Failed to load {lang} models: {e}")
        
        self.loaded = len(self.models) > 0
        if self.loaded:
            print(f"✓ Back-translation ready with {len(self.languages)} languages")
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text from source to target language."""
        key = f"{source_lang}-{target_lang}"
        if key not in self.models:
            return text
        
        try:
            model = self.models[key]
            tokenizer = self.tokenizers[key]
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Translate
            outputs = model.generate(**inputs, max_length=512)
            
            # Decode
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated
        except Exception as e:
            return text
    
    def back_translate(self, text: str, intermediate_lang: str = None) -> str:
        """Back-translate: EN -> intermediate -> EN."""
        if not self.loaded:
            return text
        
        if intermediate_lang is None:
            intermediate_lang = random.choice(self.languages)
        
        if f"en-{intermediate_lang}" not in self.models:
            return text
        
        try:
            # EN -> intermediate
            intermediate = self.translate(text, "en", intermediate_lang)
            
            # intermediate -> EN
            back = self.translate(intermediate, intermediate_lang, "en")
            
            return back
        except:
            return text


# ============================================================================
# SYNONYM REPLACEMENT
# ============================================================================

class SynonymReplacer:
    """Replace words with synonyms for text variation."""
    
    def __init__(self):
        self.available = NLTK_AVAILABLE
        
        # Custom synonyms for gossip vocabulary
        self.custom_synonyms = {
            "allegedly": ["reportedly", "supposedly", "apparently", "purportedly"],
            "reportedly": ["allegedly", "supposedly", "apparently", "it is said"],
            "claims": ["says", "states", "asserts", "maintains", "insists"],
            "revealed": ["disclosed", "shared", "divulged", "told", "confided"],
            "sources": ["insiders", "informants", "people close to", "those familiar"],
            "exclusive": ["breaking", "special", "first", "inside"],
            "shocking": ["stunning", "surprising", "jaw-dropping", "bombshell"],
            "furious": ["angry", "outraged", "livid", "enraged", "upset"],
            "devastated": ["heartbroken", "crushed", "shattered", "distraught"],
            "allegedly": ["reportedly", "supposedly", "purportedly", "apparently"],
            "secret": ["hidden", "private", "undisclosed", "confidential"],
            "spotted": ["seen", "photographed", "caught", "noticed"],
            "relationship": ["romance", "love affair", "partnership", "connection"],
            "split": ["breakup", "separation", "divorce", "parting"],
            "feud": ["rivalry", "conflict", "dispute", "beef", "tension"],
            "pregnant": ["expecting", "with child", "having a baby"],
            "married": ["wed", "tied the knot", "exchanged vows"],
            "dating": ["seeing", "romantically involved with", "going out with"],
            "friends": ["pals", "close associates", "confidants", "buddies"],
            "insiders": ["sources", "those close to", "people familiar with"],
        }
    
    def get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word."""
        word_lower = word.lower()
        
        # Check custom synonyms first
        if word_lower in self.custom_synonyms:
            return self.custom_synonyms[word_lower]
        
        # Use WordNet if available
        if self.available:
            synonyms = set()
            for syn in wordnet.synsets(word_lower):
                for lemma in syn.lemmas():
                    if lemma.name() != word_lower and '_' not in lemma.name():
                        synonyms.add(lemma.name())
            return list(synonyms)[:5]
        
        return []
    
    def replace_synonyms(self, text: str, replacement_prob: float = 0.15) -> str:
        """Replace some words with synonyms."""
        words = text.split()
        new_words = []
        
        for word in words:
            # Clean word for lookup
            clean_word = re.sub(r'[^\w]', '', word.lower())
            
            if random.random() < replacement_prob and len(clean_word) > 3:
                synonyms = self.get_synonyms(clean_word)
                if synonyms:
                    # Preserve original case and punctuation
                    new_word = random.choice(synonyms)
                    if word[0].isupper():
                        new_word = new_word.capitalize()
                    # Preserve trailing punctuation
                    trailing = re.search(r'[^\w]+$', word)
                    if trailing:
                        new_word += trailing.group()
                    new_words.append(new_word)
                    continue
            
            new_words.append(word)
        
        return ' '.join(new_words)


# ============================================================================
# TEXT AUGMENTATION PIPELINE
# ============================================================================

class TextAugmenter:
    """Combined text augmentation pipeline."""
    
    def __init__(self, use_back_translation: bool = True):
        self.use_back_translation = use_back_translation
        
        # Initialize components
        self.synonym_replacer = SynonymReplacer()
        
        if use_back_translation and TRANSFORMERS_AVAILABLE:
            self.back_translator = BackTranslator(languages=["de", "fr"])
        else:
            self.back_translator = None
            print("⚠️ Back-translation disabled")
        
        # Sentence variation templates
        self.intro_variations = [
            "Sources exclusively claim",
            "An insider exclusively revealed",
            "According to insiders",
            "Multiple sources confirm",
            "Sources close to the situation claim",
            "A source familiar with the matter said",
            "People close to the couple say",
            "Those familiar with the situation report",
            "Industry insiders are saying",
            "Well-placed sources indicate",
        ]
        
        self.attribution_variations = [
            "an insider revealed",
            "a source told us",
            "a close friend shared",
            "someone familiar with the matter said",
            "a person close to them confirmed",
            "an industry insider disclosed",
            "a source exclusively shared",
            "those in the know revealed",
        ]
    
    def augment_text(self, text: str, method: str = "random") -> str:
        """
        Augment text using various methods.
        
        Methods:
        - "back_translate": Use back-translation
        - "synonym": Replace synonyms
        - "variation": Apply sentence variations
        - "combined": Apply multiple methods
        - "random": Randomly choose a method
        """
        if method == "random":
            methods = ["synonym", "variation"]
            if self.back_translator and self.back_translator.loaded:
                methods.append("back_translate")
            method = random.choice(methods)
        
        if method == "back_translate" and self.back_translator and self.back_translator.loaded:
            # Back-translate sentences (not full text for speed)
            sentences = text.split('. ')
            if len(sentences) > 2:
                # Only back-translate 2-3 random sentences
                indices = random.sample(range(len(sentences)), min(3, len(sentences)))
                for idx in indices:
                    sentences[idx] = self.back_translator.back_translate(sentences[idx])
                return '. '.join(sentences)
            else:
                return self.back_translator.back_translate(text)
        
        elif method == "synonym":
            return self.synonym_replacer.replace_synonyms(text, replacement_prob=0.12)
        
        elif method == "variation":
            return self._apply_sentence_variations(text)
        
        elif method == "combined":
            # Apply multiple augmentations
            text = self._apply_sentence_variations(text)
            text = self.synonym_replacer.replace_synonyms(text, replacement_prob=0.08)
            return text
        
        return text
    
    def _apply_sentence_variations(self, text: str) -> str:
        """Apply sentence-level variations."""
        # Vary intro phrases
        for original in ["Sources exclusively claim", "Sources claim"]:
            if original in text:
                text = text.replace(original, random.choice(self.intro_variations), 1)
                break
        
        # Vary attribution phrases
        for original in ["an insider revealed", "a source revealed"]:
            if original in text:
                text = text.replace(original, random.choice(self.attribution_variations), 1)
                break
        
        # Vary uncertainty words
        uncertainty_map = {
            "allegedly": ["reportedly", "supposedly", "apparently", "purportedly"],
            "reportedly": ["allegedly", "supposedly", "apparently"],
            "supposedly": ["allegedly", "reportedly", "purportedly"],
        }
        
        for word, alternatives in uncertainty_map.items():
            if word in text.lower() and random.random() > 0.5:
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                replacement = random.choice(alternatives)
                text = pattern.sub(replacement, text, count=1)
                break
        
        return text


# ============================================================================
# HELPER FUNCTIONS (same as before)
# ============================================================================

def generate_id() -> str:
    return f"synthetic_{uuid.uuid4().hex[:12]}"

def generate_tweet_id() -> str:
    return str(random.randint(10**18, 10**19 - 1))

def generate_user_id() -> str:
    return str(random.randint(10**8, 10**10 - 1))

def random_date(start_year: int = 2016, end_year: int = 2019) -> str:
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    date = start + timedelta(days=random_days)
    return date.strftime("%Y-%m-%d %H:%M:%S")

def extract_celebrity_name(text: str) -> Tuple[str, str]:
    for celeb in MALE_CELEBRITIES:
        if celeb.lower() in text.lower():
            return celeb, "male"
    for celeb in FEMALE_CELEBRITIES:
        if celeb.lower() in text.lower():
            return celeb, "female"
    return "", "unknown"

def substitute_celebrity(text: str, original: str, replacement: str) -> str:
    if not original or not replacement:
        return text
    
    orig_parts = original.split()
    repl_parts = replacement.split()
    
    text = re.sub(re.escape(original), replacement, text, flags=re.IGNORECASE)
    
    if len(orig_parts) > 0 and len(repl_parts) > 0:
        text = re.sub(r'\b' + re.escape(orig_parts[0]) + r'\b', repl_parts[0], text, flags=re.IGNORECASE)
    
    if len(orig_parts) > 1 and len(repl_parts) > 1:
        text = re.sub(r'\b' + re.escape(orig_parts[-1]) + r'\b', repl_parts[-1], text, flags=re.IGNORECASE)
    
    return text

def vary_title_prefix(title: str) -> str:
    for prefix in ["EXCLUSIVE:", "BREAKING:", "REPORT:", "SHOCKING:", "REVEALED:", "INSIDER:", "BOMBSHELL:"]:
        if title.startswith(prefix):
            title = title[len(prefix):].strip()
            break
    new_prefix = random.choice(TITLE_PREFIXES)
    return new_prefix + title


# ============================================================================
# TWEET GENERATION (same as before)
# ============================================================================

def generate_tweet_count() -> int:
    mu, sigma = 2.5, 1.5
    count = int(np.random.lognormal(mu, sigma))
    return max(1, min(count, 2000))

def generate_username() -> str:
    prefixes = ["the", "real", "official", "its", "im", "just", ""]
    names = ["fan", "stan", "lover", "news", "updates", "daily", "gossip", "tea", "vibes"]
    suffixes = ["", "x", "xx", "2024", "2023", "_"]
    
    if random.random() < 0.5:
        words = ["happy", "sunny", "cool", "wild", "sweet", "star", "moon", "love", "dream"]
        name = random.choice(words) + random.choice(names)
    else:
        name = random.choice(prefixes) + random.choice(["celeb", "hollywood", "gossip", "tea"])
    
    name += random.choice(suffixes)
    name += str(random.randint(1, 999)) if random.random() < 0.4 else ""
    return name[:15]

def generate_tweet_text(celeb_name: str, news_url: str) -> str:
    category = random.choice(list(TWEET_TEMPLATES.keys()))
    template = random.choice(TWEET_TEMPLATES[category])
    tweet = template.format(celeb=celeb_name, url=news_url)
    
    if random.random() < 0.21:
        tweet += f" {random.choice(HASHTAGS)}"
    
    return tweet[:280]

def generate_tweets(celeb_name: str, news_url: str, publish_date: str) -> Dict:
    num_tweets = generate_tweet_count()
    tweets = []
    base_date = datetime.strptime(publish_date, "%Y-%m-%d %H:%M:%S")
    
    for _ in range(num_tweets):
        hours_offset = random.randint(0, 168)
        tweet_date = base_date + timedelta(hours=hours_offset)
        
        tweet = {
            "tweet_id": generate_tweet_id(),
            "user_id": generate_user_id(),
            "user_name": generate_username(),
            "text": generate_tweet_text(celeb_name, news_url),
            "created_at": tweet_date.strftime("%Y-%m-%d %H:%M:%S"),
        }
        tweets.append(tweet)
    
    return {"tweets": tweets}


# ============================================================================
# ENGAGEMENT GENERATION (same as before)
# ============================================================================

def generate_retweets(tweets_data: Dict) -> Dict:
    retweets = {}
    for tweet in tweets_data.get("tweets", []):
        tweet_id = tweet["tweet_id"]
        num_retweets = np.random.poisson(4)
        if num_retweets > 0:
            retweets[tweet_id] = [
                {
                    "user_id": generate_user_id(),
                    "user_name": generate_username(),
                    "created_at": tweet["created_at"],
                    "followers_count": int(np.random.lognormal(5.5, 2.0)),
                    "friends_count": int(np.random.lognormal(5.0, 1.5)),
                    "verified": random.random() < 0.012,
                }
                for _ in range(num_retweets)
            ]
    return retweets

def generate_likes(tweets_data: Dict) -> Dict:
    likes = {}
    for tweet in tweets_data.get("tweets", []):
        tweet_id = tweet["tweet_id"]
        num_likes = np.random.poisson(8)
        if num_likes > 0:
            likes[tweet_id] = [generate_user_id() for _ in range(num_likes)]
    return likes

def generate_replies(tweets_data: Dict, celeb_name: str) -> Dict:
    replies = {}
    reply_templates = [
        "This is crazy!", "I don't believe this", "Wow just wow",
        "Fake news", "This is so sad", "Leave them alone!",
        "I knew it!", "Not surprised tbh", f"Poor {celeb_name.split()[0]}",
        "The media is trash", "Who cares honestly", "👀👀👀", "😭😭😭",
    ]
    
    for tweet in tweets_data.get("tweets", []):
        tweet_id = tweet["tweet_id"]
        num_replies = np.random.poisson(1)
        if num_replies > 0:
            replies[tweet_id] = [
                {"user_id": generate_user_id(), "text": random.choice(reply_templates), "created_at": tweet["created_at"]}
                for _ in range(num_replies)
            ]
    return replies

def generate_user_profiles(tweets_data: Dict, retweets_data: Dict) -> List[Dict]:
    users = {}
    
    for tweet in tweets_data.get("tweets", []):
        user_id = tweet["user_id"]
        if user_id not in users:
            users[user_id] = {"user_id": user_id, "user_name": tweet["user_name"], "is_tweeter": True, "is_retweeter": False}
    
    for tweet_id, rt_list in retweets_data.items():
        for rt in rt_list:
            user_id = rt["user_id"]
            if user_id not in users:
                users[user_id] = {
                    "user_id": user_id, "user_name": rt["user_name"],
                    "is_tweeter": False, "is_retweeter": True,
                    "followers_count": rt.get("followers_count"),
                    "friends_count": rt.get("friends_count"),
                    "verified": rt.get("verified"),
                }
            else:
                users[user_id]["is_retweeter"] = True
    
    profiles = []
    for user_id, user_data in users.items():
        is_tweeter = user_data.get("is_tweeter", False)
        
        if is_tweeter:
            followers = int(np.random.lognormal(6.5, 2.0))
            friends = int(np.random.lognormal(6.2, 1.5))
            statuses = int(np.random.lognormal(9.0, 1.5))
            favourites = int(np.random.lognormal(7.2, 1.8))
            verified = random.random() < 0.053
        else:
            followers = user_data.get("followers_count") or int(np.random.lognormal(5.5, 2.0))
            friends = user_data.get("friends_count") or int(np.random.lognormal(5.0, 1.5))
            statuses = int(np.random.lognormal(7.5, 1.5))
            favourites = int(np.random.lognormal(6.0, 1.8))
            verified = user_data.get("verified", random.random() < 0.012)
        
        profiles.append({
            "user_id": user_id, "user_name": user_data["user_name"],
            "followers_count": followers, "friends_count": friends,
            "statuses_count": statuses, "favourites_count": favourites,
            "verified": verified, "geo_enabled": random.random() < 0.42,
            "protected": random.random() < 0.05,
            "is_tweeter": user_data.get("is_tweeter", False),
            "is_retweeter": user_data.get("is_retweeter", False),
        })
    
    return profiles


# ============================================================================
# NEWS GENERATION WITH AUGMENTATION
# ============================================================================

def generate_news_article(seed: Dict, replacement_celeb: str, augmenter: TextAugmenter) -> Tuple[Dict, str]:
    """Generate news article with text augmentation."""
    title = seed["title"]
    text = seed["text"]
    
    # Extract and substitute celebrity
    original_celeb, gender = extract_celebrity_name(title + " " + text)
    
    if replacement_celeb and original_celeb:
        title = substitute_celebrity(title, original_celeb, replacement_celeb)
        text = substitute_celebrity(text, original_celeb, replacement_celeb)
        celeb_name = replacement_celeb
    else:
        celeb_name = original_celeb or "the celebrity"
    
    # Apply text augmentation
    augmentation_method = random.choice(["synonym", "variation", "combined"])
    text = augmenter.augment_text(text, method=augmentation_method)
    
    # Vary title
    title = vary_title_prefix(title)
    
    # Sometimes augment title too
    if random.random() < 0.3:
        title = augmenter.augment_text(title, method="synonym")
    
    publish_date = random_date()
    url = random.choice(FAKE_URLS) + generate_id()
    
    article = {
        "title": title,
        "text": text,
        "url": url,
        "authors": [random.choice(FAKE_AUTHORS)],
        "publish_date": publish_date,
        "source": random.choice(FAKE_URLS).replace("https://www.", "").replace("/", ""),
    }
    
    return article, celeb_name


def generate_single_news(seed: Dict, output_dir: str, news_id: str,
                         replacement_celeb: str, augmenter: TextAugmenter) -> bool:
    """Generate all files for a single news item."""
    try:
        news_dir = os.path.join(output_dir, news_id)
        os.makedirs(news_dir, exist_ok=True)
        
        article, celeb_name = generate_news_article(seed, replacement_celeb, augmenter)
        tweets = generate_tweets(celeb_name, article["url"], article["publish_date"])
        retweets = generate_retweets(tweets)
        likes = generate_likes(tweets)
        replies = generate_replies(tweets, celeb_name)
        users = generate_user_profiles(tweets, retweets)
        
        with open(os.path.join(news_dir, "news_article.json"), 'w') as f:
            json.dump(article, f, indent=2)
        with open(os.path.join(news_dir, "tweets.json"), 'w') as f:
            json.dump(tweets, f, indent=2)
        with open(os.path.join(news_dir, "retweets.json"), 'w') as f:
            json.dump(retweets, f, indent=2)
        with open(os.path.join(news_dir, "likes.json"), 'w') as f:
            json.dump(likes, f, indent=2)
        with open(os.path.join(news_dir, "replies.json"), 'w') as f:
            json.dump(replies, f, indent=2)
        
        if users:
            with open(os.path.join(news_dir, "new_user.tsv"), 'w', newline='') as f:
                fieldnames = ["user_id", "user_name", "followers_count", "friends_count",
                             "statuses_count", "favourites_count", "verified",
                             "geo_enabled", "protected", "is_tweeter", "is_retweeter"]
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
                writer.writeheader()
                writer.writerows(users)
        
        return True
    except Exception as e:
        print(f"Error generating {news_id}: {e}")
        return False


# ============================================================================
# MAIN
# ============================================================================

def load_seeds(seeds_dir: str) -> List[Dict]:
    all_seeds = []
    for filename in os.listdir(seeds_dir):
        if filename.endswith("_seeds.json"):
            with open(os.path.join(seeds_dir, filename), 'r') as f:
                data = json.load(f)
                for seed in data.get("seeds", []):
                    seed["category"] = data.get("category", "UNKNOWN")
                    all_seeds.append(seed)
    return all_seeds


def main(seeds_dir: str, output_dir: str, target_count: int = TARGET_SYNTHETIC_COUNT,
         use_back_translation: bool = False):
    """Main generation pipeline."""
    
    print("=" * 60)
    print("ENHANCED SYNTHETIC FAKE NEWS GENERATOR")
    print("=" * 60)
    
    print(f"\n📂 Loading seeds from {seeds_dir}...")
    seeds = load_seeds(seeds_dir)
    print(f"   Loaded {len(seeds)} seeds")
    
    print(f"\n🔧 Initializing text augmenter...")
    augmenter = TextAugmenter(use_back_translation=use_back_translation)
    
    print(f"\n🚀 Generating {target_count} synthetic news articles...")
    os.makedirs(output_dir, exist_ok=True)
    
    generated = 0
    failed = 0
    
    while generated < target_count:
        for seed in seeds:
            if generated >= target_count:
                break
            
            _, gender = extract_celebrity_name(seed["title"] + " " + seed["text"])
            
            if gender == "male":
                celeb_pool = MALE_CELEBRITIES
            elif gender == "female":
                celeb_pool = FEMALE_CELEBRITIES
            else:
                celeb_pool = CELEBRITIES
            
            replacement_celeb = random.choice(celeb_pool)
            news_id = f"synthetic_fake_{generated:05d}"
            
            success = generate_single_news(seed, output_dir, news_id, replacement_celeb, augmenter)
            
            if success:
                generated += 1
                if generated % 500 == 0:
                    print(f"   Generated {generated}/{target_count}...")
            else:
                failed += 1
    
    print(f"\n" + "=" * 60)
    print(f"✅ GENERATION COMPLETE!")
    print(f"   Generated: {generated}")
    print(f"   Failed: {failed}")
    print(f"   Output: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic fake news with augmentation")
    parser.add_argument("--seeds", default="./seeds", help="Seeds directory")
    parser.add_argument("--output", default="./synthetic_fake_v2", help="Output directory")
    parser.add_argument("--count", type=int, default=TARGET_SYNTHETIC_COUNT, help="Target count")
    parser.add_argument("--back-translate", action="store_true", help="Enable back-translation (slow)")
    
    args = parser.parse_args()
    main(args.seeds, args.output, args.count, args.back_translate)
