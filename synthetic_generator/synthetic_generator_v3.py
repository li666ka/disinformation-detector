"""
Ultimate Synthetic Fake News Generator v3
==========================================
All augmentation methods combined (NO back-translation):

1. Celebrity Name Substitution
2. Synonym Replacement (custom gossip vocabulary + WordNet)
3. Sentence Shuffling
4. Sentence Deletion/Insertion
5. Paraphrasing with Flan-T5
6. Contextual Word Replacement (BERT MLM)
7. Detail Variations (locations, sources, timeframes)
8. Style Variations (prefixes, uncertainty words)
"""

import json
import random
import os
import csv
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import re
import numpy as np
from pathlib import Path

# ============================================================================
# DEPENDENCY MANAGEMENT
# ============================================================================

def install_dependencies():
    """Install required packages."""
    import subprocess
    import sys
    
    packages = [
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("nltk", "nltk"),
    ]
    
    for import_name, pip_name in packages:
        try:
            __import__(import_name)
        except ImportError:
            print(f"Installing {pip_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name, "-q"])

print("🔧 Checking dependencies...")
try:
    install_dependencies()
except Exception as e:
    print(f"⚠️ Could not install some dependencies: {e}")

# ============================================================================
# IMPORTS WITH FALLBACKS
# ============================================================================

# Transformers for paraphrasing and BERT MLM
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForSeq2SeqLM,
        AutoModelForMaskedLM,
        pipeline
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
    print("✓ Transformers loaded")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available")

# NLTK for synonyms
try:
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt', quiet=True)
    from nltk.corpus import wordnet
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
    print("✓ NLTK loaded")
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️ NLTK not available")


# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET_SYNTHETIC_COUNT = 10639

# Extended celebrity lists
MALE_CELEBRITIES = [
    # A-list actors
    "Brad Pitt", "George Clooney", "Leonardo DiCaprio", "Tom Hanks",
    "Denzel Washington", "Tom Cruise", "Johnny Depp", "Robert Downey Jr.",
    "Chris Hemsworth", "Chris Evans", "Chris Pratt", "Ryan Reynolds",
    "Ryan Gosling", "Timothée Chalamet", "Austin Butler", "Zac Efron",
    "Adam Driver", "Oscar Isaac", "Pedro Pascal", "Idris Elba",
    "Michael B. Jordan", "John Boyega", "Daniel Kaluuya", "Chadwick Boseman",
    "Henry Cavill", "Jason Momoa", "Keanu Reeves", "Paul Rudd",
    "Chris Pine", "Sebastian Stan", "Tom Hiddleston", "Benedict Cumberbatch",
    "Andrew Garfield", "Tobey Maguire", "Jacob Elordi", "Barry Keoghan",
    
    # Musicians
    "Justin Bieber", "Drake", "Kanye West", "Ed Sheeran", "Harry Styles",
    "The Weeknd", "Post Malone", "Bad Bunny", "Bruno Mars", "Shawn Mendes",
    "Travis Scott", "Lil Nas X", "Jack Harlow", "Machine Gun Kelly",
    "John Legend", "Adam Levine", "Nick Jonas", "Joe Jonas", "Zayn Malik",
    
    # Athletes
    "Tom Brady", "LeBron James", "Cristiano Ronaldo", "David Beckham",
    "Tiger Woods", "Patrick Mahomes", "Travis Kelce", "Steph Curry",
    "Aaron Rodgers", "Odell Beckham Jr.", "Neymar", "Lionel Messi",
    
    # Other
    "Dwayne Johnson", "Kevin Hart", "Will Smith", "Pete Davidson",
    "Ben Affleck", "Matt Damon", "Mark Wahlberg", "Hugh Jackman",
    "Jake Gyllenhaal", "Joaquin Phoenix", "Christian Bale",
    "Trevor Noah", "Jimmy Fallon", "Jimmy Kimmel", "Seth Meyers",
    "Elon Musk", "Jeff Bezos", "Mark Zuckerberg",
]

FEMALE_CELEBRITIES = [
    # A-list actors
    "Jennifer Aniston", "Angelina Jolie", "Julia Roberts", "Meryl Streep",
    "Sandra Bullock", "Nicole Kidman", "Scarlett Johansson", "Emma Stone",
    "Emma Watson", "Natalie Portman", "Anne Hathaway", "Jennifer Lawrence",
    "Margot Robbie", "Gal Gadot", "Zendaya", "Florence Pugh", "Sydney Sweeney",
    "Ana de Armas", "Anya Taylor-Joy", "Saoirse Ronan", "Dakota Johnson",
    "Kristen Stewart", "Emma Roberts", "Lily Collins", "Hailee Steinfeld",
    "Elle Fanning", "Millie Bobby Brown", "Jenna Ortega", "Sadie Sink",
    "Reese Witherspoon", "Charlize Theron", "Halle Berry", "Viola Davis",
    "Cate Blanchett", "Kate Winslet", "Amy Adams", "Jessica Chastain",
    "Lupita Nyong'o", "Awkwafina", "Constance Wu", "Gemma Chan",
    
    # Musicians
    "Taylor Swift", "Beyoncé", "Rihanna", "Lady Gaga", "Ariana Grande",
    "Selena Gomez", "Dua Lipa", "Billie Eilish", "Olivia Rodrigo", "Miley Cyrus",
    "Adele", "Katy Perry", "Shakira", "Cardi B", "Nicki Minaj", "Lizzo",
    "Doja Cat", "SZA", "Megan Thee Stallion", "Ice Spice", "Sabrina Carpenter",
    "Halsey", "Demi Lovato", "Camila Cabello", "Normani", "Kehlani",
    
    # Reality TV / Influencers
    "Kim Kardashian", "Kylie Jenner", "Kendall Jenner", "Khloé Kardashian",
    "Kourtney Kardashian", "Kris Jenner", "Paris Hilton", "Hailey Bieber",
    "Gigi Hadid", "Bella Hadid", "Cara Delevingne", "Emily Ratajkowski",
    
    # Athletes
    "Serena Williams", "Simone Biles", "Naomi Osaka", "Alex Morgan",
    
    # Other
    "Oprah Winfrey", "Ellen DeGeneres", "Blake Lively", "Megan Fox",
    "Jennifer Lopez", "Jada Pinkett Smith", "Priyanka Chopra", "Sofia Vergara",
    "Eva Longoria", "Salma Hayek", "Penelope Cruz", "Marion Cotillard",
]

CELEBRITIES = MALE_CELEBRITIES + FEMALE_CELEBRITIES

# Locations for variety
LOCATIONS = [
    "Los Angeles", "New York City", "Miami", "London", "Paris",
    "Las Vegas", "Malibu", "Beverly Hills", "Bel Air", "Hollywood",
    "Manhattan", "Brooklyn", "the Hamptons", "Calabasas", "Hidden Hills",
    "Ibiza", "St. Tropez", "Monaco", "Dubai", "Sydney",
    "Toronto", "Vancouver", "Nashville", "Austin", "Atlanta",
]

# Sources/informants for variety  
SOURCE_TYPES = [
    "a close friend", "an insider", "a source close to the couple",
    "someone familiar with the situation", "a family member",
    "a longtime friend", "a production insider", "a music industry source",
    "a Hollywood insider", "someone close to the star",
    "a person with direct knowledge", "an entertainment lawyer",
    "a member of their inner circle", "a trusted confidant",
    "multiple sources", "several insiders", "people familiar with the matter",
]

# Time expressions
TIME_EXPRESSIONS = [
    "last week", "recently", "over the weekend", "earlier this month",
    "just days ago", "in recent weeks", "this past Monday",
    "late last night", "yesterday afternoon", "this morning",
    "for months now", "for the past several weeks", "since last year",
]

# Gossip site URLs
FAKE_URLS = [
    "https://www.celebuzz.com/", "https://www.hollywoodlife.com/",
    "https://www.usmagazine.com/", "https://www.eonline.com/",
    "https://www.tmz.com/", "https://www.pagesix.com/",
    "https://www.dailymail.co.uk/", "https://www.thesun.co.uk/",
    "https://www.mirror.co.uk/", "https://www.nationalenquirer.com/",
    "https://www.radaronline.com/", "https://www.okmagazine.com/",
    "https://www.intouchweekly.com/", "https://www.lifeandstylemag.com/",
]

FAKE_AUTHORS = [
    "Staff Reporter", "Entertainment Desk", "Celebrity News Team",
    "Jessica Martinez", "Michael Thompson", "Sarah Johnson",
    "David Williams", "Emily Davis", "Chris Anderson",
    "Amanda Miller", "Ryan Garcia", "Nicole Brown",
    "Jason Lee", "Michelle Taylor", "Brandon Wilson",
    "Sophia Chen", "Marcus Johnson", "Ashley Williams",
]

TITLE_PREFIXES = [
    "EXCLUSIVE: ", "BREAKING: ", "", "", "", "",
    "REPORT: ", "SHOCKING: ", "REVEALED: ", "", "",
    "INSIDER: ", "", "", "BOMBSHELL: ", "", "",
    "SOURCES: ", "", "JUST IN: ", "", "",
]

# Tweet templates
TWEET_TEMPLATES = {
    "shock": [
        "OMG I can't believe this about {celeb}! 😱",
        "Wait WHAT?! {celeb}?! No way!",
        "This {celeb} news is INSANE",
        "I'm shook about {celeb} rn",
        "NOT {celeb}!!! 😭😭😭",
        "Is this real about {celeb}???",
        "I literally gasped at the {celeb} news",
        "My jaw DROPPED reading about {celeb}",
    ],
    "support": [
        "Sending love to {celeb} ❤️",
        "We stan {celeb} no matter what",
        "{celeb} deserves better tbh",
        "Leave {celeb} alone already!",
        "Team {celeb} forever 💪",
        "Praying for {celeb} 🙏",
        "{celeb} we love you!",
    ],
    "gossip": [
        "I KNEW something was up with {celeb}",
        "Tea about {celeb} has been brewing for months",
        "The {celeb} situation is messy",
        "Everyone saw this coming with {celeb}",
        "Spill the tea on {celeb} ☕",
        "Not {celeb} being messy again 💀",
    ],
    "skeptical": [
        "I don't believe this {celeb} story at all",
        "This sounds fake about {celeb}",
        "Another day another fake {celeb} rumor",
        "Source: trust me bro about {celeb}",
        "The media lies about {celeb} constantly",
        "Where's the proof about {celeb}? 🤔",
    ],
    "link_share": [
        "Just read this about {celeb} {url}",
        "{celeb} news 👀 {url}",
        "Y'all need to see this {celeb} story {url}",
        "OMG {celeb}!!! {url}",
    ],
}

HASHTAGS = [
    "#celebrity", "#gossip", "#breaking", "#news", "#hollywood",
    "#drama", "#tea", "#entertainment", "#exclusive", "#viral",
    "#trending", "#celebnews", "#hotgossip", "#breakingnews",
]


# ============================================================================
# AUGMENTATION CLASSES
# ============================================================================

class ParaphraserT5:
    """Paraphrasing using Flan-T5."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.loaded = False
        
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️ Paraphraser: Transformers not available")
            return
        
        try:
            print("Loading Flan-T5 for paraphrasing...")
            model_name = "google/flan-t5-small"  # Use small for speed
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.loaded = True
            print("✓ Flan-T5 loaded")
        except Exception as e:
            print(f"⚠️ Could not load Flan-T5: {e}")
    
    def paraphrase(self, text: str) -> str:
        """Paraphrase a sentence using T5."""
        if not self.loaded or len(text) < 20:
            return text
        
        try:
            prompt = f"Paraphrase this sentence: {text}"
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
            
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                temperature=0.8,
                do_sample=True,
                early_stopping=True
            )
            
            paraphrased = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Only use if it's different enough
            if paraphrased and len(paraphrased) > 10 and paraphrased.lower() != text.lower():
                return paraphrased
            return text
        except:
            return text


class BERTWordReplacer:
    """Contextual word replacement using BERT MLM."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.loaded = False
        
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️ BERT replacer: Transformers not available")
            return
        
        try:
            print("Loading BERT for contextual replacement...")
            self.pipe = pipeline("fill-mask", model="bert-base-uncased")
            self.loaded = True
            print("✓ BERT loaded")
        except Exception as e:
            print(f"⚠️ Could not load BERT: {e}")
    
    def replace_words(self, text: str, n_replacements: int = 2) -> str:
        """Replace n random words with contextually similar ones."""
        if not self.loaded:
            return text
        
        try:
            words = text.split()
            if len(words) < 10:
                return text
            
            # Find replaceable words (longer words, not names)
            replaceable_indices = [
                i for i, w in enumerate(words)
                if len(w) > 4 and w.isalpha() and w[0].islower()
            ]
            
            if not replaceable_indices:
                return text
            
            # Replace n random words
            indices_to_replace = random.sample(
                replaceable_indices, 
                min(n_replacements, len(replaceable_indices))
            )
            
            for idx in indices_to_replace:
                original_word = words[idx]
                masked_text = ' '.join(words[:idx] + ['[MASK]'] + words[idx+1:])
                
                try:
                    results = self.pipe(masked_text[:512])
                    if results:
                        # Get a random suggestion from top 5
                        new_word = random.choice(results[:5])['token_str']
                        if new_word and new_word != original_word:
                            words[idx] = new_word
                except:
                    continue
            
            return ' '.join(words)
        except:
            return text


class SynonymReplacer:
    """Enhanced synonym replacement with gossip vocabulary."""
    
    def __init__(self):
        self.custom_synonyms = {
            # Uncertainty words
            "allegedly": ["reportedly", "supposedly", "apparently", "purportedly", "it is claimed"],
            "reportedly": ["allegedly", "supposedly", "apparently", "according to reports"],
            "supposedly": ["allegedly", "reportedly", "purportedly", "as claimed"],
            "apparently": ["reportedly", "allegedly", "seemingly", "it appears"],
            
            # Source words
            "sources": ["insiders", "informants", "people close to", "those familiar with"],
            "insider": ["source", "informant", "person close to", "confidant"],
            "source": ["insider", "informant", "person familiar with the situation"],
            
            # Action words
            "revealed": ["disclosed", "shared", "divulged", "told", "confided", "admitted"],
            "claims": ["says", "states", "asserts", "maintains", "insists", "alleges"],
            "confirmed": ["verified", "corroborated", "validated", "affirmed"],
            "denied": ["refuted", "rejected", "dismissed", "disputed"],
            
            # Emotional words
            "shocking": ["stunning", "surprising", "jaw-dropping", "bombshell", "startling"],
            "furious": ["angry", "outraged", "livid", "enraged", "fuming", "incensed"],
            "devastated": ["heartbroken", "crushed", "shattered", "distraught", "gutted"],
            "thrilled": ["excited", "overjoyed", "delighted", "ecstatic", "elated"],
            "worried": ["concerned", "anxious", "troubled", "uneasy", "apprehensive"],
            
            # Relationship words
            "relationship": ["romance", "love affair", "partnership", "connection", "bond"],
            "split": ["breakup", "separation", "divorce", "parting", "end"],
            "feud": ["rivalry", "conflict", "dispute", "beef", "tension", "rift"],
            "pregnant": ["expecting", "with child", "having a baby", "awaiting a child"],
            "married": ["wed", "tied the knot", "exchanged vows", "walked down the aisle"],
            "dating": ["seeing", "romantically involved with", "going out with", "courting"],
            
            # Adjectives
            "secret": ["hidden", "private", "undisclosed", "confidential", "covert"],
            "exclusive": ["special", "first", "inside", "breaking", "unique"],
            "close": ["intimate", "trusted", "longtime", "dear"],
            
            # Nouns
            "friends": ["pals", "close associates", "confidants", "buddies", "companions"],
            "couple": ["pair", "duo", "partners", "lovebirds", "two"],
            "marriage": ["union", "relationship", "partnership", "wedlock"],
            "rumors": ["speculation", "gossip", "talk", "whispers", "reports"],
        }
    
    def get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word."""
        word_lower = word.lower()
        
        if word_lower in self.custom_synonyms:
            return self.custom_synonyms[word_lower]
        
        if NLTK_AVAILABLE:
            synonyms = set()
            for syn in wordnet.synsets(word_lower):
                for lemma in syn.lemmas():
                    if lemma.name() != word_lower and '_' not in lemma.name():
                        synonyms.add(lemma.name())
            return list(synonyms)[:5]
        
        return []
    
    def replace(self, text: str, prob: float = 0.15) -> str:
        """Replace words with synonyms."""
        words = text.split()
        new_words = []
        
        for word in words:
            clean = re.sub(r'[^\w]', '', word.lower())
            
            if random.random() < prob and len(clean) > 3:
                synonyms = self.get_synonyms(clean)
                if synonyms:
                    new_word = random.choice(synonyms)
                    # Preserve case
                    if word[0].isupper():
                        new_word = new_word.capitalize()
                    # Preserve punctuation
                    trailing = re.search(r'[^\w]+$', word)
                    if trailing:
                        new_word += trailing.group()
                    new_words.append(new_word)
                    continue
            
            new_words.append(word)
        
        return ' '.join(new_words)


class SentenceAugmenter:
    """Sentence-level augmentation: shuffling, deletion, insertion."""
    
    def __init__(self):
        # Additional sentences to insert
        self.insertable_sentences = [
            "Representatives have not responded to requests for comment.",
            "The situation remains developing.",
            "More details are expected to emerge soon.",
            "Fans have been reacting on social media.",
            "This isn't the first time such rumors have surfaced.",
            "Friends say the situation is complicated.",
            "Industry insiders are watching closely.",
            "Neither party has confirmed or denied the reports.",
            "The timing of this news has raised eyebrows.",
            "Social media has been buzzing with speculation.",
            "This comes amid ongoing tensions.",
            "The news has sent shockwaves through Hollywood.",
            "Insiders say this has been brewing for months.",
            "Those close to the situation urge caution.",
            "The full story may be more complicated.",
        ]
    
    def shuffle_sentences(self, text: str, keep_first: bool = True, keep_last: bool = True) -> str:
        """Shuffle middle sentences while keeping intro and conclusion."""
        if NLTK_AVAILABLE:
            sentences = sent_tokenize(text)
        else:
            sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        
        if len(sentences) <= 3:
            return text
        
        first = [sentences[0]] if keep_first else []
        last = [sentences[-1]] if keep_last else []
        
        middle_start = 1 if keep_first else 0
        middle_end = -1 if keep_last else len(sentences)
        middle = sentences[middle_start:middle_end]
        
        random.shuffle(middle)
        
        return ' '.join(first + middle + last)
    
    def delete_sentence(self, text: str, n_delete: int = 1) -> str:
        """Delete random sentences from the middle."""
        if NLTK_AVAILABLE:
            sentences = sent_tokenize(text)
        else:
            sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        
        if len(sentences) <= 4:
            return text
        
        # Only delete from middle
        deletable = list(range(1, len(sentences) - 1))
        if len(deletable) <= n_delete:
            return text
        
        to_delete = set(random.sample(deletable, n_delete))
        result = [s for i, s in enumerate(sentences) if i not in to_delete]
        
        return ' '.join(result)
    
    def insert_sentence(self, text: str, n_insert: int = 1) -> str:
        """Insert additional sentences."""
        if NLTK_AVAILABLE:
            sentences = sent_tokenize(text)
        else:
            sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        
        for _ in range(n_insert):
            new_sentence = random.choice(self.insertable_sentences)
            # Insert somewhere in the middle
            pos = random.randint(1, len(sentences) - 1)
            sentences.insert(pos, new_sentence)
        
        return ' '.join(sentences)


# ============================================================================
# MASTER AUGMENTER
# ============================================================================

class MasterAugmenter:
    """Combines all augmentation methods."""
    
    def __init__(self, use_paraphrasing: bool = True, use_bert: bool = True):
        print("\n" + "="*60)
        print("INITIALIZING MASTER AUGMENTER")
        print("="*60)
        
        self.synonym_replacer = SynonymReplacer()
        self.sentence_augmenter = SentenceAugmenter()
        
        self.paraphraser = None
        if use_paraphrasing:
            self.paraphraser = ParaphraserT5()
        
        self.bert_replacer = None
        if use_bert:
            self.bert_replacer = BERTWordReplacer()
        
        print("="*60 + "\n")
    
    def augment(self, text: str, intensity: str = "medium") -> str:
        """
        Apply augmentation with specified intensity.
        
        intensity: "low", "medium", "high"
        """
        # Determine which augmentations to apply based on intensity
        if intensity == "low":
            methods = random.sample([
                "synonym",
                "detail_variation",
            ], 1)
        elif intensity == "medium":
            methods = random.sample([
                "synonym",
                "sentence_shuffle",
                "detail_variation",
                "sentence_modify",
            ], 2)
        else:  # high
            methods = random.sample([
                "synonym",
                "sentence_shuffle", 
                "detail_variation",
                "sentence_modify",
                "paraphrase",
                "bert_replace",
            ], 3)
        
        # Apply selected methods
        for method in methods:
            text = self._apply_method(text, method)
        
        return text
    
    def _apply_method(self, text: str, method: str) -> str:
        """Apply a single augmentation method."""
        
        if method == "synonym":
            return self.synonym_replacer.replace(text, prob=0.12)
        
        elif method == "sentence_shuffle":
            if random.random() < 0.5:
                return self.sentence_augmenter.shuffle_sentences(text)
            return text
        
        elif method == "sentence_modify":
            choice = random.random()
            if choice < 0.33:
                return self.sentence_augmenter.delete_sentence(text, 1)
            elif choice < 0.66:
                return self.sentence_augmenter.insert_sentence(text, 1)
            return text
        
        elif method == "detail_variation":
            return self._vary_details(text)
        
        elif method == "paraphrase":
            if self.paraphraser and self.paraphraser.loaded:
                # Paraphrase 1-2 random sentences
                if NLTK_AVAILABLE:
                    sentences = sent_tokenize(text)
                else:
                    sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
                
                if len(sentences) > 3:
                    idx = random.randint(1, len(sentences) - 2)
                    sentences[idx] = self.paraphraser.paraphrase(sentences[idx])
                    return ' '.join(sentences)
            return text
        
        elif method == "bert_replace":
            if self.bert_replacer and self.bert_replacer.loaded:
                return self.bert_replacer.replace_words(text, n_replacements=2)
            return text
        
        return text
    
    def _vary_details(self, text: str) -> str:
        """Add variation to details like locations and sources."""
        
        # Vary location mentions
        location_patterns = [
            r'\bin Los Angeles\b', r'\bin New York\b', r'\bin Miami\b',
            r'\bin Hollywood\b', r'\bin Beverly Hills\b', r'\bin Malibu\b',
        ]
        for pattern in location_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                new_location = f"in {random.choice(LOCATIONS)}"
                text = re.sub(pattern, new_location, text, count=1, flags=re.IGNORECASE)
                break
        
        # Vary source types
        source_patterns = [
            r'a source close to the couple',
            r'an insider',
            r'a close friend',
            r'sources',
        ]
        for pattern in source_patterns:
            if pattern.lower() in text.lower():
                new_source = random.choice(SOURCE_TYPES)
                text = re.sub(pattern, new_source, text, count=1, flags=re.IGNORECASE)
                break
        
        # Vary time expressions
        time_patterns = [
            r'last week', r'recently', r'this week', r'earlier this month',
        ]
        for pattern in time_patterns:
            if pattern.lower() in text.lower():
                new_time = random.choice(TIME_EXPRESSIONS)
                text = re.sub(pattern, new_time, text, count=1, flags=re.IGNORECASE)
                break
        
        return text


# ============================================================================
# HELPER FUNCTIONS
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
    date = start + timedelta(days=random.randint(0, delta.days))
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
    
    # Full name
    text = re.sub(re.escape(original), replacement, text, flags=re.IGNORECASE)
    
    # First name
    if orig_parts and repl_parts:
        text = re.sub(r'\b' + re.escape(orig_parts[0]) + r'\b', 
                     repl_parts[0], text, flags=re.IGNORECASE)
    
    # Last name
    if len(orig_parts) > 1 and len(repl_parts) > 1:
        text = re.sub(r'\b' + re.escape(orig_parts[-1]) + r'\b',
                     repl_parts[-1], text, flags=re.IGNORECASE)
    
    return text

def vary_title_prefix(title: str) -> str:
    for prefix in ["EXCLUSIVE:", "BREAKING:", "REPORT:", "SHOCKING:", 
                   "REVEALED:", "INSIDER:", "BOMBSHELL:", "SOURCES:", "JUST IN:"]:
        if title.upper().startswith(prefix):
            title = title[len(prefix):].strip()
            break
    return random.choice(TITLE_PREFIXES) + title


# ============================================================================
# TWEET GENERATION
# ============================================================================

def generate_tweet_count() -> int:
    count = int(np.random.lognormal(2.5, 1.5))
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
    if random.random() < 0.4:
        name += str(random.randint(1, 999))
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
        tweet_date = base_date + timedelta(hours=random.randint(0, 168))
        tweets.append({
            "tweet_id": generate_tweet_id(),
            "user_id": generate_user_id(),
            "user_name": generate_username(),
            "text": generate_tweet_text(celeb_name, news_url),
            "created_at": tweet_date.strftime("%Y-%m-%d %H:%M:%S"),
        })
    
    return {"tweets": tweets}


# ============================================================================
# ENGAGEMENT GENERATION
# ============================================================================

def generate_retweets(tweets_data: Dict) -> Dict:
    retweets = {}
    for tweet in tweets_data.get("tweets", []):
        num = np.random.poisson(4)
        if num > 0:
            retweets[tweet["tweet_id"]] = [
                {
                    "user_id": generate_user_id(),
                    "user_name": generate_username(),
                    "created_at": tweet["created_at"],
                    "followers_count": int(np.random.lognormal(5.5, 2.0)),
                    "friends_count": int(np.random.lognormal(5.0, 1.5)),
                    "verified": random.random() < 0.012,
                }
                for _ in range(num)
            ]
    return retweets

def generate_likes(tweets_data: Dict) -> Dict:
    likes = {}
    for tweet in tweets_data.get("tweets", []):
        num = np.random.poisson(8)
        if num > 0:
            likes[tweet["tweet_id"]] = [generate_user_id() for _ in range(num)]
    return likes

def generate_replies(tweets_data: Dict, celeb_name: str) -> Dict:
    replies = {}
    templates = [
        "This is crazy!", "I don't believe this", "Wow just wow",
        "Fake news", "Leave them alone!", "I knew it!",
        f"Poor {celeb_name.split()[0]}", "The media is trash",
        "👀👀👀", "😭😭😭", "🙄", "Not surprised",
    ]
    
    for tweet in tweets_data.get("tweets", []):
        num = np.random.poisson(1)
        if num > 0:
            replies[tweet["tweet_id"]] = [
                {"user_id": generate_user_id(), "text": random.choice(templates), "created_at": tweet["created_at"]}
                for _ in range(num)
            ]
    return replies

def generate_user_profiles(tweets_data: Dict, retweets_data: Dict) -> List[Dict]:
    users = {}
    
    for tweet in tweets_data.get("tweets", []):
        uid = tweet["user_id"]
        if uid not in users:
            users[uid] = {"user_id": uid, "user_name": tweet["user_name"], "is_tweeter": True, "is_retweeter": False}
    
    for rt_list in retweets_data.values():
        for rt in rt_list:
            uid = rt["user_id"]
            if uid not in users:
                users[uid] = {
                    "user_id": uid, "user_name": rt["user_name"],
                    "is_tweeter": False, "is_retweeter": True,
                    "followers_count": rt.get("followers_count"),
                    "friends_count": rt.get("friends_count"),
                    "verified": rt.get("verified"),
                }
            else:
                users[uid]["is_retweeter"] = True
    
    profiles = []
    for uid, data in users.items():
        is_tweeter = data.get("is_tweeter", False)
        
        if is_tweeter:
            followers = int(np.random.lognormal(6.5, 2.0))
            friends = int(np.random.lognormal(6.2, 1.5))
            statuses = int(np.random.lognormal(9.0, 1.5))
            favourites = int(np.random.lognormal(7.2, 1.8))
            verified = random.random() < 0.053
        else:
            followers = data.get("followers_count") or int(np.random.lognormal(5.5, 2.0))
            friends = data.get("friends_count") or int(np.random.lognormal(5.0, 1.5))
            statuses = int(np.random.lognormal(7.5, 1.5))
            favourites = int(np.random.lognormal(6.0, 1.8))
            verified = data.get("verified", random.random() < 0.012)
        
        profiles.append({
            "user_id": uid, "user_name": data["user_name"],
            "followers_count": followers, "friends_count": friends,
            "statuses_count": statuses, "favourites_count": favourites,
            "verified": verified, "geo_enabled": random.random() < 0.42,
            "protected": random.random() < 0.05,
            "is_tweeter": data.get("is_tweeter", False),
            "is_retweeter": data.get("is_retweeter", False),
        })
    
    return profiles


# ============================================================================
# NEWS GENERATION
# ============================================================================

def generate_news_article(seed: Dict, replacement_celeb: str, augmenter: MasterAugmenter) -> Tuple[Dict, str]:
    """Generate augmented news article."""
    title = seed["title"]
    text = seed["text"]
    
    # Substitute celebrity
    original_celeb, _ = extract_celebrity_name(title + " " + text)
    
    if replacement_celeb and original_celeb:
        title = substitute_celebrity(title, original_celeb, replacement_celeb)
        text = substitute_celebrity(text, original_celeb, replacement_celeb)
        celeb_name = replacement_celeb
    else:
        celeb_name = original_celeb or "the celebrity"
    
    # Apply augmentation
    intensity = random.choice(["low", "medium", "medium", "high"])
    text = augmenter.augment(text, intensity=intensity)
    
    # Vary title
    title = vary_title_prefix(title)
    if random.random() < 0.3:
        title = augmenter.synonym_replacer.replace(title, prob=0.1)
    
    publish_date = random_date()
    
    return {
        "title": title,
        "text": text,
        "url": random.choice(FAKE_URLS) + generate_id(),
        "authors": [random.choice(FAKE_AUTHORS)],
        "publish_date": publish_date,
        "source": random.choice(FAKE_URLS).replace("https://www.", "").replace("/", ""),
    }, celeb_name


def generate_single_news(seed: Dict, output_dir: str, news_id: str,
                         replacement_celeb: str, augmenter: MasterAugmenter) -> bool:
    """Generate all files for one news item."""
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
                writer = csv.DictWriter(f, fieldnames=[
                    "user_id", "user_name", "followers_count", "friends_count",
                    "statuses_count", "favourites_count", "verified",
                    "geo_enabled", "protected", "is_tweeter", "is_retweeter"
                ], delimiter='\t')
                writer.writeheader()
                writer.writerows(users)
        
        return True
    except Exception as e:
        print(f"Error: {e}")
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
         use_paraphrasing: bool = True, use_bert: bool = True):
    """Main generation pipeline."""
    
    print("\n" + "="*60)
    print("ULTIMATE SYNTHETIC FAKE NEWS GENERATOR v3")
    print("="*60)
    
    print(f"\n📂 Loading seeds from {seeds_dir}...")
    seeds = load_seeds(seeds_dir)
    print(f"   Loaded {len(seeds)} seeds")
    
    print(f"\n🔧 Initializing augmenter...")
    augmenter = MasterAugmenter(use_paraphrasing=use_paraphrasing, use_bert=use_bert)
    
    print(f"\n🚀 Generating {target_count} synthetic news...")
    os.makedirs(output_dir, exist_ok=True)
    
    generated = 0
    failed = 0
    
    while generated < target_count:
        for seed in seeds:
            if generated >= target_count:
                break
            
            _, gender = extract_celebrity_name(seed["title"] + " " + seed["text"])
            
            if gender == "male":
                pool = MALE_CELEBRITIES
            elif gender == "female":
                pool = FEMALE_CELEBRITIES
            else:
                pool = CELEBRITIES
            
            replacement = random.choice(pool)
            news_id = f"synthetic_fake_{generated:05d}"
            
            if generate_single_news(seed, output_dir, news_id, replacement, augmenter):
                generated += 1
                if generated % 500 == 0:
                    print(f"   ✓ Generated {generated}/{target_count}")
            else:
                failed += 1
    
    print(f"\n" + "="*60)
    print(f"✅ COMPLETE!")
    print(f"   Generated: {generated}")
    print(f"   Failed: {failed}")
    print(f"   Output: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="./seeds")
    parser.add_argument("--output", default="./synthetic_fake_v3")
    parser.add_argument("--count", type=int, default=TARGET_SYNTHETIC_COUNT)
    parser.add_argument("--no-paraphrase", action="store_true", help="Disable T5 paraphrasing")
    parser.add_argument("--no-bert", action="store_true", help="Disable BERT word replacement")
    
    args = parser.parse_args()
    main(
        args.seeds, 
        args.output, 
        args.count,
        use_paraphrasing=not args.no_paraphrase,
        use_bert=not args.no_bert
    )
