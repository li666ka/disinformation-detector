"""
Synthetic Fake News Generator for FakeNewsNet Dataset Augmentation
===================================================================
Generates synthetic FAKE news with realistic social media engagement patterns.

Components:
1. News articles (title, text) - from seeds with celebrity name substitution
2. Tweets - LogNormal distribution for count
3. Retweets, Likes, Replies - Poisson distributions
4. User profiles - LogNormal for followers, Bernoulli for verified
"""

import json
import random
import os
import csv
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import re
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Target: ~11,684 synthetic FAKE news to balance the dataset
TARGET_SYNTHETIC_COUNT = 12000

# Celebrity names for substitution (will rotate through these)
CELEBRITIES = [
    # A-list actors
    "Jennifer Aniston", "Brad Pitt", "Angelina Jolie", "George Clooney",
    "Leonardo DiCaprio", "Tom Hanks", "Julia Roberts", "Meryl Streep",
    "Denzel Washington", "Sandra Bullock", "Tom Cruise", "Nicole Kidman",
    "Johnny Depp", "Scarlett Johansson", "Robert Downey Jr.", "Chris Hemsworth",
    "Chris Evans", "Chris Pratt", "Ryan Reynolds", "Ryan Gosling",
    "Emma Stone", "Emma Watson", "Natalie Portman", "Anne Hathaway",
    "Jennifer Lawrence", "Margot Robbie", "Gal Gadot", "Zendaya",
    "Timothée Chalamet", "Florence Pugh", "Austin Butler", "Sydney Sweeney",
    
    # Musicians
    "Taylor Swift", "Beyoncé", "Rihanna", "Lady Gaga", "Ariana Grande",
    "Selena Gomez", "Dua Lipa", "Billie Eilish", "Olivia Rodrigo", "Miley Cyrus",
    "Justin Bieber", "Drake", "Kanye West", "Ed Sheeran", "Harry Styles",
    "The Weeknd", "Post Malone", "Bad Bunny", "Bruno Mars", "Shawn Mendes",
    "Adele", "Katy Perry", "Shakira", "Cardi B", "Nicki Minaj", "Lizzo",
    "Doja Cat", "SZA", "Megan Thee Stallion", "Ice Spice",
    
    # Reality TV / Influencers
    "Kim Kardashian", "Kylie Jenner", "Kendall Jenner", "Khloé Kardashian",
    "Kourtney Kardashian", "Kris Jenner", "Paris Hilton", "Nicole Richie",
    
    # Athletes
    "Tom Brady", "LeBron James", "Serena Williams", "Cristiano Ronaldo",
    "David Beckham", "Tiger Woods", "Patrick Mahomes", "Travis Kelce",
    
    # Other celebrities
    "Oprah Winfrey", "Ellen DeGeneres", "Jimmy Fallon", "Trevor Noah",
    "Dwayne Johnson", "Kevin Hart", "Will Smith", "Jada Pinkett Smith",
    "Blake Lively", "Megan Fox", "Machine Gun Kelly", "Pete Davidson",
    "Ben Affleck", "Jennifer Lopez", "Matt Damon", "Mark Wahlberg",
    "Reese Witherspoon", "Charlize Theron", "Halle Berry", "Viola Davis",
    "Cate Blanchett", "Kate Winslet", "Amy Adams", "Jessica Chastain",
    "Hugh Jackman", "Jake Gyllenhaal", "Joaquin Phoenix", "Christian Bale",
    "Michael B. Jordan", "John Boyega", "Daniel Kaluuya", "Lupita Nyong'o",
]

# Male/Female split for gender-appropriate substitutions
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
    "Machine Gun Kelly", "Trevor Noah", "Jimmy Fallon",
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
]

# Celebrity couples for relationship stories
CELEBRITY_COUPLES = [
    ("Taylor Swift", "Travis Kelce"),
    ("Blake Lively", "Ryan Reynolds"),
    ("Beyoncé", "Jay-Z"),
    ("Ben Affleck", "Jennifer Lopez"),
    ("Megan Fox", "Machine Gun Kelly"),
    ("Zendaya", "Tom Holland"),
    ("Justin Bieber", "Hailey Bieber"),
    ("George Clooney", "Amal Clooney"),
    ("David Beckham", "Victoria Beckham"),
    ("Ryan Gosling", "Eva Mendes"),
    ("Ashton Kutcher", "Mila Kunis"),
    ("Kristen Bell", "Dax Shepard"),
    ("John Legend", "Chrissy Teigen"),
    ("Chris Hemsworth", "Elsa Pataky"),
    ("Matt Damon", "Luciana Barroso"),
    ("Hugh Jackman", "Deborra-Lee Furness"),
    ("Tom Hanks", "Rita Wilson"),
    ("Will Smith", "Jada Pinkett Smith"),
    ("Dwayne Johnson", "Lauren Hashian"),
    ("Adam Levine", "Behati Prinsloo"),
]

# Title prefixes for variety
TITLE_PREFIXES = [
    "EXCLUSIVE: ", "BREAKING: ", "", "", "",  # Empty string = no prefix (more common)
    "REPORT: ", "SHOCKING: ", "REVEALED: ", "", "",
    "INSIDER: ", "", "", "BOMBSHELL: ", "",
]

# Gossip site URLs for fake sources
FAKE_URLS = [
    "https://www.celebuzz.com/",
    "https://www.hollywoodlife.com/",
    "https://www.usmagazine.com/",
    "https://www.eonline.com/",
    "https://www.tmz.com/",
    "https://www.pagesix.com/",
    "https://www.dailymail.co.uk/",
    "https://www.thesun.co.uk/",
    "https://www.mirror.co.uk/",
    "https://www.nationalenquirer.com/",
    "https://www.radaronline.com/",
    "https://www.okmagazine.com/",
    "https://www.intouchweekly.com/",
    "https://www.lifeandstylemag.com/",
    "https://www.closerweekly.com/",
]

# Fake author names
FAKE_AUTHORS = [
    "Staff Reporter", "Entertainment Desk", "Celebrity News Team",
    "Jessica Martinez", "Michael Thompson", "Sarah Johnson",
    "David Williams", "Emily Davis", "Chris Anderson",
    "Amanda Miller", "Ryan Garcia", "Nicole Brown",
    "Jason Lee", "Michelle Taylor", "Brandon Wilson",
]

# Tweet templates for different reactions
TWEET_TEMPLATES = {
    "shock": [
        "OMG I can't believe this about {celeb}! 😱",
        "Wait WHAT?! {celeb}?! No way!",
        "This {celeb} news is INSANE",
        "I'm shook about {celeb} rn",
        "NOT {celeb}!!! 😭😭😭",
        "Is this real about {celeb}???",
        "BREAKING: {celeb} news has me shook",
        "I refuse to believe this about {celeb}",
    ],
    "support": [
        "Sending love to {celeb} ❤️",
        "We stan {celeb} no matter what",
        "{celeb} deserves better tbh",
        "Leave {celeb} alone already!",
        "Team {celeb} forever 💪",
        "Praying for {celeb} 🙏",
        "{celeb} is still iconic period",
    ],
    "gossip": [
        "I KNEW something was up with {celeb}",
        "My friend works in Hollywood and they said {celeb}...",
        "Tea about {celeb} has been brewing for months",
        "The {celeb} situation is messy",
        "Everyone saw this coming with {celeb}",
        "Spill the tea on {celeb} ☕",
        "{celeb} really thought we wouldn't find out",
    ],
    "skeptical": [
        "I don't believe this {celeb} story at all",
        "This sounds fake about {celeb}",
        "Another day another fake {celeb} rumor",
        "Source: trust me bro about {celeb}",
        "The media lies about {celeb} constantly",
        "Where's the proof about {celeb}?",
    ],
    "link_share": [
        "Just read this about {celeb} {url}",
        "{celeb} news 👀 {url}",
        "Y'all need to see this {celeb} story {url}",
        "OMG {celeb}!!! {url}",
        "Can't believe this about {celeb} {url}",
    ],
    "generic": [
        "{celeb} is trending and idk how to feel",
        "So {celeb} is in the news again...",
        "What's going on with {celeb}?",
        "Anyone else following the {celeb} situation?",
        "Not me refreshing for {celeb} updates",
    ],
}

# Hashtags
HASHTAGS = [
    "#celebrity", "#gossip", "#breaking", "#news", "#hollywood",
    "#drama", "#tea", "#entertainment", "#exclusive", "#viral",
    "#trending", "#celebnews", "#hotgossip", "#breakingnews",
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_id() -> str:
    """Generate a unique ID for news items."""
    return f"synthetic_{uuid.uuid4().hex[:12]}"


def generate_tweet_id() -> str:
    """Generate a fake tweet ID (19-digit number like real Twitter)."""
    return str(random.randint(10**18, 10**19 - 1))


def generate_user_id() -> str:
    """Generate a fake user ID."""
    return str(random.randint(10**8, 10**10 - 1))


def random_date(start_year: int = 2016, end_year: int = 2019) -> str:
    """Generate a random date in the dataset's time range."""
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    date = start + timedelta(days=random_days)
    return date.strftime("%Y-%m-%d %H:%M:%S")


def extract_celebrity_name(text: str) -> Tuple[str, str]:
    """Extract celebrity name from text and determine gender."""
    for celeb in MALE_CELEBRITIES:
        if celeb.lower() in text.lower():
            return celeb, "male"
    for celeb in FEMALE_CELEBRITIES:
        if celeb.lower() in text.lower():
            return celeb, "female"
    return "", "unknown"


def substitute_celebrity(text: str, original: str, replacement: str) -> str:
    """Replace celebrity name in text, handling first/last name mentions."""
    if not original or not replacement:
        return text
    
    # Split names
    orig_parts = original.split()
    repl_parts = replacement.split()
    
    # Replace full name
    text = re.sub(re.escape(original), replacement, text, flags=re.IGNORECASE)
    
    # Replace first name only (if appears alone)
    if len(orig_parts) > 0 and len(repl_parts) > 0:
        # Only replace standalone first names (not part of other words)
        text = re.sub(
            r'\b' + re.escape(orig_parts[0]) + r'\b',
            repl_parts[0],
            text,
            flags=re.IGNORECASE
        )
    
    # Replace last name only (if appears alone)
    if len(orig_parts) > 1 and len(repl_parts) > 1:
        text = re.sub(
            r'\b' + re.escape(orig_parts[-1]) + r'\b',
            repl_parts[-1],
            text,
            flags=re.IGNORECASE
        )
    
    return text


def vary_title_prefix(title: str) -> str:
    """Add variety to title prefixes."""
    # Remove existing prefix if any
    for prefix in ["EXCLUSIVE:", "BREAKING:", "REPORT:", "SHOCKING:", 
                   "REVEALED:", "INSIDER:", "BOMBSHELL:"]:
        if title.startswith(prefix):
            title = title[len(prefix):].strip()
            break
    
    # Add random prefix
    new_prefix = random.choice(TITLE_PREFIXES)
    return new_prefix + title


def vary_text_style(text: str) -> str:
    """Add small variations to text for diversity."""
    variations = [
        # Vary uncertainty words
        ("allegedly", random.choice(["reportedly", "allegedly", "supposedly", "apparently"])),
        ("reportedly", random.choice(["allegedly", "reportedly", "according to sources"])),
        ("Sources exclusively claim", random.choice([
            "Sources exclusively claim",
            "An insider exclusively revealed",
            "Sources close to the situation claim",
            "According to insiders",
            "Multiple sources confirm",
        ])),
        ("an insider revealed", random.choice([
            "an insider revealed",
            "a source told us",
            "a close friend shared",
            "someone familiar with the matter said",
        ])),
    ]
    
    for old, new in variations:
        if old in text and random.random() > 0.5:
            text = text.replace(old, new, 1)
    
    return text


# ============================================================================
# TWEET GENERATION
# ============================================================================

def generate_tweet_count() -> int:
    """Generate number of tweets using LogNormal distribution.
    Based on GossipCop FAKE stats: mean=128.9, median=13
    """
    # LogNormal parameters to match observed distribution
    mu = 2.5  # log(median) ≈ log(13)
    sigma = 1.5  # spread
    
    count = int(np.random.lognormal(mu, sigma))
    return max(1, min(count, 2000))  # Cap at reasonable max


def generate_tweet_text(celeb_name: str, news_url: str) -> str:
    """Generate a realistic tweet about the celebrity."""
    category = random.choice(list(TWEET_TEMPLATES.keys()))
    template = random.choice(TWEET_TEMPLATES[category])
    
    tweet = template.format(celeb=celeb_name, url=news_url)
    
    # Maybe add hashtag (21% chance based on data)
    if random.random() < 0.21:
        hashtag = random.choice(HASHTAGS)
        tweet += f" {hashtag}"
    
    # Truncate if too long
    if len(tweet) > 280:
        tweet = tweet[:277] + "..."
    
    return tweet


def generate_username() -> str:
    """Generate a realistic-ish Twitter username."""
    prefixes = ["the", "real", "official", "its", "im", "just", "only", ""]
    names = ["fan", "stan", "lover", "news", "updates", "daily", "gossip", 
             "tea", "vibes", "life", "world", "zone", "hub"]
    suffixes = ["", "x", "xx", "xo", "2024", "2023", "official", "_", "__"]
    
    # Random word-based username
    if random.random() < 0.5:
        words = ["happy", "sunny", "cool", "wild", "crazy", "sweet", "dark",
                 "blue", "pink", "star", "moon", "sky", "love", "dream"]
        name = random.choice(words) + random.choice(names)
    else:
        name = random.choice(prefixes) + random.choice(["celeb", "hollywood", 
                                                         "gossip", "tea", "news"])
    
    name += random.choice(suffixes)
    name += str(random.randint(1, 999)) if random.random() < 0.4 else ""
    
    return name[:15]  # Twitter username limit


def generate_tweets(celeb_name: str, news_url: str, publish_date: str) -> Dict:
    """Generate tweets.json content."""
    num_tweets = generate_tweet_count()
    tweets = []
    
    base_date = datetime.strptime(publish_date, "%Y-%m-%d %H:%M:%S")
    
    for i in range(num_tweets):
        # Tweets spread over 1-7 days after publication
        hours_offset = random.randint(0, 168)  # 0-7 days in hours
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
# ENGAGEMENT GENERATION (retweets, likes, replies)
# ============================================================================

def generate_retweets(tweets_data: Dict) -> Dict:
    """Generate retweets.json content."""
    retweets = {}
    
    for tweet in tweets_data.get("tweets", []):
        tweet_id = tweet["tweet_id"]
        
        # Poisson distribution for retweet count, λ=4
        num_retweets = np.random.poisson(4)
        
        if num_retweets > 0:
            retweet_list = []
            for _ in range(num_retweets):
                # Simplified retweet object
                rt = {
                    "user_id": generate_user_id(),
                    "user_name": generate_username(),
                    "created_at": tweet["created_at"],  # Simplified
                    "followers_count": int(np.random.lognormal(5.5, 2.0)),
                    "friends_count": int(np.random.lognormal(5.0, 1.5)),
                    "verified": random.random() < 0.012,  # 1.2% for retweeters
                }
                retweet_list.append(rt)
            retweets[tweet_id] = retweet_list
    
    return retweets


def generate_likes(tweets_data: Dict) -> Dict:
    """Generate likes.json content (just user_ids per tweet)."""
    likes = {}
    
    for tweet in tweets_data.get("tweets", []):
        tweet_id = tweet["tweet_id"]
        
        # Poisson distribution for like count, λ=8
        num_likes = np.random.poisson(8)
        
        if num_likes > 0:
            likes[tweet_id] = [generate_user_id() for _ in range(num_likes)]
    
    return likes


def generate_replies(tweets_data: Dict, celeb_name: str) -> Dict:
    """Generate replies.json content."""
    replies = {}
    
    reply_templates = [
        "This is crazy!",
        "I don't believe this",
        "Wow just wow",
        "Fake news",
        "This is so sad",
        "Leave them alone!",
        "I knew it!",
        "Not surprised tbh",
        f"Poor {celeb_name.split()[0]}",
        "The media is trash",
        "Who cares honestly",
        "This is nobody's business",
        "👀👀👀",
        "😭😭😭",
        "🙄🙄🙄",
    ]
    
    for tweet in tweets_data.get("tweets", []):
        tweet_id = tweet["tweet_id"]
        
        # Poisson distribution for reply count, λ=1
        num_replies = np.random.poisson(1)
        
        if num_replies > 0:
            reply_list = []
            for _ in range(num_replies):
                reply = {
                    "user_id": generate_user_id(),
                    "text": random.choice(reply_templates),
                    "created_at": tweet["created_at"],
                }
                reply_list.append(reply)
            replies[tweet_id] = reply_list
    
    return replies


# ============================================================================
# USER PROFILE GENERATION
# ============================================================================

def generate_user_profiles(tweets_data: Dict, retweets_data: Dict) -> List[Dict]:
    """Generate user profiles for new_user.tsv."""
    users = {}
    
    # Collect all users from tweets
    for tweet in tweets_data.get("tweets", []):
        user_id = tweet["user_id"]
        if user_id not in users:
            users[user_id] = {
                "user_id": user_id,
                "user_name": tweet["user_name"],
                "is_tweeter": True,
                "is_retweeter": False,
            }
    
    # Collect users from retweets
    for tweet_id, rt_list in retweets_data.items():
        for rt in rt_list:
            user_id = rt["user_id"]
            if user_id not in users:
                users[user_id] = {
                    "user_id": user_id,
                    "user_name": rt["user_name"],
                    "is_tweeter": False,
                    "is_retweeter": True,
                    "followers_count": rt.get("followers_count"),
                    "friends_count": rt.get("friends_count"),
                    "verified": rt.get("verified"),
                }
            else:
                users[user_id]["is_retweeter"] = True
    
    # Generate full profiles
    profiles = []
    for user_id, user_data in users.items():
        is_tweeter = user_data.get("is_tweeter", False)
        
        # Use different distributions for tweeters vs retweeters
        if is_tweeter:
            # Tweeters have more followers (mean=60,758, median=691)
            followers = int(np.random.lognormal(6.5, 2.0))
            friends = int(np.random.lognormal(6.2, 1.5))
            statuses = int(np.random.lognormal(9.0, 1.5))
            favourites = int(np.random.lognormal(7.2, 1.8))
            verified = random.random() < 0.053  # 5.3%
        else:
            # Retweeters have fewer followers (mean=5,124, median=298)
            followers = user_data.get("followers_count") or int(np.random.lognormal(5.5, 2.0))
            friends = user_data.get("friends_count") or int(np.random.lognormal(5.0, 1.5))
            statuses = int(np.random.lognormal(7.5, 1.5))
            favourites = int(np.random.lognormal(6.0, 1.8))
            verified = user_data.get("verified", random.random() < 0.012)
        
        profile = {
            "user_id": user_id,
            "user_name": user_data["user_name"],
            "followers_count": followers,
            "friends_count": friends,
            "statuses_count": statuses,
            "favourites_count": favourites,
            "verified": verified,
            "geo_enabled": random.random() < 0.42,  # 42.3%
            "protected": random.random() < 0.05,
            "is_tweeter": user_data.get("is_tweeter", False),
            "is_retweeter": user_data.get("is_retweeter", False),
        }
        profiles.append(profile)
    
    return profiles


# ============================================================================
# NEWS ARTICLE GENERATION
# ============================================================================

def generate_news_article(seed: Dict, replacement_celeb: str = None) -> Dict:
    """Generate news_article.json from a seed."""
    title = seed["title"]
    text = seed["text"]
    
    # Extract original celebrity and substitute if needed
    original_celeb, gender = extract_celebrity_name(title + " " + text)
    
    if replacement_celeb and original_celeb:
        title = substitute_celebrity(title, original_celeb, replacement_celeb)
        text = substitute_celebrity(text, original_celeb, replacement_celeb)
        celeb_name = replacement_celeb
    else:
        celeb_name = original_celeb or "the celebrity"
    
    # Add variety
    title = vary_title_prefix(title)
    text = vary_text_style(text)
    
    # Generate metadata
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


# ============================================================================
# MAIN GENERATION PIPELINE
# ============================================================================

def load_seeds(seeds_dir: str) -> List[Dict]:
    """Load all seed files."""
    all_seeds = []
    
    for filename in os.listdir(seeds_dir):
        if filename.endswith("_seeds.json"):
            filepath = os.path.join(seeds_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                category = data.get("category", "UNKNOWN")
                for seed in data.get("seeds", []):
                    seed["category"] = category
                    all_seeds.append(seed)
    
    return all_seeds


def generate_single_news(seed: Dict, output_dir: str, news_id: str, 
                         replacement_celeb: str = None) -> bool:
    """Generate all files for a single synthetic news item."""
    try:
        # Create news directory
        news_dir = os.path.join(output_dir, news_id)
        os.makedirs(news_dir, exist_ok=True)
        
        # Generate article
        article, celeb_name = generate_news_article(seed, replacement_celeb)
        
        # Generate tweets
        tweets = generate_tweets(celeb_name, article["url"], article["publish_date"])
        
        # Generate engagement
        retweets = generate_retweets(tweets)
        likes = generate_likes(tweets)
        replies = generate_replies(tweets, celeb_name)
        
        # Generate user profiles
        users = generate_user_profiles(tweets, retweets)
        
        # Save all files
        # 1. news_article.json
        with open(os.path.join(news_dir, "news_article.json"), 'w') as f:
            json.dump(article, f, indent=2)
        
        # 2. tweets.json
        with open(os.path.join(news_dir, "tweets.json"), 'w') as f:
            json.dump(tweets, f, indent=2)
        
        # 3. retweets.json
        with open(os.path.join(news_dir, "retweets.json"), 'w') as f:
            json.dump(retweets, f, indent=2)
        
        # 4. likes.json
        with open(os.path.join(news_dir, "likes.json"), 'w') as f:
            json.dump(likes, f, indent=2)
        
        # 5. replies.json
        with open(os.path.join(news_dir, "replies.json"), 'w') as f:
            json.dump(replies, f, indent=2)
        
        # 6. new_user.tsv
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


def main(seeds_dir: str, output_dir: str, target_count: int = TARGET_SYNTHETIC_COUNT):
    """Main generation pipeline."""
    print(f"Loading seeds from {seeds_dir}...")
    seeds = load_seeds(seeds_dir)
    print(f"Loaded {len(seeds)} seeds")
    
    # Calculate how many variations per seed
    variations_per_seed = max(1, target_count // len(seeds)) + 1
    print(f"Generating ~{variations_per_seed} variations per seed")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    generated = 0
    failed = 0
    
    # Generate variations - cycle through seeds
    while generated < target_count:
        for i, seed in enumerate(seeds):
            if generated >= target_count:
                break
            
            # Determine gender for this seed
            _, gender = extract_celebrity_name(seed["title"] + " " + seed["text"])
            
            # Select appropriate celebrity pool
            if gender == "male":
                celeb_pool = MALE_CELEBRITIES
            elif gender == "female":
                celeb_pool = FEMALE_CELEBRITIES
            else:
                celeb_pool = CELEBRITIES
            
            # Pick a random celebrity for substitution
            replacement_celeb = random.choice(celeb_pool)
            
            news_id = f"synthetic_fake_{generated:05d}"
            
            success = generate_single_news(
                seed=seed,
                output_dir=output_dir,
                news_id=news_id,
                replacement_celeb=replacement_celeb
            )
            
            if success:
                generated += 1
                if generated % 500 == 0:
                    print(f"Generated {generated}/{target_count} news items...")
            else:
                failed += 1
    
    print(f"\n✅ Generation complete!")
    print(f"   Generated: {generated}")
    print(f"   Failed: {failed}")
    print(f"   Output: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic fake news")
    parser.add_argument("--seeds", default="/home/claude/seeds", 
                       help="Directory containing seed JSON files")
    parser.add_argument("--output", default="/home/claude/synthetic_fake",
                       help="Output directory for generated news")
    parser.add_argument("--count", type=int, default=TARGET_SYNTHETIC_COUNT,
                       help="Target number of synthetic news to generate")
    
    args = parser.parse_args()
    main(args.seeds, args.output, args.count)
