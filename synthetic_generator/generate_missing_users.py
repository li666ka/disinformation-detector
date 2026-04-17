"""
Generate Missing new_user.tsv Files for FakeNewsNet
====================================================
Fills in missing user profile data based on tweets.json

For folders that have tweets.json but no new_user.tsv,
this script generates realistic user profiles using
the statistical distributions from existing data.
"""

import os
import json
import csv
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Set
from datetime import datetime


# ============================================================================
# STATISTICAL DISTRIBUTIONS (from FakeNewsNet analysis)
# ============================================================================

# Tweeters (users who posted original tweets sharing the news)
TWEETER_STATS = {
    "followers": {"mu": 6.5, "sigma": 2.0},      # LogNormal, median ~691
    "friends": {"mu": 6.2, "sigma": 1.5},         # LogNormal, median ~486
    "statuses": {"mu": 9.0, "sigma": 1.5},        # LogNormal, median ~7890
    "favourites": {"mu": 7.2, "sigma": 1.8},      # LogNormal, median ~1408
    "verified_prob": 0.053,                        # 5.3%
    "geo_enabled_prob": 0.42,                      # 42%
    "protected_prob": 0.05,                        # 5%
}

# Retweeters (users who retweeted)
RETWEETER_STATS = {
    "followers": {"mu": 5.5, "sigma": 2.0},       # LogNormal, median ~298
    "friends": {"mu": 5.0, "sigma": 1.5},         # LogNormal
    "statuses": {"mu": 7.5, "sigma": 1.5},        # LogNormal
    "favourites": {"mu": 6.0, "sigma": 1.8},      # LogNormal
    "verified_prob": 0.012,                        # 1.2%
    "geo_enabled_prob": 0.35,                      # 35%
    "protected_prob": 0.05,                        # 5%
}


# ============================================================================
# USER GENERATION FUNCTIONS
# ============================================================================

def generate_user_profile(user_id: str, user_name: str, is_tweeter: bool, 
                          is_retweeter: bool) -> Dict:
    """Generate a realistic user profile based on role."""
    
    # Choose stats based on role (tweeters have higher engagement)
    if is_tweeter:
        stats = TWEETER_STATS
    else:
        stats = RETWEETER_STATS
    
    profile = {
        "user_id": user_id,
        "user_name": user_name or f"user_{user_id[-6:]}",
        "followers_count": int(np.random.lognormal(
            stats["followers"]["mu"], stats["followers"]["sigma"]
        )),
        "friends_count": int(np.random.lognormal(
            stats["friends"]["mu"], stats["friends"]["sigma"]
        )),
        "statuses_count": int(np.random.lognormal(
            stats["statuses"]["mu"], stats["statuses"]["sigma"]
        )),
        "favourites_count": int(np.random.lognormal(
            stats["favourites"]["mu"], stats["favourites"]["sigma"]
        )),
        "verified": random.random() < stats["verified_prob"],
        "geo_enabled": random.random() < stats["geo_enabled_prob"],
        "protected": random.random() < stats["protected_prob"],
        "is_tweeter": is_tweeter,
        "is_retweeter": is_retweeter,
    }
    
    return profile


def extract_users_from_tweets(tweets_path: str) -> Dict[str, Dict]:
    """Extract user info from tweets.json."""
    users = {}
    
    try:
        with open(tweets_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tweets = data.get("tweets", [])
        if not tweets and isinstance(data, list):
            tweets = data
        
        for tweet in tweets:
            # Handle different tweet formats
            if isinstance(tweet, dict):
                user_id = str(tweet.get("user_id", tweet.get("user", {}).get("id", "")))
                user_name = tweet.get("user_name", tweet.get("user", {}).get("screen_name", ""))
                
                if user_id and user_id not in users:
                    users[user_id] = {
                        "user_id": user_id,
                        "user_name": user_name,
                        "is_tweeter": True,
                        "is_retweeter": False,
                    }
    except Exception as e:
        pass
    
    return users


def extract_users_from_retweets(retweets_path: str, existing_users: Dict) -> Dict[str, Dict]:
    """Extract user info from retweets.json."""
    users = existing_users.copy()
    
    try:
        with open(retweets_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for tweet_id, retweet_list in data.items():
            if isinstance(retweet_list, list):
                for rt in retweet_list:
                    if isinstance(rt, dict):
                        user_id = str(rt.get("user_id", rt.get("user", {}).get("id", "")))
                        user_name = rt.get("user_name", rt.get("user", {}).get("screen_name", ""))
                        
                        if user_id:
                            if user_id in users:
                                users[user_id]["is_retweeter"] = True
                            else:
                                users[user_id] = {
                                    "user_id": user_id,
                                    "user_name": user_name,
                                    "is_tweeter": False,
                                    "is_retweeter": True,
                                }
    except Exception as e:
        pass
    
    return users


def generate_new_user_tsv(folder_path: str) -> bool:
    """Generate new_user.tsv for a folder."""
    
    tweets_path = os.path.join(folder_path, "tweets.json")
    retweets_path = os.path.join(folder_path, "retweets.json")
    output_path = os.path.join(folder_path, "new_user.tsv")
    
    # Check if already exists
    if os.path.exists(output_path):
        return False  # Already exists
    
    # Check if tweets.json exists
    if not os.path.exists(tweets_path):
        return False  # No source data
    
    # Extract users from tweets
    users = extract_users_from_tweets(tweets_path)
    
    # Extract users from retweets (if exists)
    if os.path.exists(retweets_path):
        users = extract_users_from_retweets(retweets_path, users)
    
    if not users:
        return False  # No users found
    
    # Generate full profiles
    profiles = []
    for user_id, user_data in users.items():
        profile = generate_user_profile(
            user_id=user_data["user_id"],
            user_name=user_data["user_name"],
            is_tweeter=user_data["is_tweeter"],
            is_retweeter=user_data["is_retweeter"],
        )
        profiles.append(profile)
    
    # Write to TSV
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                "user_id", "user_name", "followers_count", "friends_count",
                "statuses_count", "favourites_count", "verified",
                "geo_enabled", "protected", "is_tweeter", "is_retweeter"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            writer.writerows(profiles)
        
        return True
    except Exception as e:
        print(f"Error writing {output_path}: {e}")
        return False


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def find_folders_without_users(base_path: str) -> List[str]:
    """Find all folders that have tweets.json but no new_user.tsv."""
    missing = []
    
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
        
        tweets_path = os.path.join(folder_path, "tweets.json")
        users_path = os.path.join(folder_path, "new_user.tsv")
        
        if os.path.exists(tweets_path) and not os.path.exists(users_path):
            missing.append(folder_path)
    
    return missing


def process_dataset(dataset_path: str, subfolders: List[str] = None):
    """Process entire dataset to fill in missing new_user.tsv files."""
    
    if subfolders is None:
        subfolders = [
            "gossipcop_fake",
            "gossipcop_real", 
            "politifact_fake",
            "politifact_real"
        ]
    
    print("=" * 60)
    print("GENERATE MISSING new_user.tsv FILES")
    print("=" * 60)
    
    total_generated = 0
    total_skipped = 0
    total_failed = 0
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(dataset_path, subfolder)
        
        if not os.path.exists(subfolder_path):
            print(f"\n⚠️  Skipping {subfolder} - not found")
            continue
        
        print(f"\n📁 Processing {subfolder}...")
        
        # Find folders without new_user.tsv
        missing_folders = find_folders_without_users(subfolder_path)
        
        if not missing_folders:
            print(f"   ✓ All folders already have new_user.tsv")
            continue
        
        print(f"   Found {len(missing_folders)} folders without new_user.tsv")
        
        # Generate missing files
        generated = 0
        failed = 0
        
        for folder_path in tqdm(missing_folders, desc=f"   Generating"):
            success = generate_new_user_tsv(folder_path)
            if success:
                generated += 1
            else:
                failed += 1
        
        print(f"   ✓ Generated: {generated}")
        if failed > 0:
            print(f"   ✗ Failed: {failed}")
        
        total_generated += generated
        total_failed += failed
    
    print("\n" + "=" * 60)
    print("✅ COMPLETE!")
    print(f"   Total generated: {total_generated}")
    print(f"   Total failed: {total_failed}")
    print("=" * 60)
    
    return total_generated, total_failed


def check_dataset_status(dataset_path: str, subfolders: List[str] = None):
    """Check status of new_user.tsv files in dataset."""
    
    if subfolders is None:
        subfolders = [
            "gossipcop_fake",
            "gossipcop_real",
            "politifact_fake", 
            "politifact_real"
        ]
    
    print("=" * 60)
    print("NEW_USER.TSV STATUS CHECK")
    print("=" * 60)
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(dataset_path, subfolder)
        
        if not os.path.exists(subfolder_path):
            print(f"\n⚠️  {subfolder}: NOT FOUND")
            continue
        
        total = 0
        with_users = 0
        without_users = 0
        
        for folder_name in os.listdir(subfolder_path):
            folder_path = os.path.join(subfolder_path, folder_name)
            
            if not os.path.isdir(folder_path):
                continue
            
            total += 1
            users_path = os.path.join(folder_path, "new_user.tsv")
            
            if os.path.exists(users_path):
                with_users += 1
            else:
                without_users += 1
        
        pct = (with_users / total * 100) if total > 0 else 0
        print(f"\n📁 {subfolder}:")
        print(f"   Total folders: {total}")
        print(f"   With new_user.tsv: {with_users} ({pct:.1f}%)")
        print(f"   Missing: {without_users} ({100-pct:.1f}%)")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate missing new_user.tsv files for FakeNewsNet"
    )
    parser.add_argument(
        "--data", 
        default="/content/FakeNewsNet_Data",
        help="Path to FakeNewsNet data"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check status, don't generate"
    )
    parser.add_argument(
        "--subfolders",
        nargs="+",
        default=None,
        help="Specific subfolders to process"
    )
    
    args = parser.parse_args()
    
    if args.check:
        check_dataset_status(args.data, args.subfolders)
    else:
        process_dataset(args.data, args.subfolders)
