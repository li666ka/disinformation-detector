"""
Synthetic Data Quality Evaluation
==================================
Compare different synthetic data generation approaches.

Metrics:
1. Lexical Diversity (TTR, unique n-grams)
2. Semantic Diversity (embedding similarity)
3. Similarity to Original Data
4. Text Statistics
"""

import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Tuple
import re
from pathlib import Path

# ============================================================================
# INSTALL DEPENDENCIES IF NEEDED
# ============================================================================

def install_if_needed():
    import subprocess
    import sys
    packages = ["scikit-learn", "nltk"]
    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

try:
    install_if_needed()
except:
    pass

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

try:
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk import ngrams
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False


# ============================================================================
# DATA LOADING
# ============================================================================

def load_texts_from_folder(folder_path: str, max_samples: int = None) -> List[str]:
    """Load news texts from a folder of synthetic/real news."""
    texts = []
    
    if not os.path.exists(folder_path):
        print(f"⚠️ Path not found: {folder_path}")
        return texts
    
    subfolders = [f for f in os.listdir(folder_path) 
                  if os.path.isdir(os.path.join(folder_path, f))]
    
    if max_samples:
        subfolders = subfolders[:max_samples]
    
    for subfolder in subfolders:
        article_path = os.path.join(folder_path, subfolder, "news_article.json")
        if os.path.exists(article_path):
            try:
                with open(article_path, 'r', encoding='utf-8') as f:
                    article = json.load(f)
                    title = article.get("title", "")
                    text = article.get("text", "")
                    full_text = f"{title} {text}".strip()
                    if full_text:
                        texts.append(full_text)
            except:
                continue
    
    return texts


def load_from_csv(csv_path: str, text_column: str = "full_text", 
                  max_samples: int = None) -> List[str]:
    """Load texts from CSV file."""
    df = pd.read_csv(csv_path)
    texts = df[text_column].dropna().tolist()
    if max_samples:
        texts = texts[:max_samples]
    return texts


# ============================================================================
# LEXICAL DIVERSITY METRICS
# ============================================================================

def calculate_ttr(texts: List[str]) -> Dict[str, float]:
    """
    Calculate Type-Token Ratio (TTR).
    Higher = more diverse vocabulary.
    """
    all_words = []
    for text in texts:
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        all_words.extend(words)
    
    total_tokens = len(all_words)
    unique_types = len(set(all_words))
    
    ttr = unique_types / total_tokens if total_tokens > 0 else 0
    
    # Root TTR (more stable for different corpus sizes)
    root_ttr = unique_types / np.sqrt(total_tokens) if total_tokens > 0 else 0
    
    return {
        "total_tokens": total_tokens,
        "unique_types": unique_types,
        "ttr": ttr,
        "root_ttr": root_ttr
    }


def calculate_ngram_diversity(texts: List[str], n: int = 2) -> Dict[str, float]:
    """
    Calculate n-gram diversity.
    Higher unique ratio = more diverse.
    """
    all_ngrams = []
    
    for text in texts:
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        if len(words) >= n:
            text_ngrams = list(zip(*[words[i:] for i in range(n)]))
            all_ngrams.extend(text_ngrams)
    
    total_ngrams = len(all_ngrams)
    unique_ngrams = len(set(all_ngrams))
    
    unique_ratio = unique_ngrams / total_ngrams if total_ngrams > 0 else 0
    
    return {
        f"total_{n}grams": total_ngrams,
        f"unique_{n}grams": unique_ngrams,
        f"unique_{n}gram_ratio": unique_ratio
    }


def calculate_vocabulary_stats(texts: List[str]) -> Dict[str, float]:
    """Calculate vocabulary statistics."""
    all_words = []
    text_lengths = []
    
    for text in texts:
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        all_words.extend(words)
        text_lengths.append(len(words))
    
    word_freq = Counter(all_words)
    
    # Hapax legomena (words appearing only once)
    hapax = sum(1 for w, c in word_freq.items() if c == 1)
    
    return {
        "vocabulary_size": len(word_freq),
        "hapax_legomena": hapax,
        "hapax_ratio": hapax / len(word_freq) if word_freq else 0,
        "avg_text_length": np.mean(text_lengths),
        "std_text_length": np.std(text_lengths),
        "min_text_length": min(text_lengths) if text_lengths else 0,
        "max_text_length": max(text_lengths) if text_lengths else 0,
    }


# ============================================================================
# SEMANTIC DIVERSITY METRICS
# ============================================================================

def calculate_pairwise_similarity(texts: List[str], 
                                   sample_size: int = 500) -> Dict[str, float]:
    """
    Calculate pairwise cosine similarity using TF-IDF.
    Lower similarity = more diverse texts.
    """
    if len(texts) > sample_size:
        indices = np.random.choice(len(texts), sample_size, replace=False)
        texts_sample = [texts[i] for i in indices]
    else:
        texts_sample = texts
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts_sample)
    
    # Pairwise cosine similarity
    sim_matrix = cosine_similarity(tfidf_matrix)
    
    # Get upper triangle (excluding diagonal)
    upper_tri = sim_matrix[np.triu_indices(len(texts_sample), k=1)]
    
    return {
        "mean_pairwise_similarity": np.mean(upper_tri),
        "std_pairwise_similarity": np.std(upper_tri),
        "min_pairwise_similarity": np.min(upper_tri),
        "max_pairwise_similarity": np.max(upper_tri),
        "median_pairwise_similarity": np.median(upper_tri),
    }


def calculate_cluster_diversity(texts: List[str], 
                                 n_clusters: int = 10,
                                 sample_size: int = 1000) -> Dict[str, float]:
    """
    Cluster texts and measure how evenly distributed they are.
    More even distribution = more diverse.
    """
    if len(texts) > sample_size:
        indices = np.random.choice(len(texts), sample_size, replace=False)
        texts_sample = [texts[i] for i in indices]
    else:
        texts_sample = texts
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts_sample)
    
    # K-Means clustering
    n_clusters = min(n_clusters, len(texts_sample) // 10)
    if n_clusters < 2:
        return {"cluster_entropy": 0, "cluster_balance": 0}
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(tfidf_matrix)
    
    # Calculate cluster distribution
    cluster_counts = np.bincount(labels, minlength=n_clusters)
    cluster_probs = cluster_counts / len(labels)
    
    # Entropy (higher = more even distribution)
    entropy = -np.sum(cluster_probs * np.log(cluster_probs + 1e-10))
    max_entropy = np.log(n_clusters)
    normalized_entropy = entropy / max_entropy
    
    # Balance (how even the clusters are)
    balance = 1 - np.std(cluster_probs) / np.mean(cluster_probs)
    
    return {
        "n_clusters": n_clusters,
        "cluster_entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "cluster_balance": balance,
        "cluster_sizes": cluster_counts.tolist()
    }


# ============================================================================
# STYLE MARKERS
# ============================================================================

def analyze_style_markers(texts: List[str]) -> Dict[str, float]:
    """Analyze gossip/tabloid style markers."""
    
    markers = {
        "allegedly": 0,
        "reportedly": 0,
        "supposedly": 0,
        "sources": 0,
        "insider": 0,
        "exclusive": 0,
        "breaking": 0,
        "shocking": 0,
        "revealed": 0,
        "claims": 0,
        "rumor": 0,
        "secret": 0,
        "question_mark_title": 0,
        "exclamation_title": 0,
        "all_caps_words": 0,
    }
    
    for text in texts:
        text_lower = text.lower()
        
        for marker in ["allegedly", "reportedly", "supposedly", "sources", 
                       "insider", "exclusive", "breaking", "shocking", 
                       "revealed", "claims", "rumor", "secret"]:
            markers[marker] += text_lower.count(marker)
        
        # Title markers (first ~100 chars)
        title_part = text[:100]
        if "?" in title_part:
            markers["question_mark_title"] += 1
        if "!" in title_part:
            markers["exclamation_title"] += 1
        
        # All caps words
        all_caps = re.findall(r'\b[A-Z]{2,}\b', text)
        markers["all_caps_words"] += len(all_caps)
    
    # Normalize by number of texts
    n_texts = len(texts)
    normalized = {f"{k}_per_text": v / n_texts for k, v in markers.items()}
    
    return normalized


# ============================================================================
# COMPARISON FUNCTIONS
# ============================================================================

def compare_datasets(dataset1: List[str], dataset2: List[str],
                     name1: str = "Dataset 1", name2: str = "Dataset 2") -> pd.DataFrame:
    """Compare two datasets on all metrics."""
    
    print(f"\n📊 Comparing: {name1} ({len(dataset1)} texts) vs {name2} ({len(dataset2)} texts)")
    print("=" * 70)
    
    results = []
    
    # 1. TTR
    print("\n📝 Calculating lexical diversity...")
    ttr1 = calculate_ttr(dataset1)
    ttr2 = calculate_ttr(dataset2)
    
    for key in ttr1:
        results.append({
            "metric": f"TTR_{key}",
            name1: ttr1[key],
            name2: ttr2[key],
            "better": name1 if ttr1[key] > ttr2[key] else name2
        })
    
    # 2. N-gram diversity
    for n in [2, 3]:
        ng1 = calculate_ngram_diversity(dataset1, n)
        ng2 = calculate_ngram_diversity(dataset2, n)
        
        for key in ng1:
            results.append({
                "metric": key,
                name1: ng1[key],
                name2: ng2[key],
                "better": name1 if ng1[key] > ng2[key] else name2
            })
    
    # 3. Vocabulary stats
    vocab1 = calculate_vocabulary_stats(dataset1)
    vocab2 = calculate_vocabulary_stats(dataset2)
    
    for key in vocab1:
        better = name1 if vocab1[key] > vocab2[key] else name2
        if "length" in key:
            better = "-"  # Length isn't better/worse
        results.append({
            "metric": f"vocab_{key}",
            name1: vocab1[key],
            name2: vocab2[key],
            "better": better
        })
    
    # 4. Pairwise similarity
    print("📐 Calculating semantic similarity...")
    sim1 = calculate_pairwise_similarity(dataset1)
    sim2 = calculate_pairwise_similarity(dataset2)
    
    for key in sim1:
        # Lower similarity = more diverse = better
        better = name1 if sim1[key] < sim2[key] else name2
        results.append({
            "metric": f"semantic_{key}",
            name1: sim1[key],
            name2: sim2[key],
            "better": better
        })
    
    # 5. Cluster diversity
    print("🎯 Calculating cluster diversity...")
    clust1 = calculate_cluster_diversity(dataset1)
    clust2 = calculate_cluster_diversity(dataset2)
    
    for key in ["normalized_entropy", "cluster_balance"]:
        if key in clust1:
            results.append({
                "metric": f"cluster_{key}",
                name1: clust1[key],
                name2: clust2[key],
                "better": name1 if clust1[key] > clust2[key] else name2
            })
    
    # 6. Style markers
    print("✨ Analyzing style markers...")
    style1 = analyze_style_markers(dataset1)
    style2 = analyze_style_markers(dataset2)
    
    for key in style1:
        results.append({
            "metric": f"style_{key}",
            name1: style1[key],
            name2: style2[key],
            "better": "-"
        })
    
    df = pd.DataFrame(results)
    return df


def evaluate_single_dataset(texts: List[str], name: str = "Dataset") -> Dict:
    """Evaluate a single dataset."""
    
    print(f"\n📊 Evaluating: {name} ({len(texts)} texts)")
    print("=" * 60)
    
    results = {}
    
    print("📝 Lexical diversity...")
    results.update(calculate_ttr(texts))
    results.update(calculate_ngram_diversity(texts, 2))
    results.update(calculate_ngram_diversity(texts, 3))
    results.update(calculate_vocabulary_stats(texts))
    
    print("📐 Semantic similarity...")
    results.update(calculate_pairwise_similarity(texts))
    
    print("🎯 Cluster diversity...")
    results.update(calculate_cluster_diversity(texts))
    
    print("✨ Style markers...")
    results.update(analyze_style_markers(texts))
    
    return results


def print_comparison_summary(df: pd.DataFrame, name1: str, name2: str):
    """Print a summary of the comparison."""
    
    print("\n" + "=" * 70)
    print("📋 COMPARISON SUMMARY")
    print("=" * 70)
    
    # Key diversity metrics
    diversity_metrics = [
        "TTR_root_ttr",
        "unique_2gram_ratio",
        "unique_3gram_ratio",
        "vocab_vocabulary_size",
        "semantic_mean_pairwise_similarity",
        "cluster_normalized_entropy"
    ]
    
    print("\n🔑 KEY DIVERSITY METRICS:")
    print("-" * 70)
    
    for metric in diversity_metrics:
        row = df[df['metric'] == metric]
        if not row.empty:
            v1 = row[name1].values[0]
            v2 = row[name2].values[0]
            better = row['better'].values[0]
            
            # Format based on metric type
            if "similarity" in metric:
                # Lower is better for similarity
                marker = "⬇️" if better == name1 else "⬆️"
                print(f"  {metric}:")
                print(f"    {name1}: {v1:.4f} {marker if better == name1 else ''}")
                print(f"    {name2}: {v2:.4f} {marker if better == name2 else ''}")
            else:
                # Higher is better for diversity
                marker = "⬆️"
                print(f"  {metric}:")
                print(f"    {name1}: {v1:.4f} {marker if better == name1 else ''}")
                print(f"    {name2}: {v2:.4f} {marker if better == name2 else ''}")
    
    # Count wins
    wins = df[df['better'] != '-']['better'].value_counts()
    
    print("\n🏆 OVERALL SCORE:")
    print("-" * 70)
    for name, count in wins.items():
        print(f"  {name}: {count} metrics better")
    
    # Verdict
    winner = wins.idxmax() if not wins.empty else "Tie"
    print(f"\n✅ MORE DIVERSE DATASET: {winner}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate synthetic data quality")
    parser.add_argument("--data1", required=True, help="Path to first dataset folder")
    parser.add_argument("--data2", help="Path to second dataset folder (for comparison)")
    parser.add_argument("--name1", default="Dataset_1", help="Name for first dataset")
    parser.add_argument("--name2", default="Dataset_2", help="Name for second dataset")
    parser.add_argument("--max-samples", type=int, default=2000, help="Max samples to load")
    parser.add_argument("--output", help="Output CSV path for results")
    
    args = parser.parse_args()
    
    # Load data
    print(f"📂 Loading data from {args.data1}...")
    texts1 = load_texts_from_folder(args.data1, args.max_samples)
    print(f"   Loaded {len(texts1)} texts")
    
    if args.data2:
        print(f"📂 Loading data from {args.data2}...")
        texts2 = load_texts_from_folder(args.data2, args.max_samples)
        print(f"   Loaded {len(texts2)} texts")
        
        # Compare
        df = compare_datasets(texts1, texts2, args.name1, args.name2)
        print_comparison_summary(df, args.name1, args.name2)
        
        # Save results
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\n💾 Results saved to {args.output}")
        
        print("\n📊 FULL RESULTS TABLE:")
        print(df.to_string())
        
    else:
        # Single dataset evaluation
        results = evaluate_single_dataset(texts1, args.name1)
        
        print("\n📊 RESULTS:")
        print("-" * 50)
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
