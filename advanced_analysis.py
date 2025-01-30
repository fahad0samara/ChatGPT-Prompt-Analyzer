import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json

# Read the processed data
df = pd.read_csv('processed_chatgpt_prompts.csv')

# Create output directory
import os
if not os.path.exists('advanced_analysis'):
    os.makedirs('advanced_analysis')

# 1. TF-IDF Analysis and Clustering
tfidf = TfidfVectorizer(max_features=100, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['clean_prompt'])
tfidf_array = tfidf_matrix.toarray()

# Perform clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(tfidf_matrix)

# Get top terms per cluster
def get_top_terms_per_cluster(tfidf_array, kmeans, terms, n_terms=10):
    clusters = {}
    for i in range(n_clusters):
        center_ind = kmeans.labels_ == i
        center_terms = np.mean(tfidf_array[center_ind], axis=0)
        terms_sorted = [terms[i] for i in center_terms.argsort()[:-n_terms-1:-1]]
        clusters[f"Cluster {i}"] = terms_sorted
    return clusters

top_terms = get_top_terms_per_cluster(
    tfidf_array,
    kmeans, 
    tfidf.get_feature_names_out()
)

# Save cluster analysis
with open('advanced_analysis/cluster_analysis.json', 'w') as f:
    json.dump({
        'cluster_sizes': df['cluster'].value_counts().to_dict(),
        'top_terms_per_cluster': top_terms
    }, f, indent=4)

# 2. Prompt Complexity Analysis
complexity_metrics = {
    'word_count': df['prompt_word_count'],
    'char_count': df['prompt_char_count'],
    'avg_word_length': df['prompt_char_count'] / df['prompt_word_count'],
    'sentiment': df['prompt_sentiment'],
    'complexity': df['prompt_complexity']
}

# Calculate correlations
correlation_matrix = pd.DataFrame(complexity_metrics).corr()

# Save complexity analysis
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Complexity Metrics')
plt.tight_layout()
plt.savefig('advanced_analysis/complexity_correlation.png')
plt.close()

# 3. Role-based Pattern Analysis
role_patterns = {}
for role in df['role_category'].unique():
    role_data = df[df['role_category'] == role]
    role_patterns[role] = {
        'avg_metrics': {
            'word_count': role_data['prompt_word_count'].mean(),
            'sentiment': role_data['prompt_sentiment'].mean(),
            'complexity': role_data['prompt_complexity'].mean()
        },
        'interaction_types': role_data['interaction_type'].value_counts().to_dict()
    }

# Save role patterns
with open('advanced_analysis/role_patterns.json', 'w') as f:
    json.dump(role_patterns, f, indent=4)

# 4. Generate Similarity Matrix
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

# Find most similar prompts
most_similar_pairs = []
for i in range(len(df)):
    for j in range(i+1, len(df)):
        if similarity_matrix[i][j] > 0.5:  # Threshold for similarity
            most_similar_pairs.append({
                'prompt1': df.iloc[i]['act'],
                'prompt2': df.iloc[j]['act'],
                'similarity': float(similarity_matrix[i][j])  # Convert to float for JSON serialization
            })

# Sort by similarity
most_similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)

# Save similar pairs
with open('advanced_analysis/similar_prompts.json', 'w', encoding='utf-8') as f:
    json.dump(most_similar_pairs[:20], f, indent=4, ensure_ascii=False)

# 5. Generate HTML Report
html_report = """
<html>
<head>
    <title>Advanced ChatGPT Prompts Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        h1, h2 { color: #333; }
        .metric { margin: 10px 0; }
        pre { background: #f5f5f5; padding: 10px; border-radius: 5px; }
        img { max-width: 100%; height: auto; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>Advanced ChatGPT Prompts Analysis Report</h1>
    
    <div class="section">
        <h2>1. Cluster Analysis</h2>
        <p>The prompts have been clustered into groups based on their content similarity. Each cluster represents a different pattern or theme in the prompts.</p>
        <img src="complexity_correlation.png" alt="Complexity Correlation Matrix">
    </div>

    <div class="section">
        <h2>2. Role Patterns</h2>
        <p>Analysis of patterns within different role categories:</p>
        <pre id="role-patterns"></pre>
    </div>

    <div class="section">
        <h2>3. Similar Prompts</h2>
        <p>Pairs of prompts that are most similar in content:</p>
        <pre id="similar-prompts"></pre>
    </div>

    <script>
        fetch('role_patterns.json')
            .then(response => response.json())
            .then(data => {
                document.getElementById('role-patterns').textContent = JSON.stringify(data, null, 2);
            });

        fetch('similar_prompts.json')
            .then(response => response.json())
            .then(data => {
                document.getElementById('similar-prompts').textContent = JSON.stringify(data, null, 2);
            });
    </script>
</body>
</html>
"""

with open('advanced_analysis/advanced_report.html', 'w') as f:
    f.write(html_report)

print("Advanced analysis completed. Check the 'advanced_analysis' directory for results.")
