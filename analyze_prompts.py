import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from collections import Counter
import json

# Read the processed data
df = pd.read_csv('processed_chatgpt_prompts.csv')

# Create a directory for saving visualizations
import os
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# 1. Role Category Distribution
plt.figure(figsize=(12, 6))
role_counts = df['role_category'].value_counts()
sns.barplot(x=role_counts.index, y=role_counts.values)
plt.title('Distribution of Role Categories')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/role_distribution.png')
plt.close()

# 2. Interaction Type Distribution
plt.figure(figsize=(12, 6))
interaction_counts = df['interaction_type'].value_counts()
sns.barplot(x=interaction_counts.index, y=interaction_counts.values)
plt.title('Distribution of Interaction Types')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/interaction_distribution.png')
plt.close()

# 3. Word Count Distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='prompt_word_count', bins=30)
plt.title('Distribution of Prompt Word Counts')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('visualizations/word_count_distribution.png')
plt.close()

# 4. Sentiment Analysis
plt.figure(figsize=(12, 6))
sns.boxplot(x='role_category', y='prompt_sentiment', data=df)
plt.title('Sentiment Distribution by Role Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/sentiment_by_role.png')
plt.close()

# 5. Complexity Analysis
plt.figure(figsize=(12, 6))
sns.boxplot(x='role_category', y='prompt_complexity', data=df)
plt.title('Complexity Distribution by Role Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/complexity_by_role.png')
plt.close()

# 6. Word Cloud of Prompts
text = ' '.join(df['clean_prompt'])
wordcloud = WordCloud(width=1600, height=800, background_color='white').generate(text)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig('visualizations/prompt_wordcloud.png')
plt.close()

# 7. Advanced Analysis
analysis_results = {
    'total_prompts': len(df),
    'avg_word_count': df['prompt_word_count'].mean(),
    'avg_sentiment': df['prompt_sentiment'].mean(),
    'avg_complexity': df['prompt_complexity'].mean(),
    'role_categories': df['role_category'].value_counts().to_dict(),
    'interaction_types': df['interaction_type'].value_counts().to_dict(),
    'sentiment_by_role': df.groupby('role_category')['prompt_sentiment'].mean().to_dict(),
    'complexity_by_role': df.groupby('role_category')['prompt_complexity'].mean().to_dict(),
    'correlation_matrix': df[['prompt_word_count', 'prompt_sentiment', 'prompt_complexity']].corr().round(3).to_dict(),
}

# 8. Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df[['prompt_word_count', 'prompt_sentiment', 'prompt_complexity']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Prompt Metrics')
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png')
plt.close()

# Save analysis results to JSON
with open('visualizations/analysis_results.json', 'w') as f:
    json.dump(analysis_results, f, indent=4)

# Generate HTML report
html_report = f"""
<html>
<head>
    <title>ChatGPT Prompts Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        .metric {{ margin: 20px 0; }}
        .metric h3 {{ color: #333; }}
    </style>
</head>
<body>
    <h1>ChatGPT Prompts Analysis Report</h1>
    
    <div class="metric">
        <h3>Basic Statistics</h3>
        <p>Total Prompts: {analysis_results['total_prompts']}</p>
        <p>Average Word Count: {analysis_results['avg_word_count']:.2f}</p>
        <p>Average Sentiment: {analysis_results['avg_sentiment']:.2f}</p>
        <p>Average Complexity: {analysis_results['avg_complexity']:.2f}</p>
    </div>

    <div class="metric">
        <h2>Visualizations</h2>
        <h3>Role Distribution</h3>
        <img src="role_distribution.png" alt="Role Distribution">
        
        <h3>Interaction Distribution</h3>
        <img src="interaction_distribution.png" alt="Interaction Distribution">
        
        <h3>Word Count Distribution</h3>
        <img src="word_count_distribution.png" alt="Word Count Distribution">
        
        <h3>Sentiment Analysis</h3>
        <img src="sentiment_by_role.png" alt="Sentiment by Role">
        
        <h3>Complexity Analysis</h3>
        <img src="complexity_by_role.png" alt="Complexity by Role">
        
        <h3>Word Cloud</h3>
        <img src="prompt_wordcloud.png" alt="Prompt Word Cloud">
        
        <h3>Correlation Heatmap</h3>
        <img src="correlation_heatmap.png" alt="Correlation Heatmap">
    </div>
</body>
</html>
"""

with open('visualizations/analysis_report.html', 'w') as f:
    f.write(html_report)
