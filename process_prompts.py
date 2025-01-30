import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
from textblob import TextBlob

# Read the CSV file
df = pd.read_csv('ChatGPT Prompts.csv')

# Basic cleaning
def clean_text(text):
    # Remove special characters and extra spaces
    text = re.sub(r'[^\w\s]', ' ', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

# Feature engineering functions
def get_word_count(text):
    return len(str(text).split())

def get_char_count(text):
    return len(str(text))

def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

def get_complexity_score(text):
    sentences = TextBlob(str(text)).sentences
    if not sentences:
        return 0
    return sum(len(str(sentence).split()) for sentence in sentences) / len(sentences)

# Clean the data
df['clean_prompt'] = df['prompt'].apply(clean_text)
df['clean_act'] = df['act'].apply(clean_text)

# Feature engineering
df['prompt_word_count'] = df['prompt'].apply(get_word_count)
df['prompt_char_count'] = df['prompt'].apply(get_char_count)
df['prompt_sentiment'] = df['prompt'].apply(get_sentiment)
df['prompt_complexity'] = df['prompt'].apply(get_complexity_score)

# Add role categories
role_categories = {
    'technical': ['developer', 'engineer', 'programmer', 'sql', 'cyber', 'tech'],
    'creative': ['artist', 'designer', 'writer', 'creator'],
    'educational': ['teacher', 'instructor', 'tutor', 'coach'],
    'professional': ['manager', 'consultant', 'expert', 'specialist'],
    'healthcare': ['doctor', 'therapist', 'counselor', 'adviser']
}

def categorize_role(act):
    act = act.lower()
    for category, keywords in role_categories.items():
        if any(keyword in act for keyword in keywords):
            return category
    return 'other'

df['role_category'] = df['clean_act'].apply(categorize_role)

# Add interaction type
interaction_keywords = {
    'analysis': ['analyze', 'evaluate', 'assess'],
    'creation': ['create', 'generate', 'design'],
    'guidance': ['help', 'guide', 'assist'],
    'teaching': ['teach', 'explain', 'instruct']
}

def get_interaction_type(prompt):
    prompt = prompt.lower()
    for interaction, keywords in interaction_keywords.items():
        if any(keyword in prompt for keyword in keywords):
            return interaction
    return 'other'

df['interaction_type'] = df['clean_prompt'].apply(get_interaction_type)

# Save the processed dataset
output_file = 'processed_chatgpt_prompts.csv'
df.to_csv(output_file, index=False)

# Generate summary statistics
summary = {
    'total_prompts': len(df),
    'role_categories': df['role_category'].value_counts().to_dict(),
    'interaction_types': df['interaction_type'].value_counts().to_dict(),
    'avg_prompt_length': df['prompt_word_count'].mean(),
    'avg_sentiment': df['prompt_sentiment'].mean(),
    'avg_complexity': df['prompt_complexity'].mean()
}

# Save summary to a text file
with open('prompt_analysis_summary.txt', 'w') as f:
    f.write("ChatGPT Prompts Analysis Summary\n")
    f.write("===============================\n\n")
    f.write(f"Total number of prompts: {summary['total_prompts']}\n\n")
    
    f.write("Role Categories Distribution:\n")
    for category, count in summary['role_categories'].items():
        f.write(f"- {category}: {count}\n")
    f.write("\n")
    
    f.write("Interaction Types Distribution:\n")
    for itype, count in summary['interaction_types'].items():
        f.write(f"- {itype}: {count}\n")
    f.write("\n")
    
    f.write(f"Average prompt length (words): {summary['avg_prompt_length']:.2f}\n")
    f.write(f"Average sentiment score: {summary['avg_sentiment']:.2f}\n")
    f.write(f"Average complexity score: {summary['avg_complexity']:.2f}\n")
