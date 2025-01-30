import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Create directories for outputs
import os
for dir_name in ['models', 'scaled_data', 'model_evaluation', 'visualizations']:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def clean_text(text):
    """Clean and normalize text"""
    # Convert to lowercase and string
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s.,!?]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove standalone numbers but keep numbers within words
    text = re.sub(r'\b\d+\b', '', text)
    
    return text.strip()

def extract_features(text):
    """Extract features from text for classification"""
    # Initialize features dictionary
    features = {}
    
    # Basic text features
    words = text.split()
    features['word_count'] = max(1, len(words))  # Avoid division by zero
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
    
    # Keyword dictionaries for different aspects
    keywords = {
        'analysis': [
            'analyze', 'evaluate', 'assess', 'examine', 'investigate',
            'research', 'study', 'compare', 'measure', 'test', 'benchmark',
            'profile', 'optimize', 'performance', 'efficiency', 'metrics'
        ],
        'teaching': [
            'teach', 'explain', 'learn', 'understand', 'guide', 'show',
            'demonstrate', 'practice', 'exercise', 'example', 'tutorial',
            'lesson', 'concept', 'theory', 'principle', 'fundamentals'
        ],
        'guidance': [
            'help', 'assist', 'support', 'advise', 'suggest', 'recommend',
            'consult', 'mentor', 'coach', 'direct', 'facilitate', 'aid',
            'solve', 'resolve', 'troubleshoot', 'diagnose', 'fix'
        ],
        'creation': [
            'create', 'build', 'develop', 'design', 'implement', 'make',
            'generate', 'construct', 'write', 'code', 'program', 'compose',
            'craft', 'produce', 'establish', 'setup', 'configure'
        ]
    }
    
    # Technical domain keywords
    technical_domains = {
        'programming': [
            'code', 'function', 'class', 'method', 'variable', 'algorithm',
            'data structure', 'api', 'framework', 'library', 'module',
            'debug', 'compile', 'runtime', 'syntax', 'parameter'
        ],
        'web': [
            'html', 'css', 'javascript', 'api', 'rest', 'http', 'server',
            'client', 'database', 'frontend', 'backend', 'fullstack',
            'responsive', 'website', 'webapp', 'browser', 'dom'
        ],
        'system': [
            'architecture', 'infrastructure', 'platform', 'service',
            'cloud', 'container', 'docker', 'kubernetes', 'deployment',
            'scaling', 'monitoring', 'security', 'network', 'storage'
        ],
        'tools': [
            'git', 'github', 'gitlab', 'jenkins', 'jira', 'aws', 'azure',
            'gcp', 'terraform', 'ansible', 'puppet', 'chef', 'kubernetes',
            'docker', 'ci/cd', 'pipeline', 'testing'
        ]
    }
    
    # Calculate keyword-based scores
    for category, words_list in keywords.items():
        score = sum(1 for word in words_list if word in text.lower())
        features[f'{category}_score'] = score / features['word_count']
    
    # Calculate technical domain scores
    for domain, terms in technical_domains.items():
        score = sum(1 for term in terms if term.lower() in text.lower())
        features[f'{domain}_score'] = score / features['word_count']
    
    # Calculate technical term ratio
    all_technical_terms = [term.lower() for terms in technical_domains.values() for term in terms]
    technical_count = sum(1 for term in all_technical_terms if term in text.lower())
    features['technical_term_ratio'] = technical_count / features['word_count']
    
    # Pattern-based features
    patterns = {
        'analysis_pattern': r'\b(analyze|evaluate|assess|compare|measure)\b',
        'analysis_verb': r'\b(is|are|was|were|should|could|would|will)\b',
        'data_term': r'\b(data|information|statistics|metrics|numbers)\b',
        'quality_term': r'\b(quality|performance|efficiency|effectiveness)\b',
        'context_term': r'\b(context|scenario|situation|case|example)\b'
    }
    
    for pattern_name, pattern in patterns.items():
        matches = len(re.findall(pattern, text.lower()))
        features[f'{pattern_name}_score'] = matches / features['word_count']
    
    # Domain-specific scores
    domains = {
        'data_analysis': r'\b(data analysis|analytics|statistics|visualization)\b',
        'performance_analysis': r'\b(performance|optimization|efficiency|speed)\b',
        'quality_analysis': r'\b(quality|testing|validation|verification)\b',
        'security_analysis': r'\b(security|vulnerability|threat|risk)\b'
    }
    
    for domain_name, pattern in domains.items():
        matches = len(re.findall(pattern, text.lower()))
        features[f'{domain_name}_score'] = matches / features['word_count']
    
    return features

def features_to_array(features_dict):
    """Convert features dictionary to array in fixed order"""
    feature_order = [
        'word_count', 'avg_word_length', 'technical_term_ratio',
        'analysis_score', 'teaching_score', 'guidance_score', 'creation_score',
        'analysis_pattern_score', 'analysis_verb_score', 'data_term_score',
        'quality_term_score', 'context_term_score', 'data_analysis_score',
        'performance_analysis_score', 'quality_analysis_score', 'security_analysis_score',
        'programming_score', 'web_score', 'system_score', 'tools_score'
    ]
    return [features_dict[feature] for feature in feature_order]

def determine_interaction_type(text, features=None):
    """Determine the type of interaction based on text and features"""
    if features is None:
        features = extract_features(text)
    
    # Get scores for each category
    scores = {
        'analysis': features['analysis_score'],
        'teaching': features['teaching_score'],
        'guidance': features['guidance_score'],
        'creation': features['creation_score']
    }
    
    # Add weights to technical scores
    if features['technical_term_ratio'] > 0.1:
        for category in scores:
            if category in ['analysis', 'creation']:
                scores[category] *= 1.2
    
    # Return the category with highest score
    return max(scores.items(), key=lambda x: x[1])[0]

def categorize_role(role):
    """Categorize roles into main categories only"""
    role = role.lower()
    
    categories = {
        'technical': [
            'developer', 'programmer', 'engineer', 'coder', 'terminal', 'tech', 'sql', 'database',
            'python', 'javascript', 'web', 'software', 'api', 'system', 'computer', 'it', 'cyber',
            'security', 'infrastructure'
        ],
        'creative': [
            'writer', 'artist', 'designer', 'composer', 'creator', 'generator', 'storyteller',
            'poet', 'screenwriter', 'novelist', 'content', 'creative', 'music', 'visual'
        ],
        'educational': [
            'teacher', 'tutor', 'instructor', 'coach', 'trainer', 'educator', 'mentor',
            'guide', 'advisor', 'counselor', 'learning', 'teaching'
        ],
        'professional': [
            'analyst', 'researcher', 'scientist', 'consultant', 'manager', 'executive',
            'business', 'marketing', 'sales', 'financial', 'accountant', 'entrepreneur',
            'strategist', 'data', 'metrics', 'performance'
        ],
        'service': [
            'assistant', 'helper', 'support', 'service', 'translator', 'interpreter',
            'editor', 'proofreader', 'coordinator', 'planner', 'organizer'
        ]
    }
    
    # Check each category's keywords
    for category, keywords in categories.items():
        if any(keyword in role for keyword in keywords):
            return category
    
    return 'other'  # Default category

def predict_interaction_type(text):
    """Predict interaction type using both rule-based and ML approaches"""
    # Clean text
    cleaned_text = clean_text(text)
    
    # Extract features
    features = extract_features(cleaned_text)
    
    try:
        # Load ML model and scaler
        model = joblib.load('models/prompt_classifier.joblib')
        scaler = joblib.load('models/feature_scaler.joblib')
        le = joblib.load('models/label_encoder.joblib')
        
        # Prepare features for ML prediction
        feature_values = np.array([features_to_array(features)])
        X_scaled = scaler.transform(feature_values)
        
        # Get ML prediction
        ml_prediction = le.inverse_transform(model.predict(X_scaled))[0]
        ml_proba = np.max(model.predict_proba(X_scaled))
        
        # Get rule-based prediction
        rule_prediction = determine_interaction_type(cleaned_text, features)
        
        # Use ML prediction if confidence is high, otherwise use rule-based
        if ml_proba > 0.7:
            return ml_prediction
        else:
            return rule_prediction
            
    except (FileNotFoundError, ValueError):
        # If model files don't exist, use rule-based approach
        return determine_interaction_type(cleaned_text, features)

# Example usage of prediction function
if __name__ == "__main__":
    example_prompt = "I want you to act as a python tutor and help me understand classes"
    print("\nExample prediction:")
    print("Prompt:", example_prompt)
    print("Predicted interaction type:", predict_interaction_type(example_prompt))
