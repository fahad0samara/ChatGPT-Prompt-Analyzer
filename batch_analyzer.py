import pandas as pd
import plotly.express as px
from model_training import extract_features, clean_text
from model_handler import PromptModelHandler
import json
from datetime import datetime

# Initialize model handler
model_handler = PromptModelHandler()
if model_handler.model is None:
    model_handler.train_model()

def batch_analyze_prompts(prompts_list):
    """Analyze multiple prompts and generate comprehensive report"""
    results = []
    
    for prompt in prompts_list:
        # Clean and extract features
        cleaned_text = clean_text(prompt)
        features = extract_features(cleaned_text)
        
        # Get ML prediction
        ml_prediction, probabilities = model_handler.predict(features)
        
        # Calculate rule-based scores
        rule_scores = {
            'analysis': (
                features['analysis_pattern_score'] * 2.0 +
                features['analysis_verb_score'] * 1.5 +
                features['data_term_score'] +
                features['quality_term_score'] +
                features['context_term_score'] +
                sum([
                    features['data_analysis_score'],
                    features['performance_analysis_score'],
                    features['quality_analysis_score'],
                    features['security_analysis_score']
                ])
            ),
            'teaching': (
                features['teaching_score'] * 1.5 +
                (2.0 if features['starts_with_teaching'] else 0) +
                (1.0 if features['has_basic_terms'] else 0)
            ),
            'guidance': (
                features['guidance_score'] * 1.5 +
                (2.0 if features['starts_with_guidance'] else 0)
            ),
            'creation': (
                features['creation_score'] * 1.5 +
                (2.0 if features['starts_with_creation'] else 0)
            )
        }
        
        # Use ML prediction if confidence is high
        max_prob = max(probabilities)
        if max_prob > 0.6:  # High confidence threshold
            prediction = ml_prediction
        else:
            # Use rule-based prediction
            max_score = max(rule_scores.values())
            prediction = max(rule_scores.items(), key=lambda x: x[1])[0] if max_score >= 1.0 else 'other'
        
        # Store results
        result = {
            'prompt': prompt,
            'prediction': prediction,
            'ml_prediction': ml_prediction,
            'ml_confidence': max_prob,
            'word_count': features['word_count'],
            'avg_word_length': features['avg_word_length'],
            'technical_term_ratio': features['technical_term_ratio'],
            'analysis_score': rule_scores['analysis'],
            'teaching_score': rule_scores['teaching'],
            'guidance_score': rule_scores['guidance'],
            'creation_score': rule_scores['creation']
        }
        results.append(result)
    
    return pd.DataFrame(results)

def generate_analysis_report(df):
    """Generate a comprehensive analysis report"""
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_prompts': len(df),
        'category_distribution': df['prediction'].value_counts().to_dict(),
        'ml_agreement': (df['prediction'] == df['ml_prediction']).mean(),
        'avg_ml_confidence': df['ml_confidence'].mean(),
        'avg_word_count': df['word_count'].mean(),
        'avg_technical_ratio': df['technical_term_ratio'].mean(),
        'category_scores': {
            'analysis': df['analysis_score'].mean(),
            'teaching': df['teaching_score'].mean(),
            'guidance': df['guidance_score'].mean(),
            'creation': df['creation_score'].mean()
        }
    }
    
    # Save report
    with open('analysis_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    return report

def plot_category_distribution(df):
    """Plot distribution of prompt categories"""
    fig = px.pie(
        names=df['prediction'].value_counts().index,
        values=df['prediction'].value_counts().values,
        title='Distribution of Prompt Categories'
    )
    return fig

def plot_score_distributions(df):
    """Plot distributions of different scores"""
    score_cols = ['analysis_score', 'teaching_score', 'guidance_score', 'creation_score']
    fig = px.box(
        df,
        y=score_cols,
        title='Distribution of Category Scores',
        points='all'
    )
    return fig

def plot_ml_performance(df):
    """Plot ML model performance metrics"""
    # Agreement between ML and rule-based predictions
    agreement_df = pd.DataFrame({
        'Category': df['prediction'].unique(),
        'Agreement Rate': [
            (df[df['prediction'] == cat]['ml_prediction'] == cat).mean()
            for cat in df['prediction'].unique()
        ]
    })
    
    fig = px.bar(
        agreement_df,
        x='Category',
        y='Agreement Rate',
        title='ML-Rule Agreement Rate by Category'
    )
    return fig

def analyze_prompt_complexity(df):
    """Analyze complexity metrics of prompts"""
    complexity_metrics = {
        'word_count': {
            'mean': df['word_count'].mean(),
            'median': df['word_count'].median(),
            'std': df['word_count'].std()
        },
        'avg_word_length': {
            'mean': df['avg_word_length'].mean(),
            'median': df['avg_word_length'].median(),
            'std': df['avg_word_length'].std()
        },
        'technical_term_ratio': {
            'mean': df['technical_term_ratio'].mean(),
            'median': df['technical_term_ratio'].median(),
            'std': df['technical_term_ratio'].std()
        },
        'ml_confidence': {
            'mean': df['ml_confidence'].mean(),
            'median': df['ml_confidence'].median(),
            'std': df['ml_confidence'].std()
        }
    }
    return complexity_metrics

# Example usage
if __name__ == "__main__":
    example_prompts = [
        "Analyze this dataset and identify trends",
        "Teach me about Python programming",
        "Help me debug this code",
        "Create a web scraping script",
        "Can you analyze the performance of this algorithm?",
        "Explain machine learning concepts",
        "Guide me through Docker setup",
        "Generate a REST API"
    ]
    
    # Analyze prompts
    results_df = batch_analyze_prompts(example_prompts)
    
    # Generate report
    report = generate_analysis_report(results_df)
    
    # Generate visualizations
    category_dist = plot_category_distribution(results_df)
    score_dist = plot_score_distributions(results_df)
    ml_perf = plot_ml_performance(results_df)
    
    # Analyze complexity
    complexity = analyze_prompt_complexity(results_df)
    
    print("Analysis complete! Check analysis_report.json for detailed results.")
