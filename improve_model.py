import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
from model_training import clean_text, extract_features
import json
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading data and models...")
# Load the saved models and data
tfidf = joblib.load('models/tfidf_vectorizer.joblib')
scaler = joblib.load('models/feature_scaler.joblib')
model = joblib.load('models/random_forest_model.joblib')
le = joblib.load('models/label_encoder.joblib')

# Test examples for each category
test_examples = {
    'teaching': [
        "Act as a math teacher and explain calculus concepts step by step",
        "Be my programming mentor and help me understand object-oriented programming",
        "Teach me the basics of machine learning algorithms with practical examples",
        "I need you to explain quantum physics in simple terms for beginners",
        "Guide me through the process of learning Spanish grammar systematically",
        "Act as a professional instructor and help me master advanced Python concepts",
        "Can you be my tutor and explain the fundamentals of web development?",
        "Demonstrate how to solve differential equations with detailed explanations",
        "I want to learn about artificial intelligence, explain it from scratch",
        "Provide a comprehensive tutorial on database design principles"
    ],
    'guidance': [
        "Help me debug this Python code that has a recursion error",
        "Assist me in writing a professional email to my supervisor",
        "Guide me through the process of setting up a PostgreSQL database",
        "Can you help me improve my resume for a tech position?",
        "I need assistance with structuring my research paper",
        "Review my code and suggest optimization improvements",
        "Help me choose the best framework for my web application",
        "Guide me in implementing proper error handling in my API",
        "Assist me with troubleshooting my Docker container issues",
        "Can you help me understand why my program is crashing?"
    ],
    'creation': [
        "Write a Python script to analyze stock market data using pandas",
        "Create a beautiful responsive HTML template for my portfolio website",
        "Generate a creative story about space exploration in the year 3000",
        "Design a comprehensive database schema for an e-commerce platform",
        "Make a detailed project plan for developing a mobile app",
        "Build an API endpoint for user authentication using Flask",
        "Create a machine learning model for sentiment analysis",
        "Generate unit tests for my JavaScript application",
        "Develop a RESTful API for a blog platform",
        "Write a script to automate file processing tasks"
    ],
    'analysis': [
        # Data Analysis
        "Analyze this dataset and identify trends in customer behavior",
        "Analyze the data to find patterns in user engagement",
        "Can you analyze this data and provide insights about market trends?",
        "Please analyze these survey responses and summarize key findings",
        "Analyze the data in detail and create a comprehensive report",
        # Performance Analysis
        "Analyze the performance of this algorithm and suggest improvements",
        "Can you analyze why this query is running slowly?",
        "Analyze the system's resource usage and identify bottlenecks",
        "Please analyze the efficiency of our current caching strategy",
        "Analyze this code's time complexity and suggest optimizations",
        # Quality Analysis
        "Review this code and analyze its maintainability",
        "Analyze the quality of this software architecture",
        "Can you analyze these test results and identify issues?",
        "Please analyze our current testing strategy",
        "Analyze this pull request and provide detailed feedback",
        # Security Analysis
        "Analyze this system for potential security vulnerabilities",
        "Can you analyze our authentication implementation?",
        "Analyze these logs for suspicious activities",
        "Please analyze our current security measures",
        "Analyze this code for potential security risks"
    ]
}

def predict_with_confidence(prompt_text):
    # Clean and extract features
    cleaned_text = clean_text(prompt_text)
    features = extract_features(cleaned_text)
    
    # Create feature vector
    tfidf_features = tfidf.transform([cleaned_text]).toarray()
    numerical_features = scaler.transform([[
        features['word_count'],
        features['avg_word_length'],
        features['contains_question'],
        features['contains_exclamation'],
        features['sentence_count'],
        features['contains_code_terms'],
        features['technical_term_ratio'],
        features['teaching_score'],
        features['guidance_score'],
        features['creation_score'],
        features['analysis_score'],
        features['starts_with_teaching'],
        features['starts_with_guidance'],
        features['starts_with_creation'],
        features['starts_with_analysis'],
        features['has_complex_terms'],
        features['has_basic_terms'],
        features['has_role_terms'],
        features['mentions_expertise'],
        features['programming_score'],
        features['web_score'],
        features['system_score'],
        features['tools_score'],
        features['analysis_pattern_score'],
        features['performance_analysis_score'],
        features['quality_analysis_score'],
        features['security_analysis_score'],
        features['optimization_score'],
        features['debugging_score'],
        features['data_analysis_score'],
        features['technical_analysis_terms'],
        features['metrics_terms'],
        features['process_terms'],
        features['improvement_terms'],
        features['has_measurement_terms'],
        features['has_evaluation_terms'],
        features['has_improvement_terms'],
        features['starts_with_analysis_verb'],
        features['has_analysis_context']
    ]])
    
    # Combine features
    X = np.hstack([numerical_features, tfidf_features])
    
    # Get prediction probabilities
    proba = model.predict_proba(X)[0]
    pred_idx = model.predict(X)[0]
    prediction = le.inverse_transform([pred_idx])[0]
    
    # Get confidence score
    confidence = proba.max()
    
    return prediction, confidence

# Test and evaluate examples
print("\nTesting model with examples...")
results = []
for category, examples in test_examples.items():
    print(f"\nTesting {category} examples:")
    for example in examples:
        pred, conf = predict_with_confidence(example)
        results.append({
            'category': category,
            'example': example,
            'prediction': pred,
            'confidence': float(conf),  # Convert numpy float to Python float for JSON
            'correct': pred == category
        })
        print(f"Example: {example}")
        print(f"Predicted: {pred} (Confidence: {conf:.2%})")
        print(f"Correct: {'✓' if pred == category else '✗'}")
        print("-" * 80)

# Calculate accuracy per category
accuracy_by_category = {}
for category in test_examples.keys():
    category_results = [r for r in results if r['category'] == category]
    accuracy = sum(1 for r in category_results if r['correct']) / len(category_results)
    accuracy_by_category[category] = accuracy

# Plot results
plt.figure(figsize=(12, 6))
categories = list(accuracy_by_category.keys())
accuracies = list(accuracy_by_category.values())
colors = ['#2ecc71' if acc >= 0.8 else '#f1c40f' if acc >= 0.6 else '#e74c3c' for acc in accuracies]
plt.bar(categories, accuracies, color=colors)
plt.title('Model Accuracy by Category', fontsize=14, pad=20)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.savefig('model_evaluation/category_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()

# Calculate average confidence for correct and incorrect predictions
confidence_analysis = {
    'correct_predictions': {
        'count': sum(1 for r in results if r['correct']),
        'avg_confidence': np.mean([r['confidence'] for r in results if r['correct']])
    },
    'incorrect_predictions': {
        'count': sum(1 for r in results if not r['correct']),
        'avg_confidence': np.mean([r['confidence'] for r in results if not r['correct']])
    }
}

# Save detailed results
with open('model_evaluation/test_results.json', 'w') as f:
    json.dump({
        'individual_results': results,
        'accuracy_by_category': accuracy_by_category,
        'average_accuracy': sum(accuracies) / len(accuracies),
        'confidence_analysis': confidence_analysis
    }, f, indent=4)

print("\nResults have been saved to model_evaluation/test_results.json")
print("Category accuracy plot saved to model_evaluation/category_accuracy.png")

# Analyze and suggest improvements
print("\nModel Performance Analysis:")
print(f"Overall Accuracy: {sum(accuracies) / len(accuracies):.1%}")
print(f"Correct Predictions: {confidence_analysis['correct_predictions']['count']} (Avg Confidence: {confidence_analysis['correct_predictions']['avg_confidence']:.1%})")
print(f"Incorrect Predictions: {confidence_analysis['incorrect_predictions']['count']} (Avg Confidence: {confidence_analysis['incorrect_predictions']['avg_confidence']:.1%})")

print("\nCategory-specific Performance:")
for category, accuracy in accuracy_by_category.items():
    print(f"\n{category.title()} category (Accuracy: {accuracy:.1%}):")
    if accuracy < 0.8:
        print("Suggested improvements:")
        if category == 'teaching':
            print("- Add more educational context keywords")
            print("- Enhance detection of teaching patterns")
            print("- Consider pedagogical terminology")
        elif category == 'guidance':
            print("- Improve recognition of help-seeking patterns")
            print("- Add more support-related keywords")
            print("- Consider question patterns")
        elif category == 'creation':
            print("- Enhance detection of creative tasks")
            print("- Add more development-related terms")
            print("- Consider project scope indicators")
        elif category == 'analysis':
            print("- Strengthen analytical term detection")
            print("- Add more evaluation-related keywords")
            print("- Consider technical analysis patterns")

print("\nModel testing completed!")
