import streamlit as st

# Must be the first Streamlit command
st.set_page_config(page_title="ChatGPT Prompt Analyzer", page_icon="🔍", layout="wide")

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from model_training import extract_features, clean_text, categorize_role
from prompt_suggestions import PromptSuggester
from batch_analyzer import batch_analyze_prompts, generate_analysis_report
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and preprocessors
@st.cache_resource
def load_model_and_preprocessors():
    try:
        model = joblib.load('models/prompt_classifier.joblib')
        tfidf = joblib.load('models/tfidf_vectorizer.joblib')
        scaler = joblib.load('models/feature_scaler.joblib')
        label_encoder = joblib.load('models/label_encoder.joblib')
        return model, tfidf, scaler, label_encoder
    except Exception as e:
        st.error(f"Model files not found. Please train the model first. Error: {str(e)}")
        return None, None, None, None

model, tfidf, scaler, label_encoder = load_model_and_preprocessors()

def predict_role_category(prompt):
    """Predict the role category for a given prompt"""
    if model is None or tfidf is None or scaler is None or label_encoder is None:
        st.error("Models not loaded. Cannot make predictions.")
        return "Unknown", 0.0
        
    try:
        # Clean text
        clean_prompt = clean_text(prompt)
        
        # Extract features
        features = extract_features(clean_prompt)
        features_df = pd.DataFrame([features])
        
        # Create TF-IDF features
        prompt_tfidf = tfidf.transform([clean_prompt])
        tfidf_df = pd.DataFrame(prompt_tfidf.toarray(), columns=tfidf.get_feature_names_out())
        
        # Combine features
        X = pd.concat([features_df, tfidf_df], axis=1)
        X = X.fillna(0).values
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        pred = model.predict(X_scaled)
        prob = model.predict_proba(X_scaled)
        
        return label_encoder.inverse_transform(pred)[0], prob[0][pred[0]]
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return "Error", 0.0

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Sidebar
st.sidebar.title("Navigation")
tool = st.sidebar.radio("Select Tool", ["Single Prompt Analysis", "Batch Analysis", "Prompt Comparison", "Role Categories"])

if tool == "Single Prompt Analysis":
    st.title("ChatGPT Prompt Analyzer 🔍")
    
    # Input section
    prompt = st.text_area("Enter your prompt:", height=100)
    
    if st.button("Analyze"):
        if prompt:
            # Clean and analyze the prompt
            cleaned_prompt = clean_text(prompt)
            features = extract_features(cleaned_prompt)
            
            # Get role category prediction
            category, confidence = predict_role_category(prompt)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Role Category Analysis")
                st.write(f"**Predicted Category:** {category}")
                st.write(f"**Confidence Score:** {confidence:.2f}")
                
                # Show feature importance for this prediction
                if model is not None:
                    feature_importance = pd.DataFrame({
                        'Feature': list(features.keys()),
                        'Importance': model.feature_importances_[:len(features)]
                    })
                    feature_importance = feature_importance.sort_values('Importance', ascending=False)
                    
                    fig = px.bar(feature_importance, x='Importance', y='Feature',
                               title='Feature Importance',
                               orientation='h')
                    st.plotly_chart(fig)
            
            with col2:
                st.subheader("Text Analysis")
                metrics = {
                    'Word Count': len(cleaned_prompt.split()),
                    'Avg Word Length': np.mean([len(word) for word in cleaned_prompt.split()]),
                    'Technical Terms': features.get('technical_term_score', 0),
                    'Teaching Terms': features.get('teaching_score', 0),
                    'Analysis Terms': features.get('analysis_score', 0)
                }
                
                for metric, value in metrics.items():
                    st.metric(metric, f"{value:.2f}")
            
            # Add to history
            st.session_state.history.append({
                'prompt': prompt,
                'category': category,
                'confidence': confidence
            })
            
            # Show history
            if st.session_state.history:
                st.subheader("Analysis History")
                history_df = pd.DataFrame(st.session_state.history)
                st.dataframe(history_df)

elif tool == "Batch Analysis":
    st.title("Batch Prompt Analysis 📊")
    
    uploaded_file = st.file_uploader("Upload CSV file with prompts", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        if 'prompt' in df.columns:
            if st.button("Analyze Batch"):
                # Process each prompt
                results = []
                progress_bar = st.progress(0)
                
                for i, row in df.iterrows():
                    category, confidence = predict_role_category(row['prompt'])
                    results.append({
                        'prompt': row['prompt'],
                        'category': category,
                        'confidence': confidence
                    })
                    progress_bar.progress((i + 1) / len(df))
                
                results_df = pd.DataFrame(results)
                
                # Display summary
                st.subheader("Analysis Results")
                st.dataframe(results_df)
                
                # Show category distribution
                st.subheader("Category Distribution")
                fig = px.pie(results_df, names='category', title='Distribution of Role Categories')
                st.plotly_chart(fig)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "Download Results",
                    csv,
                    "batch_analysis_results.csv",
                    "text/csv",
                    key='download-csv'
                )
        else:
            st.error("CSV file must contain a 'prompt' column")

elif tool == "Role Categories":
    st.title("Role Categories Explorer 🎯")
    
    # Display category information
    categories = {
        'technical': 'Technical roles like developers, programmers, and engineers',
        'creative': 'Creative roles like writers, artists, and designers',
        'educational': 'Educational roles like teachers, tutors, and mentors',
        'professional': 'Professional roles like analysts, managers, and consultants',
        'service': 'Service roles like assistants, helpers, and coordinators',
        'other': 'Other specialized or unique roles'
    }
    
    st.subheader("Available Categories")
    for category, description in categories.items():
        st.write(f"**{category.title()}**: {description}")
    
    # Test categorization
    st.subheader("Test Role Categorization")
    test_prompt = st.text_area("Enter a prompt to test categorization:", height=100)
    
    if st.button("Categorize"):
        if test_prompt:
            category, confidence = predict_role_category(test_prompt)
            
            st.write(f"**Predicted Category:** {category}")
            st.write(f"**Confidence Score:** {confidence:.2f}")
            
            # Show example prompts for the predicted category
            st.subheader(f"Example Prompts for {category.title()} Category")
            examples = {
                'technical': [
                    "Act as a Python developer and help me write clean code",
                    "Be a database engineer and optimize my SQL queries"
                ],
                'creative': [
                    "Act as a creative writer and help me write a story",
                    "Be an artist and describe a painting concept"
                ],
                'educational': [
                    "Act as a math teacher and explain calculus",
                    "Be a language tutor and help me learn Spanish"
                ],
                'professional': [
                    "Act as a business analyst and review my metrics",
                    "Be a management consultant and optimize my workflow"
                ],
                'service': [
                    "Act as a personal assistant and organize my schedule",
                    "Be a translator and help me with this document"
                ],
                'other': [
                    "Act as a meditation guide",
                    "Be a travel advisor"
                ]
            }
            
            for example in examples.get(category, []):
                st.write(f"- {example}")

elif tool == "Prompt Comparison":
    st.title("Prompt Comparison Tool 🔄")
    
    col1, col2 = st.columns(2)
    
    with col1:
        prompt1 = st.text_area("Enter first prompt:", height=100, key="prompt1")
    
    with col2:
        prompt2 = st.text_area("Enter second prompt:", height=100, key="prompt2")
    
    if st.button("Compare"):
        if prompt1 and prompt2:
            # Analyze both prompts
            category1, conf1 = predict_role_category(prompt1)
            category2, conf2 = predict_role_category(prompt2)
            
            # Display results
            st.subheader("Comparison Results")
            
            comparison_df = pd.DataFrame({
                'Metric': ['Category', 'Confidence', 'Word Count'],
                'Prompt 1': [
                    category1,
                    f"{conf1:.2f}",
                    len(clean_text(prompt1).split())
                ],
                'Prompt 2': [
                    category2,
                    f"{conf2:.2f}",
                    len(clean_text(prompt2).split())
                ]
            })
            
            st.table(comparison_df)
            
            # Show similarity score
            common_words = set(clean_text(prompt1).split()) & set(clean_text(prompt2).split())
            similarity = len(common_words) / max(len(clean_text(prompt1).split()), len(clean_text(prompt2).split()))
            
            st.metric("Similarity Score", f"{similarity:.2f}")

# Footer
st.markdown("---")
st.markdown("""
### About this Tool

This tool analyzes ChatGPT prompts and categorizes them into different role types. It uses machine learning to identify:

* Role categories (technical, creative, educational, etc.)
* Prompt characteristics and patterns
* Confidence scores for predictions

The model is trained on a diverse dataset of prompts and uses both text features and TF-IDF vectorization for classification.
""")
