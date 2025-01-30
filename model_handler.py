import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

class PromptModelHandler:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.model_path = 'models/prompt_classifier.joblib'
        self.scaler_path = 'models/feature_scaler.joblib'
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Try to load existing model and scaler
        self.load_model()
    
    def load_data(self):
        """Load and preprocess training data"""
        # Load the ChatGPT prompts dataset
        df = pd.read_csv('ChatGPT Prompts.csv')
        
        # Clean and preprocess
        from model_training import clean_text, extract_features
        
        # Extract features for each prompt
        features_list = []
        for prompt in df['prompt']:
            cleaned_text = clean_text(prompt)
            features = extract_features(cleaned_text)
            features_list.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Define feature columns
        self.feature_columns = [
            'word_count', 'avg_word_length', 'technical_term_ratio',
            'analysis_pattern_score', 'analysis_verb_score',
            'data_term_score', 'quality_term_score', 'context_term_score',
            'data_analysis_score', 'performance_analysis_score',
            'quality_analysis_score', 'security_analysis_score',
            'teaching_score', 'guidance_score', 'creation_score',
            'programming_score', 'web_score', 'system_score', 'tools_score'
        ]
        
        X = features_df[self.feature_columns]
        y = df['act']  # Using 'act' as labels
        
        return X, y
    
    def train_model(self):
        """Train the model on the data"""
        print("Loading data...")
        X, y = self.load_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("Training model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        print(f"Train accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        
        # Save model and scaler
        self.save_model()
        
        return train_score, test_score
    
    def predict(self, features_dict):
        """Predict the category for a single prompt's features"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model or scaler not loaded. Run train_model() first.")
        
        # Convert features dict to array
        features = np.array([[features_dict[col] for col in self.feature_columns]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get prediction and probabilities
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        return prediction, probabilities
    
    def save_model(self):
        """Save model and scaler to disk"""
        if self.model is not None and self.scaler is not None:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            print(f"Model and scaler saved to {self.model_path}")
    
    def load_model(self):
        """Load model and scaler from disk"""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            print("Model and scaler loaded successfully")
        except:
            print("No saved model found. Train a new model using train_model()")

# Example usage
if __name__ == "__main__":
    handler = PromptModelHandler()
    
    # Train new model if needed
    if handler.model is None:
        handler.train_model()
    
    # Test prediction
    test_features = {
        'word_count': 10,
        'avg_word_length': 5.5,
        'technical_term_ratio': 0.3,
        'analysis_pattern_score': 0.8,
        'analysis_verb_score': 0.6,
        'data_term_score': 0.4,
        'quality_term_score': 0.5,
        'context_term_score': 0.7,
        'data_analysis_score': 0.9,
        'performance_analysis_score': 0.4,
        'quality_analysis_score': 0.6,
        'security_analysis_score': 0.3,
        'teaching_score': 0.2,
        'guidance_score': 0.3,
        'creation_score': 0.1,
        'programming_score': 0.5,
        'web_score': 0.4,
        'system_score': 0.3,
        'tools_score': 0.6
    }
    
    prediction, probabilities = handler.predict(test_features)
    print(f"\nTest Prediction: {prediction}")
    print(f"Probabilities: {probabilities}")
