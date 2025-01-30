import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
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
import os

# Create directories for outputs
for dir_name in ['models', 'scaled_data', 'model_evaluation', 'visualizations']:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# Import feature extraction functions
from model_training import clean_text, extract_features, features_to_array, categorize_role

try:
    # Load and clean data
    print("Loading and cleaning data...")
    df = pd.read_csv('ChatGPT Prompts.csv')

    # Clean text
    df['clean_prompt'] = df['prompt'].apply(clean_text)
    df['clean_act'] = df['act'].apply(clean_text)

    # Categorize roles
    print("\nCategorizing roles...")
    df['role_category'] = df['clean_act'].apply(categorize_role)
    category_counts = df['role_category'].value_counts()
    print("\nRole category distribution:")
    print(category_counts)

    # Extract features
    print("\nExtracting features...")
    features_list = [extract_features(text) for text in df['clean_prompt']]
    features_df = pd.DataFrame(features_list)

    # Create TF-IDF features for text
    print("Creating TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    prompt_tfidf = tfidf.fit_transform(df['clean_prompt'])
    tfidf_df = pd.DataFrame(prompt_tfidf.toarray(), columns=tfidf.get_feature_names_out())

    # Combine all features
    X = pd.concat([features_df, tfidf_df], axis=1)

    # Convert to numpy array and handle NaN values
    X = X.fillna(0).values

    # Prepare target variable (role category)
    print("\nPreparing target labels...")
    le = LabelEncoder()
    y = le.fit_transform(df['role_category'])
    
    # Print label distribution
    label_counts = Counter(y)
    print(f"Number of unique categories: {len(label_counts)}")
    print("Category mapping:")
    for label, count in label_counts.items():
        print(f"Category {le.inverse_transform([label])[0]}: {count} samples")

    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save preprocessors
    joblib.dump(tfidf, 'models/tfidf_vectorizer.joblib')
    joblib.dump(scaler, 'models/feature_scaler.joblib')
    joblib.dump(le, 'models/label_encoder.joblib')

    # Train model
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=200,  # Increased from 100
        max_depth=10,      # Limit tree depth to prevent overfitting
        min_samples_split=5,  # Require more samples to split
        min_samples_leaf=2,   # Require at least 2 samples in leaves
        class_weight='balanced',  # Handle class imbalance
        random_state=42
    )

    # Determine number of cross-validation folds based on minimum class size
    min_samples_per_class = min(Counter(y).values())
    if min_samples_per_class >= 3:
        n_splits = 3
        print(f"\nPerforming {n_splits}-fold cross-validation...")
        cv_scores = cross_val_score(model, X_scaled, y, cv=n_splits)
        print("Cross-validation scores:", cv_scores)
        print("Average CV score:", cv_scores.mean())
    else:
        print("\nSkipping cross-validation due to insufficient samples per class")
        cv_scores = np.array([])

    # Train final model on full dataset
    model.fit(X_scaled, y)

    # Save model
    joblib.dump(model, 'models/prompt_classifier.joblib')

    # Generate evaluation metrics
    print("\nGenerating evaluation metrics...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    y_pred = model.predict(X_test)

    # Get classification report
    class_report = classification_report(y_test, y_pred, 
                                      target_names=le.classes_,
                                      output_dict=True)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Category')
    plt.xlabel('Predicted Category')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('model_evaluation/confusion_matrix.png')
    plt.close()

    # Plot feature importance
    feature_names = list(features_df.columns) + list(tfidf_df.columns)
    importances = pd.Series(model.feature_importances_, index=feature_names)
    plt.figure(figsize=(12, 6))
    importances.nlargest(20).plot(kind='bar')
    plt.title('Top 20 Most Important Features')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('model_evaluation/feature_importance.png')
    plt.close()

    # Save evaluation results
    evaluation_results = {
        'classification_report': class_report,
        'feature_importance': dict(zip(feature_names, model.feature_importances_)),
        'model_parameters': model.get_params(),
        'training_size': len(X_train),
        'test_size': len(X_test),
        'dataset_stats': {
            'total_samples': len(df),
            'num_classes': len(label_counts),
            'class_distribution': {le.inverse_transform([label])[0]: count 
                                 for label, count in label_counts.items()}
        }
    }

    if len(cv_scores) > 0:
        evaluation_results['cross_validation_scores'] = {
            'scores': cv_scores.tolist(),
            'mean': float(cv_scores.mean()),
            'std': float(cv_scores.std())
        }

    # Save raw predictions for analysis
    predictions_df = pd.DataFrame({
        'true_category': le.inverse_transform(y_test),
        'predicted_category': le.inverse_transform(y_pred),
        'correct': y_test == y_pred
    })
    predictions_df.to_csv('model_evaluation/predictions.csv', index=False)

    with open('model_evaluation/evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    print("\nTraining completed! Model and evaluation results saved.")
    print("\nModel performance summary:")
    print(f"Accuracy: {class_report['accuracy']:.2f}")
    print("\nPer-category F1 scores:")
    for category in le.classes_:
        if category in class_report:
            print(f"{category}: {class_report[category]['f1-score']:.2f}")

except Exception as e:
    print(f"\nError occurred: {str(e)}")
    import traceback
    traceback.print_exc()
