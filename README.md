# ChatGPT Prompt Analyzer

A powerful tool for analyzing and categorizing ChatGPT prompts using machine learning and natural language processing.

## Features

- **Single Prompt Analysis**: Analyze individual prompts to determine their role category and characteristics
- **Batch Analysis**: Process multiple prompts at once and get aggregate statistics
- **Role Categories**: 
  - Technical (developers, programmers, engineers)
  - Creative (writers, artists, designers)
  - Educational (teachers, tutors, mentors)
  - Professional (analysts, managers, consultants)
  - Service (assistants, helpers, coordinators)
  - Other (specialized roles)
- **Prompt Comparison**: Compare two prompts to see their similarities and differences
- **Interactive Visualizations**: View feature importance, category distributions, and more

## Installation

1. Clone the repository:
```bash
git clone https://github.com/fahad0samara/ChatGPT-Prompt-Analyzer.git
cd chatgpt-prompt-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit app:
```bash
python -m streamlit run app.py
```

2. Use the navigation menu to access different tools:
   - Single Prompt Analysis
   - Batch Analysis
   - Prompt Comparison
   - Role Categories

## Model Training

The model uses both text features and TF-IDF vectorization for classification. To train the model:

```bash
python train_model.py
```

This will create the following files in the `models` directory:
- `prompt_classifier.joblib`: The trained classifier
- `tfidf_vectorizer.joblib`: TF-IDF vectorizer
- `feature_scaler.joblib`: Feature scaler
- `label_encoder.joblib`: Label encoder

## Project Structure

- `app.py`: Main Streamlit application
- `train_model.py`: Model training script
- `model_training.py`: Core model functionality
- `models/`: Trained model files
- `data/`: Training data and resources

## Dependencies

- streamlit
- pandas
- numpy
- scikit-learn
- plotly
- joblib

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - feel free to use this project for any purpose.
