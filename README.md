# HRV-Driven Adaptive Yoga Intensity Recommendation System

A comprehensive Python GUI application that provides personalized yoga recommendations based on Heart Rate Variability (HRV) and other health/lifestyle parameters.

## Features

### Core Functionality
- **HRV-Driven Recommendations**: Uses HRV RMSSD as a key health metric for adaptive intensity recommendations
- **Multiple ML Algorithms**: 
  - Collaborative Filtering (SVD)
  - Content-Based Filtering (Cosine Similarity)
  - Hybrid Recommendation System
  - Random Forest Classifier
  - Gradient Boosting Regressor

### User Interface
- **Data Management**: Load and preprocess datasets
- **Exploratory Data Analysis**: Interactive visualizations of health parameters
- **Model Training**: Train and evaluate multiple recommendation algorithms
- **Recommendation Engine**: Get personalized recommendations for existing users
- **New User Profile**: Input health parameters and get instant recommendations

### Health Parameters
- **Primary Metrics**: HRV RMSSD, Stress Index, Sleep Quality
- **Secondary Metrics**: BMI, SpO2, Mood Baseline
- **Lifestyle Factors**: Age, Gender, Activity Level, Flexibility Level, Yoga Experience
- **Medical History**: Chronic Conditions (diabetes, hypertension, arthritis, asthma)

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

## Usage

### 1. Data Loading
- Click "Load Sample Data" to use demonstration datasets
- Or "Load Custom Data" to import your own CSV files

### 2. Data Preprocessing
- Click "Preprocess Data" to encode categorical features and scale numerical values
- View feature correlations to understand parameter relationships

### 3. Model Training
- Train individual models:
  - Collaborative Filtering (SVD)
  - Content-Based Filtering
  - Random Forest Classifier
  - Gradient Boosting Regressor
- Compare model performance using "Compare Models"

### 4. Get Recommendations
**For Existing Users:**
- Select User ID from dropdown
- Choose recommendation type (CF, CB, or Hybrid)
- Click "Get Recommendations"

**For New Users:**
- Fill in health and lifestyle profile
- Click "Get Personalized Recommendations"
- View suitability scores for recommended asanas

### 5. Visualizations
- Health Parameter Distributions
- User Demographics
- Recommendation Score Analysis

## Technical Architecture

### Data Structure
- **Asanas Dataset**: 45 yoga poses with metadata (difficulty, intensity, benefits, contraindications)
- **Users Dataset**: Health and lifestyle parameters for 1000+ users
- **Sessions Dataset**: 5000+ session records with feedback scores

### Machine Learning Pipeline
1. **Feature Engineering**: Encode categorical variables, scale numerical features
2. **Collaborative Filtering**: SVD algorithm for user-item interactions
3. **Content-Based Filtering**: Cosine similarity between user health profile and asana features
4. **Hybrid Approach**: Weighted combination (60% CF + 40% CB)
5. **Classification**: Random Forest for predicting recommendation quality
6. **Regression**: Gradient Boosting for predicting raw scores

### Key Features
- **HRV Integration**: HRV RMSSD is a primary feature in all models
- **Adaptive Intensity**: Recommendations adapt based on stress levels and fitness
- **Personalization**: Uses comprehensive health profile for accurate recommendations
- **Real-time Inference**: Instant recommendations for new users

## Model Performance
- **Random Forest Accuracy**: ~82% (5-fold CV)
- **SVD RMSE**: ~0.14
- **Gradient Boosting R²**: ~0.85
- **Top Features**: HRV RMSSD, Stress Index, Sleep Quality, BMI

## File Structure
```
yoga/
|-- app.py                 # Main GUI application
|-- requirements.txt       # Python dependencies
|-- README.md             # This documentation
```

## Dependencies
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-learn >= 1.0.0
- scikit-surprise >= 1.1.0
- tkinter (included with Python)

## Notes
- The application uses sample data for demonstration
- HRV RMSSD values should be measured using proper heart rate monitoring devices
- Consult healthcare professionals before starting new yoga routines
- The system is designed to complement, not replace, professional medical advice
- Real-time adaptation based on user feedback
