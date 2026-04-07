# HRV-Driven Adaptive Yoga Intensity Recommendation System

## Overview

This project implements a comprehensive yoga recommendation system that uses Heart Rate Variability (HRV) and other health parameters to provide personalized yoga recommendations. The system combines multiple machine learning techniques including collaborative filtering, content-based filtering, and hybrid approaches.

## Features

- **HRV Integration**: Uses HRV RMSSD and Stress Index for personalized recommendations
- **Multiple ML Approaches**: 
  - Collaborative Filtering (SVD/TruncatedSVD)
  - Content-Based Filtering (Cosine Similarity)
  - Hybrid Recommender System
  - Random Forest Classifier
  - Gradient Boosting Models
- **NumPy 2.0 Compatible**: Fully compatible with latest NumPy version
- **Comprehensive Error Handling**: Robust variable dependency management
- **Multiple Interfaces**: Jupyter notebook, GUI application, and standalone scripts

## Dataset

The system uses three main datasets:
- `yoga_asanas_knowledge_base.csv` - 45 yoga poses with metadata
- `yoga_users_dataset.csv` - 1000 users with health & lifestyle parameters  
- `yoga_sessions_feedback.csv` - 5000 session records with feedback scores

## Health Parameters

The system considers the following health parameters:
- **HRV Metrics**: HRV RMSSD, Stress Index
- **Vital Signs**: Age, BMI, SpO2, Sleep Quality
- **Lifestyle**: Activity Level, Flexibility, Chronic Conditions
- **Experience**: Yoga Experience Months, Mood Baseline

## Installation

### Prerequisites

- Python 3.8+
- NumPy 2.0+ (compatible)
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/deeboing/Yoga-recommendation.git
cd Yoga-recommendation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the system test:
```bash
python yoga_system_complete.py
```

## Usage

### 1. Jupyter Notebook

Open `yoga_recommendation_system.ipynb` and run cells sequentially:

```bash
jupyter notebook yoga_recommendation_system.ipynb
```

### 2. GUI Application

Run the desktop GUI application:

```bash
python app.py
```

### 3. Complete System Test

Verify everything works correctly:

```bash
python yoga_system_complete.py
```

### 4. Emergency Fix

If you encounter any issues in Jupyter cells, use the emergency fix:

```python
# Copy this into any problematic cell
from yoga_system_complete import emergency_fix
emergency_fix()
```

## Project Structure

```
Yoga-recommendation/
|
|-- yoga_recommendation_system.ipynb    # Main Jupyter notebook
|-- yoga_system_complete.py           # Complete solution & fixes
|-- app.py                            # GUI application
|-- requirements.txt                  # Dependencies
|-- README.md                         # This file
|
|-- yoga_asanas_knowledge_base.csv     # Yoga poses dataset
|-- yoga_users_dataset.csv            # Users health dataset
|-- yoga_sessions_feedback.csv        # Sessions feedback dataset
```

## Algorithm Pipeline

1. **Data Loading & EDA**: Load and explore datasets
2. **Preprocessing**: Encode categorical features, scale numeric features
3. **Collaborative Filtering**: SVD/TruncatedSVD for user-item recommendations
4. **Content-Based Filtering**: Cosine similarity based on health parameters
5. **Hybrid Recommender**: Combine CF and CB approaches
6. **ML Classification**: Random Forest for recommendation score prediction
7. **Model Evaluation**: Cross-validation and performance comparison

## Key Features

### HRV-Driven Recommendations
- Uses HRV RMSSD to assess autonomic nervous system state
- Stress Index calculation for personalized intensity levels
- Adaptive recommendations based on current physiological state

### Multi-Model Approach
- **Collaborative Filtering**: Matrix factorization for user similarity
- **Content-Based**: Health parameter to yoga pose matching
- **Hybrid**: Weighted combination of multiple approaches
- **Classification**: Predict recommendation success probability

### Robust Error Handling
- Automatic variable initialization
- NumPy 2.0 compatibility fixes
- Graceful fallback mechanisms
- Comprehensive error recovery

## Performance Metrics

- **Random Forest Accuracy**: ~86%
- **Cross-Validation Score**: 5-fold CV with consistent performance
- **Collaborative Filtering RMSE**: ~0.047
- **Hybrid Recommendation Precision**: Optimized with alpha=0.6

## Technologies Used

- **Core Libraries**: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn
- **ML Algorithms**: SVD, TruncatedSVD, Random Forest, Gradient Boosting
- **Compatibility**: NumPy 2.0, Python 3.8+
- **Visualization**: Matplotlib, Seaborn for comprehensive plots

## Troubleshooting

### Common Issues

1. **NumPy 2.0 Compatibility**: Already fixed in `yoga_system_complete.py`
2. **Variable NameError**: Use `emergency_fix()` function
3. **Import Issues**: Run `python yoga_system_complete.py` first
4. **Jupyter Cell Errors**: Copy emergency fix into problematic cells

### Getting Help

1. Run the complete system test first
2. Check the emergency fix function
3. Use the GUI application for easier interaction
4. Review the Jupyter notebook for detailed examples

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `python yoga_system_complete.py`
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built with HRV research and yoga science principles
- Incorporates machine learning best practices
- Designed for personalized wellness recommendations

## Contact

For questions, issues, or contributions, please use the GitHub repository issues section.

---

**Note**: This system is designed for educational and research purposes. Always consult with healthcare professionals before starting new yoga practices, especially if you have existing health conditions.

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
=======
# Yoga-recommendation
Ai based yoga recommendation system using basic ML elements.
>>>>>>> fcf6c08e76bd41c802d245a56bdf31a873e8ef83
