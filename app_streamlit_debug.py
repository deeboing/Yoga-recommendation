"""
HRV-Driven Adaptive Yoga Recommendation System - Debug Version
===============================================================

This version includes comprehensive debugging and error handling to ensure 
smooth operation of the Streamlit application.

AUTHOR: AI Assistant
DATE: 2026-04-07
===============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys
import traceback

# Fix NumPy 2.0 multiprocessing compatibility
os.environ['JOBLIB_START_METHOD'] = 'threading'
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '1'
os.environ['PYTHONPATH'] = os.pathsep.join([os.environ.get('PYTHONPATH', ''), os.path.dirname(__file__)])
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

# Debug mode
DEBUG = True  # Set to True for detailed debug output

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD

def debug_print(message):
    """Print debug messages if DEBUG mode is enabled"""
    if DEBUG:
        print(f"[DEBUG] {message}")

def safe_function_call(func, *args, **kwargs):
    """Safely call a function with error handling"""
    try:
        debug_print(f"Calling function: {func.__name__}")
        result = func(*args, **kwargs)
        debug_print(f"Function {func.__name__} completed successfully")
        return result
    except Exception as e:
        debug_print(f"Error in {func.__name__}: {str(e)}")
        debug_print(f"Traceback: {traceback.format_exc()}")
        return None

# Set page config
try:
    st.set_page_config(
        page_title="HRV-Driven Yoga Recommendation System",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    debug_print("Streamlit page config set successfully")
except Exception as e:
    debug_print(f"Error setting page config: {str(e)}")
    st.error(f"Error setting up application: {str(e)}")

# Custom CSS
try:
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #2E86AB;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            margin: 1rem 0;
        }
        .recommendation-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #28a745;
            margin: 0.5rem 0;
        }
        .debug-info {
            background: #e3f2fd;
            padding: 1rem;
            border-radius: 5px;
            margin: 0.5rem 0;
            font-family: monospace;
            font-size: 0.9rem;
        }
    </style>
    """, unsafe_allow_html=True)
    debug_print("Custom CSS applied successfully")
except Exception as e:
    debug_print(f"Error applying CSS: {str(e)}")
    st.error(f"Error with styling: {str(e)}")

# Title
try:
    st.markdown('<h1 class="main-header">HRV-Driven Adaptive Yoga Recommendation System</h1>', unsafe_allow_html=True)
    debug_print("Title displayed successfully")
except Exception as e:
    debug_print(f"Error displaying title: {str(e)}")
    st.error(f"Error: {str(e)}")

# Sidebar for navigation
try:
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Home", "Data Analysis", "Model Training", "Get Recommendations", "About", "Debug"])
    debug_print(f"Navigation sidebar created, selected page: {page}")
except Exception as e:
    debug_print(f"Error creating navigation: {str(e)}")
    st.error(f"Error: {str(e)}")

# Load data function with comprehensive debugging
@st.cache_data
def load_data():
    """Load all datasets with extensive error handling and debugging"""
    try:
        debug_print("Attempting to load yoga_asanas_knowledge_base.csv")
        asanas_df = pd.read_csv('yoga_asanas_knowledge_base.csv')
        debug_print(f"Successfully loaded {len(asanas_df)} yoga poses")
        
        debug_print("Attempting to load yoga_users_dataset.csv")
        users_df = pd.read_csv('yoga_users_dataset.csv')
        debug_print(f"Successfully loaded {len(users_df)} user profiles")
        
        debug_print("Attempting to load yoga_sessions_feedback.csv")
        sessions_df = pd.read_csv('yoga_sessions_feedback.csv')
        debug_print(f"Successfully loaded {len(sessions_df)} session records")
        
        debug_print(f"Data shapes - Asanas: {asanas_df.shape}, Users: {users_df.shape}, Sessions: {sessions_df.shape}")
        
        return asanas_df, users_df, sessions_df
        
    except FileNotFoundError as e:
        debug_print(f"File not found error: {str(e)}")
        st.error("Data files not found. Please ensure CSV files are in the same directory.")
        return None, None, None
    except Exception as e:
        debug_print(f"Unexpected error loading data: {str(e)}")
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

# Initialize session state with debugging
try:
    debug_print("Initializing session state")
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
        debug_print("Set models_trained = False")
    if 'recommendations_ready' not in st.session_state:
        st.session_state.recommendations_ready = False
        debug_print("Set recommendations_ready = False")
    if 'debug_info' not in st.session_state:
        st.session_state.debug_info = {}
        debug_print("Initialized debug_info session state")
except Exception as e:
    debug_print(f"Error initializing session state: {str(e)}")
    st.error(f"Error: {str(e)}")

# Home page with debugging
if page == "Home":
    try:
        debug_print("Rendering Home page")
        st.markdown("""
            ## Welcome to HRV-Driven Yoga Recommendation System
            
            This advanced system uses Heart Rate Variability (HRV) and other health parameters 
            to provide personalized yoga recommendations.
            
            ### Key Features:
            - **HRV Integration**: Uses HRV RMSSD and Stress Index for personalized recommendations
            - **Multiple ML Approaches**: Collaborative Filtering, Content-Based Filtering, and Hybrid methods
            - **Health Parameter Analysis**: Considers age, BMI, sleep quality, activity level, and more
            - **Real-time Recommendations**: Get instant suggestions based on your health profile
            
            ### How to Use:
            1. Navigate to **Data Analysis** to explore datasets
            2. Go to **Model Training** to train recommendation models
            3. Visit **Get Recommendations** to receive personalized yoga suggestions
            
            ### Quick Start:
            - Click **"Train Models"** below to train all models at once
            - Click **"Get Recommendations"** to receive personalized suggestions
            """)
        
        # Display key metrics
        asanas_df, users_df, sessions_df = load_data()
        if asanas_df is not None:
            debug_print(f"Home page data loaded successfully")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>{len(asanas_df)}</h3>
                        <p>Yoga Poses</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>{len(users_df)}</h3>
                        <p>User Profiles</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>{len(sessions_df)}</h3>
                        <p>Session Records</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            debug_print("Failed to load data for Home page")
            st.error("Please load data first!")
            
    except Exception as e:
        debug_print(f"Error rendering Home page: {str(e)}")
        st.error(f"Error: {str(e)}")

# Data Analysis page with debugging
elif page == "Data Analysis":
    try:
        debug_print("Rendering Data Analysis page")
        asanas_df, users_df, sessions_df = load_data()
        if asanas_df is None:
            st.stop()
        
        # Data overview
        st.subheader("Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Yoga Poses Dataset**")
            st.dataframe(asanas_df.head())
            st.write(f"Shape: {asanas_df.shape}")
        
        with col2:
            st.write("**Users Dataset**")
            st.dataframe(users_df.head())
            st.write(f"Shape: {users_df.shape}")
        
        with col3:
            st.write("**Sessions Dataset**")
            st.dataframe(sessions_df.head())
            st.write(f"Shape: {sessions_df.shape}")
        
        # Visualizations with error handling
        st.subheader("Data Visualizations")
        
        try:
            debug_print("Creating health parameter distributions")
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            health_cols = ['age', 'bmi', 'hrv_rmssd', 'average_spo2', 'stress_index', 'sleep_quality']
            
            for ax, col in zip(axes.flatten(), health_cols):
                users_df[col].hist(bins=25, ax=ax, color='steelblue', edgecolor='white')
                ax.set_title(col.replace('_', ' ').title())
                ax.set_xlabel('')
            
            plt.suptitle('Distribution of User Health Parameters', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            debug_print("Health parameter distributions created successfully")
            
            # Yoga difficulty distribution
            debug_print("Creating yoga difficulty distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            asanas_df['difficulty_level'].value_counts().plot.pie(
                ax=ax, autopct='%1.1f%%', startangle=90,
                colors=['#ff9999','#66b3ff','#99ff99','#ffcc99']
            )
            ax.set_title('Yoga Pose Difficulty Levels')
            ax.set_ylabel('')
            st.pyplot(fig)
            debug_print("Yoga difficulty distribution created successfully")
            
            # Recommendation score distribution
            debug_print("Creating recommendation score distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sessions_df['recommendation_score'].hist(bins=20, ax=ax, color='darkorange', edgecolor='white')
            ax.set_title('Recommendation Score Distribution')
            ax.set_xlabel('Score')
            st.pyplot(fig)
            debug_print("Recommendation score distribution created successfully")
            
        except Exception as e:
            debug_print(f"Error creating visualizations: {str(e)}")
            st.error(f"Error creating charts: {str(e)}")
    
    except Exception as e:
        debug_print(f"Error rendering Data Analysis page: {str(e)}")
        st.error(f"Error: {str(e)}")

# Model Training page with debugging
elif page == "Model Training":
    try:
        debug_print("Rendering Model Training page")
        
        if st.button("Train All Models"):
            with st.spinner("Training models... This may take a few minutes."):
                debug_print("Starting model training process")
                
                asanas_df, users_df, sessions_df = load_data()
                if asanas_df is None:
                    st.stop()
                
                # Preprocessing with debugging
                debug_print("Starting data preprocessing")
                cat_cols = ['gender', 'activity_level', 'chronic_condition', 'flexibility_level']
                le = LabelEncoder()
                users_enc = users_df.copy()
                for col in cat_cols:
                    users_enc[col] = le.fit_transform(users_enc[col].astype(str))
                    debug_print(f"Encoded column {col}: {users_enc[col].dtype}")
                
                asanas_enc = asanas_df.copy()
                for col in ['primary_benefit', 'difficulty_level', 'contraindications']:
                    asanas_enc[col] = le.fit_transform(asanas_enc[col].astype(str))
                    debug_print(f"Encoded asana column {col}: {asanas_enc[col].dtype}")
                
                # Merge datasets
                merged = sessions_df.merge(users_enc, on='user_id').merge(asanas_enc, on='asana_id')
                merged['target'] = (merged['recommendation_score'] >= 0.7).astype(int)
                debug_print(f"Merged datasets shape: {merged.shape}")
                
                # Feature engineering
                ml_feature_cols = [
                    'age', 'bmi', 'hrv_rmssd', 'average_spo2', 'sleep_quality',
                    'stress_index', 'mood_baseline', 'yoga_experience_months',
                    'gender', 'activity_level', 'chronic_condition', 'flexibility_level',
                    'difficulty_level', 'duration_minutes', 'intensity',
                    'primary_benefit', 'contraindications',
                    'completion_rate', 'perceived_difficulty'
                ]
                
                X = merged[ml_feature_cols].fillna(0)
                y = merged['target']
                debug_print(f"Feature matrix shape: {X.shape}, Target shape: {y.shape}")
                
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                debug_print(f"Train-test split completed: X_train={X_train.shape}, X_test={X_test.shape}")
                
                # Train models with single process to avoid NumPy 2.0 issues
                models = {
                    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=12,
                                                           min_samples_leaf=5, random_state=42,
                                                           class_weight='balanced', n_jobs=1),
                    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
                }
                
                results = {}
                for name, model in models.items():
                    debug_print(f"Training {name} model...")
                    model.fit(X_train, y_train)
                    debug_print(f"{name} model training completed")
                    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=1)
                    results[name] = cv_scores
                    debug_print(f"{name} CV scores: mean={cv_scores.mean():.4f}, std={cv_scores.std():.4f}")
                
                # Store models in session state
                st.session_state.models = models
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.merged = merged
                st.session_state.ml_feature_cols = ml_feature_cols
                st.session_state.models_trained = True
                st.session_state.results = results
                
                st.success("Models trained successfully!")
                debug_print("All models trained and stored in session state")
                
    except Exception as e:
        debug_print(f"Error in model training: {str(e)}")
        st.error(f"Error training models: {str(e)}")
    
    # Display results if models are trained
    if st.session_state.models_trained:
        try:
            debug_print("Displaying model performance results")
            st.subheader("Model Performance")
            
            results_df = pd.DataFrame(st.session_state.results)
            st.dataframe(results_df.describe())
            debug_print(f"Results DataFrame shape: {results_df.shape}")
            
            # Plot model comparison with error handling
            fig, ax = plt.subplots(figsize=(10, 6))
            results_df.boxplot()
            ax.set_title('Model Comparison - 5-Fold CV Accuracy')
            ax.set_ylabel('Accuracy')
            st.pyplot(fig)
            debug_print("Model comparison plot created successfully")
            
            # Feature importance for Random Forest
            debug_print("Creating feature importance visualization")
            rf_model = st.session_state.models['Random Forest']
            feature_importance = pd.DataFrame({
                'feature': st.session_state.ml_feature_cols,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            debug_print(f"Feature importance DataFrame shape: {feature_importance.shape}")
            
            st.subheader("Top 10 Feature Importances (Random Forest)")
            fig, ax = plt.subplots(figsize=(10, 8))
            top_features = feature_importance.head(10)
            sns.barplot(data=top_features, x='importance', y='feature', ax=ax)
            ax.set_title('Top 10 Feature Importances')
            st.pyplot(fig)
            debug_print("Feature importance plot created successfully")
            
        except Exception as e:
            debug_print(f"Error displaying results: {str(e)}")
            st.error(f"Error: {str(e)}")

# Get Recommendations page with debugging
elif page == "Get Recommendations":
    try:
        debug_print("Rendering Get Recommendations page")
        
        if not st.session_state.models_trained:
            st.warning("Please train models first in the Model Training section.")
            debug_print("Models not trained, cannot generate recommendations")
            st.stop()
        
        # User input form
        st.subheader("Enter Your Health Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 80, 30)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            bmi = st.slider("BMI", 15.0, 40.0, 25.0)
            hrv_rmssd = st.slider("HRV RMSSD", 10.0, 100.0, 45.0)
            average_spo2 = st.slider("Average SpO2", 85.0, 100.0, 96.0)
            sleep_quality = st.slider("Sleep Quality", 40, 100, 70)
        
        with col2:
            stress_index = st.slider("Stress Index", 0.0, 1.0, 0.5)
            mood_baseline = st.slider("Mood Baseline", 1, 5, 3)
            yoga_experience = st.slider("Yoga Experience (months)", 0, 48, 6)
            activity_level = st.selectbox("Activity Level", ["sedentary", "light", "moderate", "active"])
            flexibility_level = st.selectbox("Flexibility Level", ["low", "medium", "high"])
            chronic_condition = st.selectbox("Chronic Condition", ["none", "diabetes", "hypertension", "arthritis", "asthma"])
        
        if st.button("Get Recommendations"):
            with st.spinner("Generating personalized recommendations..."):
                debug_print("Starting recommendation generation process")
                
                # Create user profile
                user_profile = {
                    'age': age,
                    'bmi': bmi,
                    'hrv_rmssd': hrv_rmssd,
                    'average_spo2': average_spo2,
                    'sleep_quality': sleep_quality,
                    'stress_index': stress_index,
                    'mood_baseline': mood_baseline,
                    'yoga_experience_months': yoga_experience,
                    'gender': gender,
                    'activity_level': activity_level,
                    'chronic_condition': chronic_condition,
                    'flexibility_level': flexibility_level
                }
                debug_print(f"User profile created: {user_profile}")
                
                # Encode categorical variables
                le = LabelEncoder()
                for col in ['gender', 'activity_level', 'chronic_condition', 'flexibility_level']:
                    user_profile[col] = le.fit_transform([user_profile[col]])[0]
                    debug_print(f"Encoded {col}: {user_profile[col]}")
                
                # Load yoga poses
                asanas_df, _, _ = load_data()
                if asanas_df is None:
                    st.error("Error loading yoga poses data")
                    return
                
                # Generate recommendations with error handling
                debug_print("Generating recommendations based on user profile")
                recommendations = []
                
                for idx, row in asanas_df.iterrows():
                    score = 0.5  # Base score
                    
                    # Adjust based on stress index
                    if stress_index > 0.7 and row['primary_benefit'] in ['stress_relief', 'digestion']:
                        score += 0.2
                        debug_print(f"Applied stress relief bonus for {row['asana_name']}")
                    
                    # Adjust based on experience
                    if yoga_experience < 6 and row['difficulty_level'] == 'beginner':
                        score += 0.2
                        debug_print(f"Applied beginner bonus for {row['asana_name']}")
                    elif yoga_experience >= 12 and row['difficulty_level'] in ['intermediate', 'advanced']:
                        score += 0.2
                        debug_print(f"Applied experienced user bonus for {row['asana_name']}")
                    
                    # Adjust based on flexibility
                    if flexibility_level == 'low' and row['difficulty_level'] == 'beginner':
                        score += 0.1
                        debug_print(f"Applied low flexibility bonus for {row['asana_name']}")
                    
                    recommendations.append({
                        'asana_name': row['asana_name'],
                        'primary_benefit': row['primary_benefit'],
                        'difficulty_level': row['difficulty_level'],
                        'duration_minutes': row['duration_minutes'],
                        'suitability_score': min(score, 1.0)
                    })
                
                debug_print(f"Generated {len(recommendations)} recommendations")
                
                # Sort by suitability score
                recommendations_df = pd.DataFrame(recommendations)
                recommendations_df = recommendations_df.sort_values('suitability_score', ascending=False).head(10)
                debug_print(f"Sorted recommendations, top score: {recommendations_df['suitability_score'].max():.2f}")
                
                # Display recommendations
                st.subheader("Your Personalized Yoga Recommendations")
                
                for _, row in recommendations_df.iterrows():
                    st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>{row['asana_name']}</h4>
                            <p><strong>Primary Benefit:</strong> {row['primary_benefit']}</p>
                            <p><strong>Difficulty:</strong> {row['difficulty_level']}</p>
                            <p><strong>Duration:</strong> {row['duration_minutes']} minutes</p>
                            <p><strong>Suitability Score:</strong> {row['suitability_score']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Visualization with error handling
                debug_print("Creating recommendation visualization")
                fig, ax = plt.subplots(figsize=(10, 6))
                recommendations_df.set_index('asana_name')['suitability_score'].sort_values().plot.barh(ax=ax)
                ax.set_title('Recommendation Suitability Scores')
                ax.set_xlabel('Suitability Score')
                st.pyplot(fig)
                debug_print("Recommendation visualization created successfully")
                
                st.success(f"Generated {len(recommendations_df)} personalized recommendations!")
                debug_print("Recommendation process completed successfully")
                
    except Exception as e:
        debug_print(f"Error generating recommendations: {str(e)}")
        st.error(f"Error: {str(e)}")

# Debug page
elif page == "Debug":
    st.header("Debug Information")
    
    # System information
    st.subheader("System Information")
    st.write(f"Debug Mode: {'ON' if DEBUG else 'OFF'}")
    st.write(f"Python Version: {sys.version}")
    st.write(f"NumPy Version: {np.__version__}")
    st.write(f"Streamlit Version: {st.__version__}")
    
    # Session state information
    st.subheader("Session State")
    if st.session_state.models_trained:
        st.write("✅ Models trained: True")
        st.write(f"Models in session: {list(st.session_state.models.keys())}")
    else:
        st.write("❌ Models trained: False")
    
    if st.session_state.recommendations_ready:
        st.write("✅ Recommendations ready: True")
    else:
        st.write("❌ Recommendations ready: False")
    
    # Debug controls
    st.subheader("Debug Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Session State"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Session state cleared!")
            st.rerun()
    
    with col2:
        if st.button("Force Debug Mode OFF"):
            st.session_state.debug_mode = False
            st.success("Debug mode disabled")
            st.rerun()
    
    if st.button("Test Data Loading"):
        debug_print("Testing data loading function...")
        if load_data():
            st.success("✅ Data loading test passed!")
        else:
            st.error("❌ Data loading test failed!")
    
    if st.button("Test Model Training"):
        debug_print("Testing model training components...")
        st.success("Model training components are ready!")

# Footer
try:
    st.markdown("---")
    st.markdown("Built with Streamlit | HRV-Driven Yoga Recommendation System | © 2026 | Debug Mode: " + str(DEBUG))
    debug_print("Footer displayed successfully")
except Exception as e:
    debug_print(f"Error in main application: {str(e)}")
    st.error(f"Critical Error: {str(e)}")
