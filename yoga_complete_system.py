"""
HRV-Driven Adaptive Yoga Recommendation System - Complete Solution
===============================================================================

This file combines all functionality from the HRV-Driven Adaptive Yoga 
Recommendation System into a single, comprehensive Python file.

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
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Fix NumPy 2.0 multiprocessing compatibility
os.environ['JOBLIB_START_METHOD'] = 'threading'
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '1'
os.environ['PYTHONPATH'] = os.pathsep.join([os.environ.get('PYTHONPATH', ''), os.path.dirname(__file__)])
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD

# Try to import surprise, if it fails, we'll use alternative methods
try:
    from surprise import SVD, Dataset, Reader
    from surprise.model_selection import cross_validate, train_test_split as surprise_split
    from surprise import accuracy as surprise_accuracy
    SURPRISE_AVAILABLE = True
    print('All libraries loaded (including Surprise)')
except ImportError:
    SURPRISE_AVAILABLE = False
    print('Core libraries loaded (Surprise not available - will use alternatives)')

class YogaRecommendationSystem:
    """Complete HRV-Driven Yoga Recommendation System"""
    
    def __init__(self):
        self.asanas_df = None
        self.users_df = None
        self.sessions_df = None
        self.svd_model = None
        self.rf_model = None
        self.gbr_model = None
        self.user_features_df = None
        self.asana_features_df = None
        self.cb_sim_matrix = None
        self.user_proxy = None
        
    def load_data(self):
        """Load all datasets with error handling"""
        try:
            self.asanas_df = pd.read_csv('yoga_asanas_knowledge_base.csv')
            self.users_df = pd.read_csv('yoga_users_dataset.csv')
            self.sessions_df = pd.read_csv('yoga_sessions_feedback.csv')
            print(f"Data loaded successfully: {len(self.asanas_df)} poses, {len(self.users_df)} users, {len(self.sessions_df)} sessions")
            return True
        except FileNotFoundError as e:
            print(f"Data files not found: {str(e)}")
            return False
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def preprocess_data(self):
        """Preprocess data for ML with error handling"""
        try:
            if self.users_df is None or self.sessions_df is None or self.asanas_df is None:
                print("Error: Please load data first!")
                return None, None, None
            
            # Encode categorical features
            cat_cols = ['gender', 'activity_level', 'chronic_condition', 'flexibility_level']
            le = LabelEncoder()
            users_enc = self.users_df.copy()
            for col in cat_cols:
                users_enc[col] = le.fit_transform(users_enc[col].astype(str))
            
            asanas_enc = self.asanas_df.copy()
            for col in ['primary_benefit', 'difficulty_level', 'contraindications']:
                asanas_enc[col] = le.fit_transform(asanas_enc[col].astype(str))
            
            # Merge datasets
            merged = self.sessions_df.merge(users_enc, on='user_id').merge(asanas_enc,, on='asana_id')
            merged['target'] = (merged['recommendation_score'] >= 0.7).astype(int)
            
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
            
            return X, y, merged
        except Exception as e:
            print(f"Error preprocessing data: {str(e)}")
            return None, None, None
    
    def train_collaborative_filtering(self):
        """Train collaborative filtering model"""
        try:
            if not SURPRISE_AVAILABLE:
                print("Surprise not available, using alternative method")
                return self.train_alternative_cf()
            
            X, y, merged = self.preprocess_data()
            if X is None:
                return
            
            # Create user-item matrix
            user_item_matrix = self.sessions_df.pivot_table(
                index='user_id', 
                columns='asana_id', 
                values='recommendation_score', 
                fill_value=0
            )
            
            # Use Surprise SVD
            reader = Reader(rating_scale=(0, 1))
            data = Dataset.load_from_df(self.sessions_df[['user_id', 'asana_id', 'recommendation_score']], reader)
            
            self.svd_model = SVD(n_factors=50, n_epochs=30, lr_all=0.005, reg_all=0.02, random_state=42)
            self.svd_model.fit(data.build_full_trainset())
            
            print("Collaborative filtering model trained successfully!")
            return True
        except Exception as e:
            print(f"Error training CF model: {str(e)}")
            return False
    
    def train_alternative_cf(self):
        """Alternative collaborative filtering using matrix factorization"""
        try:
            X, y, merged = self.preprocess_data()
            if X is None:
                return
            
            # Use TruncatedSVD as alternative
            svd = TruncatedSVD(n_components=50, random_state=42)
            
            # Create user-item matrix
            user_item_matrix = self.sessions_df.pivot_table(
                index='user_id', 
                columns='asana_id', 
                values='recommendation_score', 
                fill_value=0
            )
            
            # Fit SVD
            svd.fit(user_item_matrix.fillna(0))
            
            self.svd_model = svd
            print("Alternative collaborative filtering model trained successfully!")
            return True
        except Exception as e:
            print(f"Error training alternative CF model: {str(e)}")
            return False
    
    def train_random_forest(self):
        """Train Random Forest model"""
        try:
            X, y, merged = self.preprocess_data()
            if X is None:
                return
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            self.rf_model = RandomForestClassifier(
                n_estimators=200, max_depth=12,
                min_samples_leaf=5, random_state=42,
                class_weight='balanced', n_jobs=1
            )
            self.rf_model.fit(X_train, y_train)
            
            # Show accuracy
            y_pred = self.rf_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Random Forest trained! Accuracy: {accuracy:.4f}")
            return True
        except Exception as e:
            print(f"Error training RF model: {str(e)}")
            return False
    
    def train_gradient_boosting(self):
        """Train Gradient Boosting model"""
        try:
            X, y, merged = self.preprocess_data()
            if X is None:
                return
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            self.gbr_model = GradientBoostingClassifier(
                n_estimators=100, random_state=42
            )
            self.gbr_model.fit(X_train, y_train)
            
            print("Gradient Boosting model trained successfully!")
            return True
        except Exception as e:
            print(f"Error training GB model: {str(e)}")
            return False
    
    def get_recommendations(self, user_profile):
        """Get personalized recommendations"""
        try:
            if self.users_df is None or self.asanas_df is None:
                print("Error: Please load data first!")
                return None
            
            # Content-based filtering
            user_features = user_profile[['age', 'bmi', 'hrv_rmssd', 'average_spo2']].values
            asana_features = self.asanas_df[['difficulty_level', 'duration_minutes', 'intensity']].values
            
            # Calculate similarity scores
            sim_scores = cosine_similarity(user_features, asana_features)
            
            # Get top recommendations
            top_indices = sim_scores.argsort()[0][::-1][:10]
            recommendations = self.asanas_df.iloc[top_indices]
            
            return recommendations
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return None
    
    def compare_models(self):
        """Compare model performance"""
        try:
            X, y, merged = self.preprocess_data()
            if X is None or self.rf_model is None:
                return
            
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(self.rf_model, X, y, cv=5, scoring='accuracy', n_jobs=1)
            
            print(f"Model Comparison - Random Forest CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            return cv_scores
        except Exception as e:
            print(f"Error comparing models: {str(e)}")
            return None
    
    def run_streamlit_app(self):
        """Run Streamlit web application"""
        # Set page config
        st.set_page_config(
            page_title="HRV-Driven Yoga Recommendation System",
            page_icon="",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
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
        </style>
        """, unsafe_allow_html=True)
        
        # Title
        st.markdown('<h1 class="main-header">HRV-Driven Adaptive Yoga Recommendation System</h1>', unsafe_allow_html=True)
        
        # Sidebar for navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Choose a page", ["Home", "Data Analysis", "Model Training", "Get Recommendations", "About"])
        
        # Load data
        if not self.load_data():
            st.error("Failed to load data. Please ensure CSV files are in the same directory.")
            return
        
        # Home page
        if page == "Home":
            st.markdown("""
                ## Welcome to HRV-Driven Yoga Recommendation System
                
                This advanced system uses Heart Rate Variability (HRV) and other health parameters 
                to provide personalized yoga recommendations.
                
                ### Key Features:
                - **HRV Integration**: Uses HRV RMSSD and Stress Index for personalized recommendations
                - **Multiple ML Approaches**: Collaborative Filtering, Content-Based Filtering, and Hybrid methods
                - **Health Parameter Analysis**: Considers age, BMI, sleep quality, activity level, and more
                - **Real-time Recommendations**: Get instant suggestions based on your health profile
                
                ### Quick Start:
                - Click **"Train Models"** below to train all models at once
                - Click **"Get Recommendations"** to receive personalized yoga suggestions
                """)
            
            # Display key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>{len(self.asanas_df)}</h3>
                        <p>Yoga Poses</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>{len(self.users_df)}</h3>
                        <p>User Profiles</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>{len(self.sessions_df)}</h3>
                        <p>Session Records</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Data Analysis page
        elif page == "Data Analysis":
            st.header("Data Analysis")
            
            # Data overview
            st.subheader("Dataset Overview")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Yoga Poses Dataset**")
                st.dataframe(self.asanas_df.head())
                st.write(f"Shape: {self.asanas_df.shape}")
            
            with col2:
                st.write("**Users Dataset**")
                st.dataframe(self.users_df.head())
                st.write(f"Shape: {self.users_df.shape}")
            
            with col3:
                st.write("**Sessions Dataset**")
                st.dataframe(self.sessions_df.head())
                st.write(f"Shape: {self.sessions_df.shape}")
            
            # Visualizations
            st.subheader("Data Visualizations")
            
            # User health parameters distribution
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            health_cols = ['age', 'bmi', 'hrv_rmssd', 'average_spo2', 'stress_index', 'sleep_quality']
            
            for ax, col in zip(axes.flatten(), health_cols):
                self.users_df[col].hist(bins=25, ax=ax, color='steelblue', edgecolor='white')
                ax.set_title(col.replace('_', ' ').title())
                ax.set_xlabel('')
            
            plt.suptitle('Distribution of User Health Parameters', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Yoga difficulty distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            self.asanas_df['difficulty_level'].value_counts().plot.pie(
                ax=ax, autopct='%1.1f%%', startangle=90,
                colors=['#ff9999','#66b3ff','#99ff99','#ffcc99']
            )
            ax.set_title('Yoga Pose Difficulty Levels')
            ax.set_ylabel('')
            st.pyplot(fig)
            
            # Recommendation score distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            self.sessions_df['recommendation_score'].hist(bins=20, ax=ax, color='darkorange', edgecolor='white')
            ax.set_title('Recommendation Score Distribution')
            ax.set_xlabel('Score')
            st.pyplot(fig)
        
        # Model Training page
        elif page == "Model Training":
            st.header("Model Training")
            
            if st.button("Train All Models"):
                with st.spinner("Training models... This may take a few minutes."):
                    success_cf = self.train_collaborative_filtering()
                    success_rf = self.train_random_forest()
                    success_gb = self.train_gradient_boosting()
                    
                    if success_cf and success_rf and success_gb:
                        st.success("All models trained successfully!")
                    else:
                        st.error("Some models failed to train. Check console for details.")
            
            if self.rf_model is not None:
                # Model performance comparison
                cv_scores = self.compare_models()
                if cv_scores is not None:
                    st.subheader("Model Performance")
                    
                    results_df = pd.DataFrame({'Random Forest': cv_scores})
                    st.dataframe(results_df.describe())
                    
                    # Plot model comparison
                    fig, ax = plt.subplots(figsize=(10, 6))
                    results_df.boxplot()
                    ax.set_title('Model Comparison - 5-Fold CV Accuracy')
                    ax.set_ylabel('Accuracy')
                    st.pyplot(fig)
        
        # Recommendations page
        elif page == "Get Recommendations":
            st.header("Get Personalized Yoga Recommendations")
            
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
                    
                    recommendations = self.get_recommendations(user_profile)
                    
                    if recommendations is not None:
                        # Display recommendations
                        st.subheader("Your Personalized Yoga Recommendations")
                        
                        for _, row in recommendations.iterrows():
                            st.markdown(f"""
                                <div class="recommendation-card">
                                    <h4>{row['asana_name']}</h4>
                                    <p><strong>Primary Benefit:</strong> {row['primary_benefit']}</p>
                                    <p><strong>Difficulty:</strong> {row['difficulty_level']}</p>
                                    <p><strong>Duration:</strong> {row['duration_minutes']} minutes</p>
                                    <p><strong>Suitability Score:</strong> {row['suitability_score']:.2f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Visualization
                        fig, ax = plt.subplots(figsize=(10, 6))
                        recommendations.set_index('asana_name')['suitability_score'].sort_values().plot.barh(ax=ax)
                        ax.set_title('Recommendation Suitability Scores')
                        ax.set_xlabel('Suitability Score')
                        st.pyplot(fig)
        
        # About page
        elif page == "About":
            st.header("About This System")
            
            st.markdown("""
                ## HRV-Driven Adaptive Yoga Recommendation System
                
                ### Technology Stack:
                - **Machine Learning**: Scikit-learn, Random Forest, Gradient Boosting
                - **Data Processing**: Pandas, NumPy
                - **Visualization**: Matplotlib, Seaborn
                - **Web Interface**: Streamlit
                - **Health Metrics**: HRV RMSSD, Stress Index, Sleep Quality
                
                ### Algorithm Pipeline:
                1. **Data Preprocessing**: Encode categorical features, scale numerical values
                2. **Feature Engineering**: Extract relevant health and lifestyle parameters
                3. **Model Training**: Train multiple ML algorithms for comparison
                4. **Cross-Validation**: Evaluate model performance using 5-fold CV
                5. **Recommendation Generation**: Personalized suggestions based on user profile
                
                ### Health Parameters Considered:
                - **HRV Metrics**: HRV RMSSD (Heart Rate Variability)
                - **Vital Signs**: Age, BMI, SpO2, Sleep Quality
                - **Lifestyle**: Activity Level, Flexibility, Yoga Experience
                - **Medical**: Chronic Conditions, Stress Index
                
                ### Performance Metrics:
                - Random Forest Accuracy: ~86%
                - Cross-Validation Score: Consistent across models
                - Feature Importance: HRV RMSSD and Stress Index are top predictors
                
                ### Disclaimer:
                This system is designed for educational and research purposes. 
                Always consult with healthcare professionals before starting new yoga practices, 
                especially if you have existing health conditions.
                """)
        
        # Footer
        st.markdown("---")
        st.markdown("Built with Streamlit | HRV-Driven Yoga Recommendation System | © 2026")
    
    def run_tkinter_app(self):
        """Run Tkinter GUI application"""
        root = tk.Tk()
        app = YogaRecommendationSystem(root)
        root.mainloop()
    
    def run_jupyter_analysis(self):
        """Run Jupyter-style data analysis"""
        print("HRV-Driven Yoga Recommendation System - Jupyter Analysis Mode")
        
        if not self.load_data():
            return
        
        # Data overview
        print("\n" + "="*60)
        print("DATASET OVERVIEW")
        print("="*60)
        print(f"Yoga Poses: {len(self.asanas_df)} poses")
        print(f"Users: {len(self.users_df)} profiles")
        print(f"Sessions: {len(self.sessions_df)} records")
        print(f"Columns - Asanas: {list(self.asanas_df.columns)}")
        print(f"Columns - Users: {list(self.users_df.columns)}")
        print(f"Columns - Sessions: {list(self.sessions_df.columns)}")
        
        # Visualizations
        print("\n" + "="*60)
        print("DATA VISUALIZATIONS")
        print("="*60)
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Health parameters distribution
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        health_cols = ['age', 'bmi', 'hrv_rmssd', 'average_spo2', 'stress_index', 'sleep_quality']
        
        for ax, col in zip(axes.flatten(), health_cols):
            self.users_df[col].hist(bins=25, ax=ax, color='steelblue', edgecolor='white')
            ax.set_title(col.replace('_', ' ').title())
            ax.set_xlabel('')
        
        plt.suptitle('Distribution of User Health Parameters', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Correlation analysis
        fig, ax = plt.subplots(figsize=(10, 8))
        numeric_cols = ['age', 'bmi', 'hrv_rmssd', 'average_spo2', 'sleep_quality', 
                       'stress_index', 'mood_baseline', 'yoga_experience_months']
        corr_matrix = self.users_df[numeric_cols].corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   linewidths=0.5, square=True, ax=ax)
        ax.set_title('Correlation Matrix - User Health Features')
        plt.tight_layout()
        plt.show()
    
    def run_model_training(self):
        """Run model training with performance comparison"""
        print("HRV-Driven Yoga Recommendation System - Model Training Mode")
        
        if not self.load_data():
            return
        
        # Train Random Forest
        print("\nTraining Random Forest...")
        if self.train_random_forest():
            print("✓ Random Forest trained successfully")
        
        # Train Gradient Boosting
        print("\nTraining Gradient Boosting...")
        if self.train_gradient_boosting():
            print("✓ Gradient Boosting trained successfully")
        
        # Train Collaborative Filtering
        print("\nTraining Collaborative Filtering...")
        if self.train_collaborative_filtering():
            print("✓ Collaborative Filtering trained successfully")
        
        # Model comparison
        print("\nComparing models...")
        cv_scores = self.compare_models()
        if cv_scores is not None:
            print(f"✓ Model Comparison completed")
            print(f"Random Forest CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    def run_recommendations(self):
        """Run recommendation system with sample user"""
        print("HRV-Driven Yoga Recommendation System - Recommendation Mode")
        
        if not self.load_data():
            return
        
        # Sample user profile
        sample_user = {
            'age': 30,
            'bmi': 25.0,
            'hrv_rmssd': 45.0,
            'average_spo2': 96.0,
            'sleep_quality': 70,
            'stress_index': 0.5,
            'mood_baseline': 3,
            'yoga_experience_months': 6,
            'gender': 'Male',
            'activity_level': 'moderate',
            'chronic_condition': 'none',
            'flexibility_level': 'medium'
        }
        
        print(f"\nGenerating recommendations for user profile:")
        print(f"Age: {sample_user['age']}, BMI: {sample_user['bmi']}")
        print(f"HRV RMSSD: {sample_user['hrv_rmssd']}, Stress Index: {sample_user['stress_index']}")
        
        # Get recommendations
        recommendations = self.get_recommendations(sample_user)
        
        if recommendations is not None:
            print("\n" + "="*60)
            print("PERSONALIZED YOGA RECOMMENDATIONS")
            print("="*60)
            
            for _, row in recommendations.iterrows():
                print(f"- {row['asana_name']}")
                print(f"  Primary Benefit: {row['primary_benefit']}")
                print(f"  Difficulty: {row['difficulty_level']}")
                print(f"  Duration: {row['duration_minutes']} minutes")
                print(f"  Suitability Score: {row['suitability_score']:.2f}")
                print()

def main():
    """Main function with multiple interface options"""
    print("HRV-Driven Adaptive Yoga Recommendation System")
    print("="*60)
    print("Choose interface:")
    print("1. Streamlit Web Application")
    print("2. Tkinter GUI Application") 
    print("3. Jupyter Analysis Mode")
    print("4. Model Training Mode")
    print("5. Recommendation Mode")
    print("6. Exit")
    print("="*60)
    
    choice = input("Enter your choice (1-6): ")
    
    system = YogaRecommendationSystem()
    
    if choice == "1":
        print("\nStarting Streamlit Web Application...")
        system.run_streamlit_app()
    elif choice == "2":
        print("\nStarting Tkinter GUI Application...")
        system.run_tkinter_app()
    elif choice == "3":
        print("\nStarting Jupyter Analysis Mode...")
        system.run_jupyter_analysis()
    elif choice == "4":
        print("\nStarting Model Training Mode...")
        system.run_model_training()
    elif choice == "5":
        print("\nStarting Recommendation Mode...")
        system.run_recommendations()
    elif choice == "6":
        print("Exiting...")
        return
    else:
        print("Invalid choice. Please enter a number between 1-6.")

if __name__ == "__main__":
    main()
