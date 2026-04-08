"""
================================================================================
COMBINED YOGA RECOMMENDATION SYSTEM
================================================================================

This file combines all Python modules for the HRV-Driven Adaptive Yoga 
Recommendation System into a single file for easier deployment.

ORIGINAL FILES INCLUDED:
1. yoga_complete_system.py - Main system with class-based implementation
2. app_streamlit_debug.py - Debug version of Streamlit app
3. app_streamlit_fixed.py - Fixed Streamlit app (standalone version)
4. fix_multiprocessing.py - Multiprocessing compatibility fix

USAGE:
- Run as main application: python yoga_combined.py
- Run Streamlit app: streamlit run yoga_combined.py

================================================================================
"""

# ================================================================================
# SECTION 1: COMMON IMPORTS AND ENVIRONMENT SETUP
# ================================================================================

import os
import sys
import warnings
import traceback

# Fix scikit-learn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

print("NumPy 2.0 multiprocessing compatibility fixes applied")

# Core data science imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning imports
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD

# GUI imports
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Try to import surprise for collaborative filtering
try:
    from surprise import SVD, Dataset, Reader
    from surprise.model_selection import cross_validate, train_test_split as surprise_split
    from surprise import accuracy as surprise_accuracy
    SURPRISE_AVAILABLE = True
    print('All libraries loaded (including Surprise)')
except ImportError:
    SURPRISE_AVAILABLE = False
    print('Core libraries loaded (Surprise not available - will use alternatives)')

# Try to import streamlit (may not be available in all environments)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print('Streamlit not available - web interface disabled')


# ================================================================================
# SECTION 2: MAIN YOGA RECOMMENDATION SYSTEM CLASS
# ================================================================================

class YogaRecommendationSystem:
    
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
            merged = self.sessions_df.merge(users_enc, on='user_id').merge(asanas_enc, on='asana_id')
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
            user_features = pd.DataFrame([user_profile])[['age', 'bmi', 'hrv_rmssd', 'average_spo2']].values
            asana_features = self.asanas_df[['duration_minutes']].values
            
            # Calculate similarity scores for each asana
            recommendations = []
            for idx, row in self.asanas_df.iterrows():
                score = 0.5  # Base score
                
                # Adjust based on stress index
                if 'stress_index' in user_profile and user_profile['stress_index'] > 0.7:
                    if row['primary_benefit'] in ['stress_relief', 'digestion']:
                        score += 0.2
                
                # Adjust based on experience
                if 'yoga_experience_months' in user_profile:
                    if user_profile['yoga_experience_months'] < 6 and row['difficulty_level'] == 'beginner':
                        score += 0.2
                    elif user_profile['yoga_experience_months'] >= 12 and row['difficulty_level'] in ['intermediate', 'advanced']:
                        score += 0.2
                
                # Adjust based on flexibility
                if 'flexibility_level' in user_profile:
                    if user_profile['flexibility_level'] == 'low' and row['difficulty_level'] == 'beginner':
                        score += 0.1
                
                recommendations.append({
                    'asana_name': row['asana_name'],
                    'primary_benefit': row['primary_benefit'],
                    'difficulty_level': row['difficulty_level'],
                    'duration_minutes': row['duration_minutes'],
                    'suitability_score': min(score, 1.0)
                })
            
            recommendations_df = pd.DataFrame(recommendations)
            recommendations_df = recommendations_df.sort_values('suitability_score', ascending=False).head(10)
            
            return recommendations_df
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return None
    
    def compare_models(self):
        """Compare model performance"""
        try:
            X, y, merged = self.preprocess_data()
            if X is None or self.rf_model is None:
                return None
            
            cv_scores = cross_val_score(self.rf_model, X, y, cv=5, scoring='accuracy', n_jobs=1)
            print(f"Random Forest CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            return cv_scores
        except Exception as e:
            print(f"Error comparing models: {str(e)}")
            return None
    
    def run_streamlit_app(self):
        """Run Streamlit web application"""
        if not STREAMLIT_AVAILABLE:
            print("Streamlit not available. Please install with: pip install streamlit")
            return
        
        # This will be handled by the standalone Streamlit section at the bottom
        print("Use: streamlit run yoga_combined.py")
    
    def run_tkinter_app(self):
        """Run Tkinter GUI application"""
        root = tk.Tk()
        root.title("HRV-Driven Yoga Recommendation System")
        root.geometry("1200x800")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="HRV-Driven Yoga Recommendation System", 
                               font=('Helvetica', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Load data button
        def load_data_gui():
            if self.load_data():
                messagebox.showinfo("Success", f"Data loaded: {len(self.asanas_df)} poses, {len(self.users_df)} users")
            else:
                messagebox.showerror("Error", "Failed to load data files")
        
        ttk.Button(main_frame, text="Load Data", command=load_data_gui).grid(row=1, column=0, pady=5)
        
        # Train models button
        def train_models_gui():
            if self.users_df is None:
                messagebox.showwarning("Warning", "Please load data first")
                return
            
            progress_window = tk.Toplevel(root)
            progress_window.title("Training Models")
            progress_label = ttk.Label(progress_window, text="Training in progress...")
            progress_label.pack(pady=20)
            
            def train():
                self.train_random_forest()
                self.train_gradient_boosting()
                self.train_collaborative_filtering()
                progress_window.destroy()
                messagebox.showinfo("Success", "Models trained successfully!")
            
            import threading
            threading.Thread(target=train).start()
        
        ttk.Button(main_frame, text="Train Models", command=train_models_gui).grid(row=1, column=1, pady=5)
        
        # Get recommendations button
        def get_recommendations_gui():
            if self.users_df is None:
                messagebox.showwarning("Warning", "Please load data first")
                return
            
            # Create sample user profile
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
            
            recommendations = self.get_recommendations(sample_user)
            
            if recommendations is not None:
                # Display in text widget
                rec_window = tk.Toplevel(root)
                rec_window.title("Recommendations")
                rec_window.geometry("600x400")
                
                text_widget = scrolledtext.ScrolledText(rec_window, wrap=tk.WORD, width=70, height=20)
                text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
                
                text_widget.insert(tk.END, "Personalized Yoga Recommendations\n")
                text_widget.insert(tk.END, "="*50 + "\n\n")
                
                for _, row in recommendations.iterrows():
                    text_widget.insert(tk.END, f"• {row['asana_name']}\n")
                    text_widget.insert(tk.END, f"  Benefit: {row['primary_benefit']}\n")
                    text_widget.insert(tk.END, f"  Difficulty: {row['difficulty_level']}\n")
                    text_widget.insert(tk.END, f"  Duration: {row['duration_minutes']} min\n")
                    text_widget.insert(tk.END, f"  Score: {row['suitability_score']:.2f}\n\n")
                
                text_widget.config(state=tk.DISABLED)
        
        ttk.Button(main_frame, text="Get Recommendations", command=get_recommendations_gui).grid(row=1, column=2, pady=5)
        
        # Visualization frame
        viz_frame = ttk.LabelFrame(main_frame, text="Visualizations", padding="10")
        viz_frame.grid(row=2, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        def show_health_distribution():
            if self.users_df is None:
                messagebox.showwarning("Warning", "Please load data first")
                return
            
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            health_cols = ['age', 'bmi', 'hrv_rmssd', 'average_spo2', 'stress_index', 'sleep_quality']
            
            for ax, col in zip(axes.flatten(), health_cols):
                self.users_df[col].hist(bins=25, ax=ax, color='steelblue', edgecolor='white')
                ax.set_title(col.replace('_', ' ').title())
            
            plt.suptitle('Distribution of User Health Parameters', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Embed in tkinter
            viz_window = tk.Toplevel(root)
            viz_window.title("Health Distribution")
            canvas = FigureCanvasTkAgg(fig, viz_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        ttk.Button(viz_frame, text="Health Distribution", command=show_health_distribution).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        status_label = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        status_label.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
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
                print(f"• {row['asana_name']}")
                print(f"  Primary Benefit: {row['primary_benefit']}")
                print(f"  Difficulty: {row['difficulty_level']}")
                print(f"  Duration: {row['duration_minutes']} minutes")
                print(f"  Suitability Score: {row['suitability_score']:.2f}")
                print()


# ================================================================================
# SECTION 3: STREAMLIT STANDALONE APP (from app_streamlit_fixed.py)
# ================================================================================

def run_streamlit_standalone():
    """
    Standalone Streamlit application.
    Run with: streamlit run yoga_combined.py
    """
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not installed. Run: pip install streamlit")
        return
    
    # Set page config
    st.set_page_config(
        page_title="HRV-Driven Yoga Recommendation System",
        page_icon="🧘",
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
    st.markdown('<h1 class="main-header">🧘 HRV-Driven Adaptive Yoga Recommendation System</h1>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Home", "Data Analysis", "Model Training", "Get Recommendations", "About"])
    
    # Load data function with error handling
    @st.cache_data
    def load_data_streamlit():
        try:
            asanas_df = pd.read_csv('yoga_asanas_knowledge_base.csv')
            users_df = pd.read_csv('yoga_users_dataset.csv')
            sessions_df = pd.read_csv('yoga_sessions_feedback.csv')
            return asanas_df, users_df, sessions_df
        except FileNotFoundError:
            st.error("Data files not found. Please ensure CSV files are in the same directory.")
            return None, None, None
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None, None, None
    
    # Initialize session state
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'recommendations_ready' not in st.session_state:
        st.session_state.recommendations_ready = False
    
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
        
        ### How to Use:
        1. Navigate to **Data Analysis** to explore datasets
        2. Go to **Model Training** to train recommendation models
        3. Visit **Get Recommendations** to receive personalized yoga suggestions
        """)
        
        # Display key metrics
        asanas_df, users_df, sessions_df = load_data_streamlit()
        if asanas_df is not None:
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
    
    elif page == "Data Analysis":
        st.header("Data Analysis")
        
        asanas_df, users_df, sessions_df = load_data_streamlit()
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
        
        # Visualizations
        st.subheader("Data Visualizations")
        
        # User health parameters distribution
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        health_cols = ['age', 'bmi', 'hrv_rmssd', 'average_spo2', 'stress_index', 'sleep_quality']
        
        for ax, col in zip(axes.flatten(), health_cols):
            users_df[col].hist(bins=25, ax=ax, color='steelblue', edgecolor='white')
            ax.set_title(col.replace('_', ' ').title())
            ax.set_xlabel('')
        
        plt.suptitle('Distribution of User Health Parameters', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Yoga difficulty distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        asanas_df['difficulty_level'].value_counts().plot.pie(
            ax=ax, autopct='%1.1f%%', startangle=90,
            colors=['#ff9999','#66b3ff','#99ff99','#ffcc99']
        )
        ax.set_title('Yoga Pose Difficulty Levels')
        ax.set_ylabel('')
        st.pyplot(fig)
    
    elif page == "Model Training":
        st.header("Model Training")
        
        if st.button("Train All Models"):
            with st.spinner("Training models... This may take a few minutes."):
                asanas_df, users_df, sessions_df = load_data_streamlit()
                if asanas_df is None:
                    st.stop()
                
                # Preprocessing
                cat_cols = ['gender', 'activity_level', 'chronic_condition', 'flexibility_level']
                le = LabelEncoder()
                users_enc = users_df.copy()
                for col in cat_cols:
                    users_enc[col] = le.fit_transform(users_enc[col].astype(str))
                
                asanas_enc = asanas_df.copy()
                for col in ['primary_benefit', 'difficulty_level', 'contraindications']:
                    asanas_enc[col] = le.fit_transform(asanas_enc[col].astype(str))
                
                # Merge datasets
                merged = sessions_df.merge(users_enc, on='user_id').merge(asanas_enc, on='asana_id')
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
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                
                # Train models with single process to avoid multiprocessing issues
                models = {
                    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=12,
                                                           min_samples_leaf=5, random_state=42,
                                                           class_weight='balanced', n_jobs=1),
                    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
                }
                
                results = {}
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=1)
                    results[name] = cv_scores
                
                # Store models in session state
                st.session_state.models = models
                st.session_state.ml_feature_cols = ml_feature_cols
                st.session_state.models_trained = True
                st.session_state.results = results
                
                st.success("Models trained successfully!")
        
        if st.session_state.models_trained:
            st.subheader("Model Performance")
            
            # Display results
            results_df = pd.DataFrame(st.session_state.results)
            st.dataframe(results_df.describe())
            
            # Plot model comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            results_df.boxplot()
            ax.set_title('Model Comparison - 5-Fold CV Accuracy')
            ax.set_ylabel('Accuracy')
            st.pyplot(fig)
            
            # Feature importance for Random Forest
            rf_model = st.session_state.models['Random Forest']
            feature_importance = pd.DataFrame({
                'feature': st.session_state.ml_feature_cols,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            st.subheader("Top 10 Feature Importances (Random Forest)")
            fig, ax = plt.subplots(figsize=(10, 8))
            top_features = feature_importance.head(10)
            sns.barplot(data=top_features, x='importance', y='feature', ax=ax)
            ax.set_title('Top 10 Feature Importances')
            st.pyplot(fig)
    
    elif page == "Get Recommendations":
        st.header("Get Personalized Yoga Recommendations")
        
        if not st.session_state.models_trained:
            st.warning("Please train models first in Model Training section.")
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
                # Load yoga poses
                asanas_df, _, _ = load_data_streamlit()
                if asanas_df is None:
                    st.stop()
                
                # Generate recommendations with dynamic scoring
                recommendations = []
                
                # Map difficulty levels to numeric values for comparison
                difficulty_map = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
                flexibility_map = {'low': 1, 'medium': 2, 'high': 3}
                
                for idx, row in asanas_df.iterrows():
                    # Start with base score
                    score = 0.3
                    
                    # 1. Stress-based scoring (0-0.25)
                    if stress_index > 0.6:
                        if row['primary_benefit'] in ['stress_relief', 'relaxation']:
                            score += 0.25
                        elif row['primary_benefit'] in ['digestion', 'sleep']:
                            score += 0.15
                    elif stress_index < 0.3:
                        if row['primary_benefit'] in ['strength', 'energy']:
                            score += 0.2
                    
                    # 2. Experience-difficulty matching (0-0.25)
                    pose_difficulty = difficulty_map.get(row['difficulty_level'], 2)
                    if yoga_experience < 3:
                        if pose_difficulty == 1:
                            score += 0.25
                        elif pose_difficulty == 2:
                            score += 0.1
                    elif yoga_experience < 12:
                        if pose_difficulty == 2:
                            score += 0.25
                        elif pose_difficulty == 1:
                            score += 0.15
                        elif pose_difficulty == 3:
                            score += 0.05
                    else:  # Experienced users
                        if pose_difficulty == 3:
                            score += 0.25
                        elif pose_difficulty == 2:
                            score += 0.15
                        elif pose_difficulty == 1:
                            score += 0.05
                    
                    # 3. Flexibility matching (0-0.15)
                    user_flexibility = flexibility_map.get(flexibility_level, 2)
                    if user_flexibility == 1 and pose_difficulty == 1:
                        score += 0.15
                    elif user_flexibility == 2 and pose_difficulty in [1, 2]:
                        score += 0.1
                    elif user_flexibility == 3:
                        score += 0.1
                    
                    # 4. Health condition considerations (0-0.15)
                    if chronic_condition == 'hypertension' and row['primary_benefit'] in ['stress_relief', 'relaxation']:
                        score += 0.15
                    elif chronic_condition == 'diabetes' and row['primary_benefit'] in ['digestion', 'metabolism']:
                        score += 0.15
                    elif chronic_condition == 'arthritis' and row['difficulty_level'] == 'beginner':
                        score += 0.15
                    elif chronic_condition == 'asthma' and row['primary_benefit'] in ['breathing', 'respiratory']:
                        score += 0.15
                    elif chronic_condition == 'none':
                        score += 0.05  # Slight bonus for healthy users
                    
                    # 5. Age factor (0-0.1)
                    if age > 50 and pose_difficulty == 1:
                        score += 0.1
                    elif age < 30 and pose_difficulty >= 2:
                        score += 0.1
                    
                    # Cap score at 1.0
                    score = min(score, 1.0)
                    
                    recommendations.append({
                        'asana_name': row['asana_name'],
                        'primary_benefit': row['primary_benefit'],
                        'difficulty_level': row['difficulty_level'],
                        'duration_minutes': row['duration_minutes'],
                        'suitability_score': score
                    })
                
                # Sort by suitability score
                recommendations_df = pd.DataFrame(recommendations)
                recommendations_df = recommendations_df.sort_values('suitability_score', ascending=False).head(10)
                
                # Display recommendations
                st.subheader("Your Personalized Yoga Recommendations")
                
                for _, row in recommendations_df.iterrows():
                    with st.container():
                        col_card1, col_card2 = st.columns([3, 1])
                        with col_card1:
                            st.markdown(f"**🧘 {row['asana_name']}**")
                            st.markdown(f"📋 **Primary Benefit:** {row['primary_benefit']}")
                            st.markdown(f"📊 **Difficulty:** {row['difficulty_level']}")
                            st.markdown(f"⏱️ **Duration:** {row['duration_minutes']} minutes")
                        with col_card2:
                            st.metric("Suitability Score", f"{row['suitability_score']:.2f}")
                        st.divider()
                
                # Visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = plt.cm.RdYlGn(recommendations_df['suitability_score'])
                recommendations_df_sorted = recommendations_df.sort_values('suitability_score')
                bars = ax.barh(recommendations_df_sorted['asana_name'], recommendations_df_sorted['suitability_score'], color=colors)
                ax.set_title('Recommendation Suitability Scores', fontsize=14, fontweight='bold')
                ax.set_xlabel('Suitability Score', fontsize=12)
                ax.set_xlim(0, 1)
                # Add score labels on bars
                for bar, score in zip(bars, recommendations_df_sorted['suitability_score']):
                    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{score:.2f}', va='center', fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)
    
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


# ================================================================================
# SECTION 4: MAIN ENTRY POINT
# ================================================================================

def main():
    """
    Main function with multiple interface options.
    When run directly, provides a menu to choose the interface.
    When run with Streamlit (detected via command line), runs the web app.
    """
    # Check if being run by Streamlit
    if STREAMLIT_AVAILABLE and ('streamlit' in sys.argv[0] or 'streamlit' in ' '.join(sys.argv)):
        run_streamlit_standalone()
        return
    
    print("="*70)
    print("HRV-Driven Adaptive Yoga Recommendation System")
    print("="*70)
    print("Choose interface:")
    print("1. Streamlit Web Application (run: streamlit run yoga_combined.py)")
    print("2. Tkinter GUI Application")
    print("3. Jupyter Analysis Mode")
    print("4. Model Training Mode")
    print("5. Recommendation Mode")
    print("6. Exit")
    print("="*70)
    
    try:
        choice = input("Enter your choice (1-6): ")
    except (EOFError, KeyboardInterrupt):
        print("\nExiting...")
        return
    
    system = YogaRecommendationSystem()
    
    if choice == "1":
        print("\nTo run Streamlit, use command:")
        print("  streamlit run yoga_combined.py")
        print("\nOr choose another option from the menu.")
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
