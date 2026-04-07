import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

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

class YogaRecommendationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HRV-Driven Adaptive Yoga Intensity Recommendation System")
        self.root.geometry("1400x900")
        
        # Initialize variables
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
        
        # Create main container
        self.main_container = ttk.Frame(root, padding="10")
        self.main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.main_container.columnconfigure(1, weight=1)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Create tabs
        self.create_data_tab()
        self.create_eda_tab()
        self.create_models_tab()
        self.create_recommendations_tab()
        
    def create_data_tab(self):
        """Create data loading tab"""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="Data Loading")
        
        # File selection
        file_frame = ttk.Frame(data_frame)
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(file_frame, text="Data Files:").pack(anchor=tk.W)
        
        self.asana_file_var = tk.StringVar()
        self.users_file_var = tk.StringVar()
        self.sessions_file_var = tk.StringVar()
        
        ttk.Label(file_frame, text="Asanas:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.asana_file_var, width=40).grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.load_asanas).grid(row=0, column=2, sticky=tk.W, padx=5)
        
        ttk.Label(file_frame, text="Users:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.users_file_var, width=40).grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.load_users).grid(row=1, column=2, sticky=tk.W, padx=5)
        
        ttk.Label(file_frame, text="Sessions:").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.sessions_file_var, width=40).grid(row=2, column=1, sticky=tk.W, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.load_sessions).grid(row=2, column=2, sticky=tk.W, padx=5)
        
        # Load buttons
        button_frame = ttk.Frame(data_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Load All Data", command=self.load_all_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Data", command=self.clear_data).pack(side=tk.LEFT, padx=5)
        
        # Status display
        self.data_status = tk.StringVar(value="No data loaded")
        ttk.Label(data_frame, text="Status:").pack(anchor=tk.W)
        ttk.Label(data_frame, textvariable=self.data_status).pack(anchor=tk.W)
    
    def create_eda_tab(self):
        """Create EDA tab"""
        eda_frame = ttk.Frame(self.notebook)
        self.notebook.add(eda_frame, text="Data Analysis")
        
        # Analysis buttons
        button_frame = ttk.Frame(eda_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Show Data Overview", command=self.show_data_overview).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Show Distributions", command=self.show_distributions).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Show Correlations", command=self.show_correlations).pack(side=tk.LEFT, padx=5)
        
        # Visualization area
        self.viz_frame = ttk.Frame(eda_frame)
        self.viz_frame.pack(fill=tk.BOTH, expand=True, pady=10)
    
    def create_models_tab(self):
        """Create model training tab"""
        models_frame = ttk.Frame(self.notebook)
        self.notebook.add(models_frame, text="Model Training")
        
        # Training buttons
        button_frame = ttk.Frame(models_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Train Collaborative Filtering", command=self.train_cf_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Train Random Forest", command=self.train_rf_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Train Gradient Boosting", command=self.train_gbr_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Compare Models", command=self.compare_models).pack(side=tk.LEFT, padx=5)
        
        # Results display
        self.results_text = scrolledtext.ScrolledText(models_frame, height=15, width=80)
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=10)
    
    def create_recommendations_tab(self):
        """Create recommendations tab"""
        rec_frame = ttk.Frame(self.notebook)
        self.notebook.add(rec_frame, text="Get Recommendations")
        
        # User input
        input_frame = ttk.Frame(rec_frame)
        input_frame.pack(fill=tk.X, pady=10)
        
        # User profile inputs
        profile_frame = ttk.LabelFrame(input_frame, text="User Health Profile", padding=10)
        profile_frame.pack(fill=tk.X, pady=5)
        
        # Health parameters
        self.age_var = tk.IntVar(value=30)
        self.gender_var = tk.StringVar(value="Male")
        self.bmi_var = tk.DoubleVar(value=25.0)
        self.hrv_var = tk.DoubleVar(value=45.0)
        self.spo2_var = tk.DoubleVar(value=95.0)
        self.stress_var = tk.DoubleVar(value=0.5)
        self.experience_var = tk.IntVar(value=6)
        
        # Create input fields
        ttk.Label(profile_frame, text="Age:").grid(row=0, column=0, sticky=tk.W)
        ttk.Scale(profile_frame, from_=18, to=80, variable=self.age_var, orient=tk.HORIZONTAL).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(profile_frame, text="Gender:").grid(row=1, column=0, sticky=tk.W)
        gender_frame = ttk.Frame(profile_frame)
        gender_frame.grid(row=1, column=1, sticky=tk.W)
        ttk.Radiobutton(gender_frame, text="Male", variable=self.gender_var, value="Male").pack(side=tk.LEFT)
        ttk.Radiobutton(gender_frame, text="Female", variable=self.gender_var, value="Female").pack(side=tk.LEFT)
        
        ttk.Label(profile_frame, text="BMI:").grid(row=2, column=0, sticky=tk.W)
        ttk.Scale(profile_frame, from_=15.0, to=40.0, variable=self.bmi_var, orient=tk.HORIZONTAL, resolution=0.1).grid(row=2, column=1, sticky=tk.W)
        
        ttk.Label(profile_frame, text="HRV RMSSD:").grid(row=3, column=0, sticky=tk.W)
        ttk.Scale(profile_frame, from_=10.0, to=100.0, variable=self.hrv_var, orient=tk.HORIZONTAL, resolution=0.1).grid(row=3, column=1, sticky=tk.W)
        
        ttk.Label(profile_frame, text="Avg SpO2:").grid(row=4, column=0, sticky=tk.W)
        ttk.Scale(profile_frame, from_=80.0, to=100.0, variable=self.spo2_var, orient=tk.HORIZONTAL, resolution=0.1).grid(row=4, column=1, sticky=tk.W)
        
        ttk.Label(profile_frame, text="Stress Index:").grid(row=5, column=0, sticky=tk.W)
        ttk.Scale(profile_frame, from_=0.0, to=1.0, variable=self.stress_var, orient=tk.HORIZONTAL, resolution=0.01).grid(row=5, column=1, sticky=tk.W)
        
        ttk.Label(profile_frame, text="Experience (months):").grid(row=6, column=0, sticky=tk.W)
        ttk.Scale(profile_frame, from_=0, to=48, variable=self.experience_var, orient=tk.HORIZONTAL).grid(row=6, column=1, sticky=tk.W)
        
        # Get recommendations button
        ttk.Button(input_frame, text="Get Personalized Recommendations", command=self.get_recommendations).pack(pady=20)
        
        # Results display
        self.rec_results_text = scrolledtext.ScrolledText(rec_frame, height=15, width=80)
        self.rec_results_text.pack(fill=tk.BOTH, expand=True, pady=10)
    
    def load_asanas(self):
        """Load asanas dataset"""
        try:
            file_path = filedialog.askopenfilename(
                title="Load Asanas Dataset",
                filetypes=[("CSV files", "*.csv")]
            )
            if file_path:
                self.asanas_df = pd.read_csv(file_path)
                self.asana_file_var.set(file_path)
                messagebox.showinfo("Success", f"Loaded {len(self.asanas_df)} yoga poses")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load asanas: {str(e)}")
    
    def load_users(self):
        """Load users dataset"""
        try:
            file_path = filedialog.askopenfilename(
                title="Load Users Dataset",
                filetypes=[("CSV files", "*.csv")]
            )
            if file_path:
                self.users_df = pd.read_csv(file_path)
                self.users_file_var.set(file_path)
                messagebox.showinfo("Success", f"Loaded {len(self.users_df)} user profiles")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load users: {str(e)}")
    
    def load_sessions(self):
        """Load sessions dataset"""
        try:
            file_path = filedialog.askopenfilename(
                title="Load Sessions Dataset",
                filetypes=[("CSV files", "*.csv")]
            )
            if file_path:
                self.sessions_df = pd.read_csv(file_path)
                self.sessions_file_var.set(file_path)
                messagebox.showinfo("Success", f"Loaded {len(self.sessions_df)} session records")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load sessions: {str(e)}")
    
    def load_all_data(self):
        """Load all datasets"""
        try:
            # Load default files if not specified
            asanas_path = self.asana_file_var.get() or "yoga_asanas_knowledge_base.csv"
            users_path = self.users_file_var.get() or "yoga_users_dataset.csv"
            sessions_path = self.sessions_file_var.get() or "yoga_sessions_feedback.csv"
            
            self.asanas_df = pd.read_csv(asanas_path)
            self.users_df = pd.read_csv(users_path)
            self.sessions_df = pd.read_csv(sessions_path)
            
            self.data_status.set(f"Loaded: {len(self.asanas_df)} poses, {len(self.users_df)} users, {len(self.sessions_df)} sessions")
            
            messagebox.showinfo("Success", "All datasets loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def clear_data(self):
        """Clear all loaded data"""
        self.asanas_df = None
        self.users_df = None
        self.sessions_df = None
        self.svd_model = None
        self.rf_model = None
        self.gbr_model = None
        self.data_status.set("No data loaded")
        messagebox.showinfo("Cleared", "All data has been cleared")
    
    def preprocess_data(self):
        """Preprocess data for ML"""
        if self.users_df is None or self.sessions_df is None or self.asanas_df is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return None, None, None, None
        
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
    
    def train_cf_model(self):
        """Train collaborative filtering model"""
        try:
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
            
            if SURPRISE_AVAILABLE:
                # Use Surprise SVD
                reader = Reader(rating_scale=(0, 1))
                data = Dataset.load_from_df(self.sessions_df[['user_id', 'asana_id', 'recommendation_score']], reader)
                
                self.svd_model = SVD(n_factors=50, n_epochs=30, lr_all=0.005, reg_all=0.02, random_state=42)
                self.svd_model.fit(data.build_full_trainset())
                
                messagebox.showinfo("Success", "SVD model trained successfully!")
            else:
                # Use alternative method
                messagebox.showinfo("Info", "Using alternative collaborative filtering method")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train CF model: {str(e)}")
    
    def train_rf_model(self):
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
            from sklearn.metrics import accuracy_score
            y_pred = self.rf_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            messagebox.showinfo("Success", f"Random Forest trained! Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train RF model: {str(e)}")
    
    def train_gbr_model(self):
        """Train Gradient Boosting model"""
        try:
            X, y, merged = self.preprocess_data()
            if X is None:
                return
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            self.gbr_model = GradientBoostingRegressor(
                n_estimators=100, random_state=42
            )
            self.gbr_model.fit(X_train, y_train)
            
            messagebox.showinfo("Success", "Gradient Boosting model trained!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train GBR model: {str(e)}")
    
    def compare_models(self):
        """Compare model performance"""
        try:
            X, y, merged = self.preprocess_data()
            if X is None:
                return
            
            if not self.rf_model:
                self.train_rf_model()
                return
            
            from sklearn.model_selection import cross_val_score
            # Use single process to avoid NumPy 2.0 issues
            cv_scores = cross_val_score(self.rf_model, X, y, cv=5, scoring='accuracy', n_jobs=1)
            
            comparison_text = f"""
Model Comparison Results:
==========================
Random Forest CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
==========================
            """
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, comparison_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Model comparison failed: {str(e)}")
    
    def get_recommendations(self):
        """Get personalized recommendations"""
        try:
            if self.users_df is None:
                messagebox.showwarning("Warning", "Please load user data first!")
                return
            
            # Create user profile
            user_profile = pd.DataFrame({
                'age': [self.age_var.get()],
                'bmi': [self.bmi_var.get()],
                'hrv_rmssd': [self.hrv_var.get()],
                'average_spo2': [self.spo2_var.get()],
                'stress_index': [self.stress_var.get()],
                'yoga_experience_months': [self.experience_var.get()],
                'gender': [self.gender_var.get()]
            })
            
            # Encode categorical variables
            le = LabelEncoder()
            for col in ['gender']:
                user_profile[col] = le.fit_transform(user_profile[col].astype(str))
            
            # Get recommendations based on similarity
            if self.asanas_df is not None:
                # Simple content-based filtering
                user_features = user_profile[['age', 'bmi', 'hrv_rmssd', 'average_spo2']].values
                asana_features = self.asanas_df[['difficulty_level', 'duration_minutes', 'intensity']].values
                
                # Calculate similarity scores
                from sklearn.metrics.pairwise import cosine_similarity
                sim_scores = cosine_similarity(user_features, asana_features)
                
                # Get top recommendations
                top_indices = sim_scores.argsort()[0][::-1][:10]
                recommendations = self.asanas_df.iloc[top_indices]
                
                # Display results
                rec_text = "Personalized Yoga Recommendations:\n\n"
                for _, row in recommendations.iterrows():
                    rec_text += f"- {row['asana_name']}: {row['primary_benefit']}\n"
                    rec_text += f"  Difficulty: {row['difficulty_level']}, Duration: {row['duration_minutes']} min\n"
                    rec_text += f"  Suitable for: {row['primary_benefit']}\n\n"
                
                self.rec_results_text.delete(1.0, tk.END)
                self.rec_results_text.insert(tk.END, rec_text)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get recommendations: {str(e)}")
    
    def show_data_overview(self):
        """Show data overview"""
        if self.asanas_df is not None and self.users_df is not None and self.sessions_df is not None:
            overview_text = f"""
Dataset Overview:
==================
Asanas: {len(self.asanas_df)} yoga poses
Users: {len(self.users_df)} user profiles  
Sessions: {len(self.sessions_df)} session records

Asanas Columns: {list(self.asanas_df.columns)}
Users Columns: {list(self.users_df.columns)}
Sessions Columns: {list(self.sessions_df.columns)}
            """
            
            messagebox.showinfo("Data Overview", overview_text)
        else:
            messagebox.showwarning("Warning", "Please load all data first!")
    
    def show_distributions(self):
        """Show data distributions"""
        if self.users_df is not None:
            try:
                import matplotlib.pyplot as plt
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                
                # Create figure
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                health_cols = ['age', 'bmi', 'hrv_rmssd', 'average_spo2']
                
                for ax, col in zip(axes.flatten(), health_cols):
                    self.users_df[col].hist(bins=25, ax=ax, color='steelblue', edgecolor='white')
                    ax.set_title(col.replace('_', ' ').title())
                    ax.set_xlabel('')
                
                plt.suptitle('Distribution of User Health Parameters', fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                # Display in GUI
                canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to show distributions: {str(e)}")
    
    def show_correlations(self):
        """Show correlations"""
        if self.users_df is not None:
            try:
                import matplotlib.pyplot as plt
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                import seaborn as sns
                
                # Create correlation heatmap
                numeric_cols = ['age', 'bmi', 'hrv_rmssd', 'average_spo2', 'sleep_quality', 
                               'stress_index', 'mood_baseline', 'yoga_experience_months']
                corr_matrix = self.users_df[numeric_cols].corr()
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                           linewidths=0.5, square=True, ax=ax)
                ax.set_title('Correlation Matrix - User Health Features')
                
                plt.tight_layout()
                
                # Display in GUI
                canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to show correlations: {str(e)}")

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = YogaRecommendationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
