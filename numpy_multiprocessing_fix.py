# Fix for NumPy 2.0 multiprocessing compatibility issues
import os
import warnings

def fix_numpy_multiprocessing():
    """
    Fix NumPy 2.0 multiprocessing compatibility issues
    This should be called before any cross_val_score with n_jobs=-1
    """
    # Set environment variables to fix multiprocessing issues
    os.environ['JOBLIB_START_METHOD'] = 'threading'
    os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '1'
    
    # Suppress warnings about multiprocessing
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    
    print("NumPy multiprocessing compatibility fixes applied")

def safe_cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=1):
    """
    Safe cross-validation that works with NumPy 2.0
    Uses single process to avoid multiprocessing issues
    """
    from sklearn.model_selection import cross_val_score
    
    # Force single process to avoid NumPy 2.0 multiprocessing issues
    if n_jobs == -1 or n_jobs > 1:
        print("Using single process (n_jobs=1) to avoid NumPy 2.0 multiprocessing issues")
        n_jobs = 1
    
    return cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)

# Apply the fix immediately
fix_numpy_multiprocessing()
