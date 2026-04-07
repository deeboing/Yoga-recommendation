"""
NUMPY 2.0 MULTIPROCESSING FIX
Copy this code into your Jupyter cell to fix the cross_val_score issue
"""

import os
import warnings
from sklearn.model_selection import cross_val_score

# Fix NumPy 2.0 multiprocessing compatibility
os.environ['JOBLIB_START_METHOD'] = 'threading'
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

def safe_cross_val_score(model, X, y, cv=5, scoring='accuracy'):
    """Safe cross-validation that works with NumPy 2.0"""
    # Use single process to avoid multiprocessing issues
    return cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)

# Now use this instead of cross_val_score:
# cv_acc = safe_cross_val_score(rf, X, y, cv=5, scoring='accuracy')
# print(f'5-Fold CV Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}')
