# Simple fix for scikit-learn multiprocessing issue with NumPy 2.0
import os
import warnings

# Set environment variables to fix multiprocessing issues
os.environ['JOBLIB_START_METHOD'] = 'threading'
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

print("NumPy 2.0 multiprocessing compatibility fixes applied")
print("You can now run your Streamlit application with:")
print("streamlit run app_streamlit.py")
