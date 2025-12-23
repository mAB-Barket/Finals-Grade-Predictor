"""
AI Student Performance Predictor - Model Training
==================================================
This script trains a Random Forest model to predict final exam scores.

Prerequisites:
- Run generate_data.py first to create 'student_data_advanced.csv'
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_SEED = 42

# Weightages (must sum to 1.0)
WEIGHTS = {
    'finals': 0.40,
    'mids': 0.25,
    'quizzes': 0.15,
    'assignments': 0.10,
    'presentation': 0.05,
    'internal': 0.05
}

# GPA Points for each grade
GPA_SCALE = {
    'A':  4.0,
    'A-': 3.7,
    'B+': 3.3,
    'B':  3.0,
    'B-': 2.7,
    'C+': 2.3,
    'C':  2.0,
    'C-': 1.7,
    'D+': 1.3,
    'D':  1.0,
    'F':  0.0
}


def assign_relative_grade(score, mean, std):
    """
    Assign grade based on relative position in class distribution.
    Using standard deviation-based cutoffs for relative grading.
    """
    z_score = (score - mean) / std
    
    if z_score >= 1.5:
        return 'A'      # Top ~7%
    elif z_score >= 1.0:
        return 'A-'     # Next ~9%
    elif z_score >= 0.5:
        return 'B+'     # Next ~15%
    elif z_score >= 0.0:
        return 'B'      # Next ~19%
    elif z_score >= -0.5:
        return 'B-'     # Next ~19%
    elif z_score >= -1.0:
        return 'C+'     # Next ~15%
    elif z_score >= -1.5:
        return 'C'      # Next ~9%
    elif z_score >= -2.0:
        return 'C-'     # Next ~4%
    elif z_score >= -2.5:
        return 'D+' if z_score >= -2.25 else 'D'  # Next ~2%
    else:
        return 'F'      # Bottom ~1%


DATA_FILE = 'student_data_advanced.csv'
MODEL_FILE = 'grade_predictor_model.pkl'


def train_model():
    """Train the Random Forest model for grade prediction."""
    
    # Check if data file exists
    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] '{DATA_FILE}' not found!")
        print("   Please run 'python generate_data.py' first to generate the dataset.")
        return None
    
    # Load data
    print("=" * 60)
    print("STEP 1: Loading Dataset")
    print("=" * 60)
    
    df = pd.read_csv(DATA_FILE)
    print(f"[OK] Loaded {len(df)} records from '{DATA_FILE}'")
    
    # Calculate class statistics for relative grading
    mean_score = df['Total_Weighted'].mean()
    std_score = df['Total_Weighted'].std()
    print(f"   Class Mean: {mean_score:.2f}")
    print(f"   Class Std:  {std_score:.2f}")
    
    # Prepare features
    print("\n" + "=" * 60)
    print("STEP 2: Training the Model")
    print("=" * 60)
    
    feature_columns = [
        'Attendance', 'Hours_Studied', 'Prev_GPA', 'Quiz_Avg', 
        'Midterm_Score', 'Assignment_Avg', 'Presentation_Marks', 'Internal_Marks'
    ]
    
    X = df[feature_columns]
    y = df['Final_Score']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size:     {len(X_test)}")
    
    # Hyperparameter tuning
    print("\n[...] Performing hyperparameter tuning...")
    param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [None, 10, 15, 20, 25],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 2, 3],
        'max_features': ['sqrt', 'log2', None]
    }
    
    rf = RandomForestRegressor(random_state=RANDOM_SEED)
    search = RandomizedSearchCV(
        rf, 
        param_distributions=param_dist, 
        n_iter=20, 
        scoring='neg_mean_absolute_error',
        cv=5, 
        random_state=RANDOM_SEED, 
        n_jobs=-1,
        verbose=0
    )
    
    search.fit(X_train, y_train)
    
    print(f"[OK] Best Parameters: {search.best_params_}")
    
    # Get best model
    best_model = search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n[STATS] Model Performance:")
    print(f"   Mean Absolute Error: {mae:.2f} marks")
    print(f"   R² Score:            {r2:.4f}")
    
    # Feature importance
    print(f"\n[INFO] Feature Importance:")
    importances = best_model.feature_importances_
    for name, imp in sorted(zip(feature_columns, importances), key=lambda x: -x[1]):
        bar = '#' * int(imp * 50)
        print(f"   {name:20s}: {imp:.3f} {bar}")
    
    # Save model
    print("\n" + "=" * 60)
    print("STEP 3: Saving the Model")
    print("=" * 60)
    
    joblib.dump(best_model, MODEL_FILE)
    print(f"[OK] Model saved to '{MODEL_FILE}'")
    
    # Quick verification
    print("\n" + "=" * 60)
    print("STEP 4: Quick Verification")
    print("=" * 60)
    
    test_cases = [
        # Good student
        {'Attendance': 95, 'Hours_Studied': 15, 'Prev_GPA': 3.8, 'Quiz_Avg': 85,
         'Midterm_Score': 88, 'Assignment_Avg': 90, 'Presentation_Marks': 85, 'Internal_Marks': 88},
        # Average student
        {'Attendance': 80, 'Hours_Studied': 8, 'Prev_GPA': 3.0, 'Quiz_Avg': 70,
         'Midterm_Score': 65, 'Assignment_Avg': 72, 'Presentation_Marks': 68, 'Internal_Marks': 70},
        # Struggling student
        {'Attendance': 60, 'Hours_Studied': 3, 'Prev_GPA': 2.2, 'Quiz_Avg': 50,
         'Midterm_Score': 45, 'Assignment_Avg': 55, 'Presentation_Marks': 50, 'Internal_Marks': 52},
    ]
    
    print("\nSample Predictions:")
    for i, case in enumerate(test_cases, 1):
        test_df = pd.DataFrame([case])
        pred = best_model.predict(test_df)[0]
        
        # Calculate what the total weighted score would be
        total = (
            pred * WEIGHTS['finals'] +
            case['Midterm_Score'] * WEIGHTS['mids'] +
            case['Quiz_Avg'] * WEIGHTS['quizzes'] +
            case['Assignment_Avg'] * WEIGHTS['assignments'] +
            case['Presentation_Marks'] * WEIGHTS['presentation'] +
            case['Internal_Marks'] * WEIGHTS['internal']
        )
        grade = assign_relative_grade(total, mean_score, std_score)
        
        print(f"\n  Student {i}: Predicted Final = {pred:.1f}/100")
        print(f"             Total Weighted   = {total:.1f}/100")
        print(f"             Expected Grade   = {grade} ({GPA_SCALE[grade]} GPA)")
    
    return best_model


if __name__ == "__main__":
    model = train_model()
    
    if model:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print("\nYou can now run: streamlit run app.py")
