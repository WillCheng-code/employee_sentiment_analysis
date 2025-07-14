"""
Sentiment analysis and predictive modeling for employee data
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta

class EmployeeSentimentModel:
    """
    Model for predicting employee sentiment and flight risk
    """
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def prepare_features(self, df, text_col, additional_features=None):
        """
        Prepare features for modeling
        """
        # Basic text features
        features = pd.DataFrame()
        
        # Text length
        features['text_length'] = df[text_col].str.len()
        
        # Word count
        features['word_count'] = df[text_col].str.split().str.len()
        
        # Sentiment score (if not already calculated)
        if 'sentiment_score' not in df.columns:
            from ..utils.sentiment_utils import get_sentiment_score
            features['sentiment_score'] = df[text_col].apply(get_sentiment_score)
        else:
            features['sentiment_score'] = df['sentiment_score']
        
        # Add additional features if provided
        if additional_features:
            for feature in additional_features:
                if feature in df.columns:
                    features[feature] = df[feature]
        
        return features
    
    def fit(self, X, y):
        """
        Fit the model
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Choose model
        if self.model_type == 'linear':
            self.model = LinearRegression()
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        if self.model is None:
            raise ValueError("Model type not recognized or model not initialized.")
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
    def predict(self, X):
        """
        Make predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.model is None:
            raise ValueError("Model has not been initialized. Please fit the model before predicting.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        """
        y_pred = self.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2
        }
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        if self.is_fitted:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type
            }, filepath)
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained model
        """
        model_data = joblib.load(filepath)
        instance = cls(model_type=model_data['model_type'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.is_fitted = True
        return instance

def identify_flight_risk(df, employee_col='from', date_col='date', sentiment_col='sentiment_label'):
    """
    Identify employees at flight risk based on the requirement:
    Flight risk = any employee who has sent 4 or more negative messages in any 30-day rolling period
    """
    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Filter for negative messages only
    negative_messages = df[df[sentiment_col] == 'Negative'].copy()
    
    # Sort by employee and date
    negative_messages = negative_messages.sort_values([employee_col, date_col])
    
    # Initialize flight risk column
    df['flight_risk'] = 'Low'
    
    # Dictionary to store flight risk employees
    flight_risk_employees = set()
    
    # For each employee, check rolling 30-day windows
    for employee in negative_messages[employee_col].unique():
        employee_messages = negative_messages[negative_messages[employee_col] == employee]
        
        if len(employee_messages) < 4:
            continue  # Can't have flight risk with less than 4 negative messages
        
        # Check each possible 30-day window
        for i in range(len(employee_messages)):
            start_date = employee_messages.iloc[i][date_col]
            end_date = start_date + timedelta(days=30)
            
            # Count negative messages in this 30-day window
            window_messages = employee_messages[
                (employee_messages[date_col] >= start_date) & 
                (employee_messages[date_col] <= end_date)
            ]
            
            if len(window_messages) >= 4:
                flight_risk_employees.add(employee)
                break  # Once identified as flight risk, no need to check further
    
    # Update flight risk status
    df.loc[df[employee_col].isin(flight_risk_employees), 'flight_risk'] = 'High'
    
    return df

def get_flight_risk_employees(df, employee_col='from'):
    """
    Extract list of employees identified as flight risk
    """
    flight_risk_employees = df[df['flight_risk'] == 'High'][employee_col].unique()
    return list(flight_risk_employees) 