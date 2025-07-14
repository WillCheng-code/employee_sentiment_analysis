"""
Utility functions for sentiment analysis
"""
import pandas as pd
import numpy as np
from textblob import TextBlob
import re
from datetime import datetime, timedelta

# LLM-based sentiment analysis imports
try:
    from transformers.pipelines import pipeline
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    try:
        # Fallback to the standard import
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        import torch
        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        TRANSFORMERS_AVAILABLE = False
        print("Warning: transformers library not available. Install with: pip install transformers torch")

def clean_text(text):
    """
    Clean text data by removing special characters and extra whitespace
    """
    if pd.isna(text):
        return ""
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', '', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_sentiment_score(text):
    """
    Calculate sentiment score using TextBlob
    Returns polarity score between -1 (negative) and 1 (positive)
    """
    if pd.isna(text) or text == "":
        return 0

    blob = TextBlob(clean_text(text))
    sentiment = blob.sentiment
    polarity = getattr(sentiment, "polarity", 0)
    return polarity

def categorize_sentiment(score):
    """
    Categorize sentiment score into labels
    """
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"

def assign_message_score(sentiment_label):
    """
    Assign score to each message based on sentiment:
    Positive Message: +1
    Negative Message: -1
    Neutral Message: 0
    """
    if sentiment_label == "Positive":
        return 1
    elif sentiment_label == "Negative":
        return -1
    else:
        return 0

def calculate_monthly_employee_scores(df, employee_col='from', date_col='date', sentiment_col='sentiment_label'):
    """
    Calculate monthly sentiment scores for each employee based on their messages.
    Score resets at the beginning of each new month.
    """
    # Convert date column to datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Create year-month column for grouping
    df['year_month'] = df[date_col].dt.to_period('M')
    
    # Assign message scores
    df['message_score'] = df[sentiment_col].apply(assign_message_score)
    
    # Calculate monthly scores for each employee
    monthly_scores = df.groupby([employee_col, 'year_month'])['message_score'].sum().reset_index()
    monthly_scores.columns = [employee_col, 'year_month', 'monthly_score']
    
    # Merge back with original dataframe
    df = df.merge(monthly_scores, on=[employee_col, 'year_month'], how='left')
    
    return df

def get_employee_rankings(df, employee_col='from', score_col='monthly_score', year_month=None):
    """
    Generate ranked lists of employees based on their monthly sentiment scores.
    Returns top 3 positive and top 3 negative employees for a given month.
    """
    if year_month is not None:
        # Filter for specific month
        month_data = df[df['year_month'] == year_month]
    else:
        # Use the most recent month
        latest_month = df['year_month'].max()
        month_data = df[df['year_month'] == latest_month]
    
    # Get unique employee scores for the month
    employee_scores = month_data.groupby(employee_col)[score_col].first().reset_index()
    
    # Sort by score (descending for positive, ascending for negative)
    employee_scores_sorted = employee_scores.sort_values([score_col, employee_col], 
                                                        ascending=[False, True])
    
    # Get top 3 positive (highest scores)
    top_positive = employee_scores_sorted.head(3)
    
    # Get top 3 negative (lowest scores)
    top_negative = employee_scores_sorted.tail(3).sort_values([score_col, employee_col], 
                                                             ascending=[True, True])
    
    return {
        'top_positive': top_positive,
        'top_negative': top_negative,
        'month': year_month if year_month else latest_month
    }

def calculate_employee_score(df, sentiment_col, weight=0.7):
    """
    Calculate overall employee score based on sentiment and other factors
    This is kept for backward compatibility but the new monthly scoring should be used
    """
    # Normalize sentiment scores to 0-100 scale
    sentiment_scores = (df[sentiment_col] + 1) * 50
    
    # You can add other factors here (e.g., performance metrics)
    # For now, using only sentiment
    employee_scores = sentiment_scores * weight
    
    return employee_scores

def get_llm_sentiment_analyzer():
    """
    Initialize and return an LLM-based sentiment analyzer
    Uses a pre-trained model optimized for sentiment analysis
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is required for LLM sentiment analysis")
    
    # Use a robust sentiment analysis model
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    try:
        # Initialize the sentiment analysis pipeline
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1  # Use GPU if available
        )
        print(f"✓ LLM sentiment analyzer initialized: {model_name}")
        return sentiment_analyzer
    except Exception as e:
        print(f"Error loading {model_name}, falling back to default model")
        # Fallback to a more basic but reliable model
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1
        )
        print("✓ LLM sentiment analyzer initialized with fallback model")
        return sentiment_analyzer

# Global variable to cache the analyzer
_llm_analyzer = None

def get_llm_sentiment_score(text, analyzer=None):
    """
    Calculate sentiment score using LLM (Large Language Model)
    Returns polarity score between -1 (negative) and 1 (positive)
    """
    global _llm_analyzer
    
    if pd.isna(text) or text == "":
        return 0
    
    if not TRANSFORMERS_AVAILABLE:
        print("Warning: Falling back to TextBlob sentiment analysis")
        return get_sentiment_score(text)
    
    if analyzer is None:
        if _llm_analyzer is None:
            _llm_analyzer = get_llm_sentiment_analyzer()
        analyzer = _llm_analyzer
    
    try:
        # Clean and truncate text for the model
        clean_text_input = clean_text(text)
        # Truncate to avoid token limits (most models have 512 token limit)
        if len(clean_text_input) > 500:
            clean_text_input = clean_text_input[:500]
        
        # Get prediction from the model
        results = analyzer(clean_text_input)
        
        # Process results based on model output format
        if isinstance(results[0], list):
            # Model returns all scores
            scores = results[0]
        else:
            scores = results
        
        # Convert to standardized format
        sentiment_score = 0
        
        for score_item in scores:
            label = score_item['label'].lower()  # Convert to lowercase for consistency
            confidence = score_item['score']
            
            if label in ['positive', 'pos']:
                sentiment_score += confidence
            elif label in ['negative', 'neg']:
                sentiment_score -= confidence
            # NEUTRAL contributes 0 to the score
        
        # Ensure score is between -1 and 1
        sentiment_score = max(-1, min(1, sentiment_score))
        
        return sentiment_score
        
    except Exception as e:
        print(f"Error in LLM sentiment analysis: {e}")
        # Fallback to TextBlob
        return get_sentiment_score(text)

def analyze_sentiment_batch_llm(texts, batch_size=32):
    """
    Analyze sentiment for multiple texts using LLM in batches for efficiency
    """
    global _llm_analyzer
    
    if not TRANSFORMERS_AVAILABLE:
        print("Warning: Falling back to TextBlob for batch processing")
        return [get_sentiment_score(text) for text in texts]
    
    if _llm_analyzer is None:
        _llm_analyzer = get_llm_sentiment_analyzer()
    
    results = []
    
    # Process in batches to avoid memory issues
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_results = []
        
        for text in batch:
            score = get_llm_sentiment_score(text, _llm_analyzer)
            batch_results.append(score)
        
        results.extend(batch_results)
        
        # Progress indicator
        if i % (batch_size * 10) == 0:
            print(f"Processed {i + len(batch)}/{len(texts)} texts...")
    
    return results 