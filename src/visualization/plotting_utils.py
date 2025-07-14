"""
Visualization utilities for employee sentiment analysis
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def setup_plotting_style():
    """
    Set up consistent plotting style
    """
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12

def plot_sentiment_distribution(df, sentiment_col, title="Sentiment Distribution"):
    """
    Plot distribution of sentiment scores
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram of sentiment scores
    ax1.hist(df[sentiment_col], bins=30, alpha=0.7, edgecolor='black')
    ax1.set_title(f'{title} - Scores')
    ax1.set_xlabel('Sentiment Score')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Count plot of sentiment labels
    if 'sentiment_label' in df.columns:
        sentiment_counts = df['sentiment_label'].value_counts()
        colors = ['green' if x == 'Positive' else 'red' if x == 'Negative' else 'gray' 
                 for x in sentiment_counts.index]
        ax2.bar(sentiment_counts.index, sentiment_counts.values, color=colors, alpha=0.7)
        ax2.set_title(f'{title} - Labels')
        ax2.set_xlabel('Sentiment Label')
        ax2.set_ylabel('Count')
        
        # Add value labels on bars
        for i, (label, count) in enumerate(sentiment_counts.items()):
            ax2.text(i, count + max(sentiment_counts.values) * 0.01, 
                    f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def plot_employee_ranking(df, score_col, top_n=10, title="Top Employee Rankings"):
    """
    Plot top N employees by score
    """
    if 'monthly_score' in df.columns:
        # Use monthly scores and group by employee
        employee_scores = df.groupby('from')['monthly_score'].max().reset_index()
        employee_scores = employee_scores.sort_values('monthly_score', ascending=False)
        top_employees = employee_scores.head(top_n)
        score_column = 'monthly_score'
    else:
        top_employees = df.nlargest(top_n, score_col)
        score_column = score_col
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(top_employees)), top_employees[score_column])
    
    # Create labels (truncate long email addresses)
    if 'from' in top_employees.columns:
        labels = [email.split('@')[0] if '@' in email else email 
                 for email in top_employees['from']]
    else:
        labels = [f"Employee {i+1}" for i in range(len(top_employees))]
    
    plt.yticks(range(len(top_employees)), labels)
    plt.xlabel('Employee Score')
    plt.title(title)
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}', ha='left', va='center')
    
    plt.tight_layout()
    return plt.gcf()

def plot_monthly_rankings(df, month_rankings, title="Monthly Employee Rankings"):
    """
    Plot top 3 positive and negative employees for a specific month
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Top 3 Positive
    top_pos = month_rankings['top_positive']
    if not top_pos.empty:
        # Create employee labels
        pos_labels = [email.split('@')[0] if '@' in email else email 
                     for email in top_pos['from']]
        ax1.barh(range(len(top_pos)), top_pos['monthly_score'], color='green', alpha=0.7)
        ax1.set_yticks(range(len(top_pos)))
        ax1.set_yticklabels(pos_labels)
        ax1.set_xlabel('Monthly Score')
        ax1.set_title('Top 3 Positive Employees')
        ax1.invert_yaxis()
        
        # Add value labels
        for i, score in enumerate(top_pos['monthly_score']):
            ax1.text(score + 0.1, i, f'{score}', va='center')
    
    # Top 3 Negative
    top_neg = month_rankings['top_negative']
    if not top_neg.empty:
        # Create employee labels
        neg_labels = [email.split('@')[0] if '@' in email else email 
                     for email in top_neg['from']]
        ax2.barh(range(len(top_neg)), top_neg['monthly_score'], color='red', alpha=0.7)
        ax2.set_yticks(range(len(top_neg)))
        ax2.set_yticklabels(neg_labels)
        ax2.set_xlabel('Monthly Score')
        ax2.set_title('Top 3 Negative Employees')
        ax2.invert_yaxis()
        
        # Add value labels
        for i, score in enumerate(top_neg['monthly_score']):
            ax2.text(score - 0.1, i, f'{score}', va='center', ha='right')
    
    plt.suptitle(f'{title} - {month_rankings["month"]}')
    plt.tight_layout()
    return fig

def create_wordcloud(text_data, title="Word Cloud"):
    """
    Create word cloud from text data
    """
    # Combine all text
    if isinstance(text_data, pd.Series):
        text = ' '.join(text_data.dropna().astype(str))
    else:
        text = ' '.join(text_data)
    
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         max_words=100).generate(text)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

def plot_flight_risk_analysis(df, risk_col, title="Flight Risk Analysis"):
    """
    Plot flight risk distribution
    """
    plt.figure(figsize=(10, 6))
    risk_counts = df[risk_col].value_counts()
    colors = ['green' if x == 'Low' else 'orange' if x == 'Medium' else 'red' 
              for x in risk_counts.index]
    
    bars = plt.bar(risk_counts.index, risk_counts.values, color=colors, alpha=0.7)
    plt.title(title)
    plt.xlabel('Risk Level')
    plt.ylabel('Number of Employees')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt.gcf()

def plot_time_series_analysis(df, date_col='date', sentiment_col='sentiment_score', 
                             title="Sentiment Trends Over Time"):
    """
    Plot sentiment trends over time
    """
    # Convert date column to datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Create monthly aggregations
    monthly_sentiment = df.groupby(df[date_col].dt.to_period('M')).agg({
        sentiment_col: ['mean', 'count'],
        'sentiment_label': lambda x: (x == 'Positive').sum()
    }).reset_index()
    
    # Flatten column names
    monthly_sentiment.columns = [date_col, 'avg_sentiment', 'message_count', 'positive_count']
    monthly_sentiment[date_col] = monthly_sentiment[date_col].dt.to_timestamp()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Average sentiment over time
    ax1.plot(monthly_sentiment[date_col], monthly_sentiment['avg_sentiment'], 
             marker='o', linewidth=2, markersize=6)
    ax1.set_title('Average Sentiment Score Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Average Sentiment Score')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Message count over time (change to line chart)
    ax2.plot(monthly_sentiment[date_col], monthly_sentiment['message_count'],
             marker='o', color='skyblue', linewidth=2)
    ax2.set_title('Message Volume Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Number of Messages')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_eda_summary(df, text_col='Subject', sentiment_col='sentiment_score'):
    """
    Create comprehensive EDA summary plots
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # 1. Data structure overview
    ax1.text(0.1, 0.9, f"Dataset Overview:", fontsize=16, weight='bold', transform=ax1.transAxes)
    ax1.text(0.1, 0.8, f"Total Records: {len(df):,}", fontsize=12, transform=ax1.transAxes)
    ax1.text(0.1, 0.7, f"Unique Employees: {df['from'].nunique():,}", fontsize=12, transform=ax1.transAxes)
    ax1.text(0.1, 0.6, f"Date Range: {df['date'].min()} to {df['date'].max()}", fontsize=12, transform=ax1.transAxes)
    ax1.text(0.1, 0.5, f"Missing Values: {df.isnull().sum().sum()}", fontsize=12, transform=ax1.transAxes)
    
    # Data types
    ax1.text(0.1, 0.3, "Data Types:", fontsize=14, weight='bold', transform=ax1.transAxes)
    for i, (col, dtype) in enumerate(df.dtypes.items()):
        ax1.text(0.1, 0.25 - i*0.05, f"{col}: {dtype}", fontsize=10, transform=ax1.transAxes)
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('Dataset Structure', fontsize=16, weight='bold')
    
    # 2. Sentiment distribution
    if 'sentiment_label' in df.columns:
        sentiment_counts = df['sentiment_label'].value_counts()
        colors = ['green' if x == 'Positive' else 'red' if x == 'Negative' else 'gray' 
                 for x in sentiment_counts.index]
        ax2.pie(sentiment_counts.values, labels=sentiment_counts.index, colors=colors, 
                autopct='%1.1f%%', startangle=90)
        ax2.set_title('Sentiment Distribution', fontsize=16, weight='bold')
    
    # 3. Message length distribution
    if text_col in df.columns:
        message_lengths = df[text_col].str.len()
        ax3.hist(message_lengths, bins=50, alpha=0.7, edgecolor='black')
        ax3.set_title('Message Length Distribution', fontsize=16, weight='bold')
        ax3.set_xlabel('Message Length (characters)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
    
    # 4. Top employees by message count
    top_senders = df['from'].value_counts().head(10)
    sender_labels = [email.split('@')[0] if '@' in email else email 
                    for email in top_senders.index]
    ax4.barh(range(len(top_senders)), top_senders.values)
    ax4.set_yticks(range(len(top_senders)))
    ax4.set_yticklabels(sender_labels)
    ax4.set_xlabel('Number of Messages')
    ax4.set_title('Top 10 Message Senders', fontsize=16, weight='bold')
    ax4.invert_yaxis()
    
    # Add value labels
    for i, count in enumerate(top_senders.values):
        ax4.text(count + max(top_senders.values) * 0.01, i, f'{count}', va='center')
    
    plt.tight_layout()
    return fig 

def plot_all_monthly_rankings(df, monthly_rankings, output_dir='data/plots', months=None, title_prefix="Monthly Employee Rankings"):
    """
    Generate and save ranking graphs for all months (or a specified subset).
    Args:
        df: The full DataFrame (not used, but kept for API compatibility)
        monthly_rankings: dict of {month: rankings_dict} as produced by get_employee_rankings
        output_dir: Directory to save plots
        months: List of months to plot (as strings or Periods). If None, plot all months in monthly_rankings.
        title_prefix: Prefix for plot titles
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which months to plot
    if months is None:
        months_to_plot = sorted(monthly_rankings.keys())
    else:
        months_to_plot = months
    
    for month in months_to_plot:
        rankings = monthly_rankings[month]
        fig = plot_monthly_rankings(df, rankings, title=f"{title_prefix} - {month}")
        # Clean month string for filename
        month_str = str(month).replace('/', '-')
        filename = f"monthly_rankings_{month_str}.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved ranking plot for {month} to {filepath}") 