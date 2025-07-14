# Employee Sentiment Analysis

A comprehensive sentiment analysis system for employee communications using advanced NLP techniques and machine learning models.

## Executive Summary

This project analyzes employee sentiment patterns from email communications to identify engagement levels, rank employee performance, detect flight risk, and provide predictive insights for workforce management.

### Key Results Summary

**Dataset Overview:**
- **Total Messages Analyzed**: 2,191 messages
- **Employees**: 10 unique employees  
- **Analysis Period**: January 1, 2010 - December 31, 2011 (24 months)
- **Communication Channels**: Email (Subject + Body text)

**Sentiment Distribution:**
- **Neutral**: 1,672 messages (76.3%)
- **Positive**: 404 messages (18.4%)
- **Negative**: 115 messages (5.2%)

## 🏆 Top 3 Positive Employees

Based on comprehensive sentiment analysis across all communications:

1. **bobette.riner@ipgdirect.com**
   - Average Sentiment Score: 0.189
   - Positive Messages: 24.0%
   - Total Messages: 217
   - **Recognition**: Consistently positive communication, excellent team contributor

2. **eric.bass@enron.com**
   - Average Sentiment Score: 0.186
   - Positive Messages: 21.4%
   - Total Messages: 210
   - **Recognition**: Strong positive engagement, reliable performer

3. **johnny.palmer@enron.com**
   - Average Sentiment Score: 0.183
   - Positive Messages: 23.5%
   - Total Messages: 213
   - **Recognition**: High engagement levels, positive team influence

## ⚠️ Top 3 Employees Needing Support

Employees with lower sentiment scores requiring attention:

1. **kayne.coulter@enron.com**
   - Average Sentiment Score: 0.081
   - Negative Messages: 5.7%
   - Total Messages: 174
   - **Recommendation**: Targeted support and engagement initiatives

2. **patti.thompson@enron.com**
   - Average Sentiment Score: 0.080
   - Negative Messages: 6.2%
   - Total Messages: 225
   - **Recommendation**: Monitor closely, provide additional resources

3. **john.arnold@enron.com**
   - Average Sentiment Score: 0.062
   - Negative Messages: 7.4%
   - Total Messages: 256
   - **Recommendation**: Proactive intervention and support programs

## 🛡️ Flight Risk Analysis

**Current Status**: ✅ **No employees identified as flight risk**

**Flight Risk Criteria**: 4+ negative messages within any 30-day rolling period (irrespective of months)

**Analysis Results**: 
- **High Risk Employees**: 0 (0.0%)
- **Low Risk Employees**: All 10 employees (100.0%)
- **Team Stability**: Excellent - indicates stable, engaged workforce

## 📊 Key Insights and Recommendations

### 1. **Positive Work Environment**
- **94.8% of messages** are neutral or positive, indicating healthy workplace communication
- **Professional communication standards** maintained across all employees
- **No flight risk employees** demonstrates exceptional team stability

### 2. **Employee Recognition Program**
- **Immediate Action**: Implement recognition programs for top performers
- **Focus**: bobette.riner, eric.bass, johnny.palmer deserve public acknowledgment
- **Strategy**: Create peer recognition systems based on positive sentiment contributions

### 3. **Targeted Support Initiatives**
- **Priority**: Provide additional support for employees with lower sentiment scores
- **Focus**: kayne.coulter, patti.thompson, john.arnold need attention
- **Implementation**: Mentoring programs, coaching, or additional resources

### 4. **Continuous Monitoring System**
- **Dashboard**: Establish monthly sentiment monitoring dashboard
- **Alerts**: Set up early warning system for declining sentiment trends
- **Reviews**: Implement quarterly sentiment review meetings with management

### 5. **Communication Enhancement**
- **Training**: Provide positive communication techniques training
- **Feedback**: Implement feedback mechanisms for communication effectiveness
- **Best Practices**: Learn from high performers to improve team dynamics

## 🔧 Technical Performance

### Predictive Model Results
- **Model Type**: Random Forest Classifier
- **Overall Accuracy**: 74.3%
- **Features Used**: 10 observable message-level characteristics
- **Training Data**: 1,752 messages (80%)
- **Test Data**: 439 messages (20%)

### Model Performance by Class
- **Neutral Messages**: 96.1% recall (excellent performance)
- **Positive Messages**: 4.9% recall (limited due to class imbalance)
- **Negative Messages**: 0.0% recall (limited due to class imbalance)

### Top Feature Importance
1. **Text Length** (31.1%) - Longer messages express clearer sentiment
2. **Word Count** (26.9%) - Detailed communication correlates with sentiment
3. **Month** (16.4%) - Seasonal patterns in communication
4. **Day of Week** (13.0%) - Weekly communication patterns
5. **Employee Average Sentiment** (2.7%) - Historical patterns matter

## 🚀 Business Impact

### Immediate Benefits
- **Workforce Stability**: No flight risk employees identified
- **Positive Environment**: 94.8% neutral/positive communication
- **Targeted Support**: Clear identification of employees needing attention
- **Recognition Opportunities**: Specific high-performing employees identified

### Long-term Value
- **Predictive Monitoring**: Foundation for ongoing sentiment tracking
- **Intervention Systems**: Early warning capabilities for employee disengagement
- **Performance Management**: Data-driven employee evaluation support
- **Organizational Health**: Quantitative metrics for workplace communication quality

## 📁 Project Structure

```
employee_sentiment_analysis/
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Processed datasets and results
│   └── plots/                  # Generated visualizations
├── src/
│   ├── utils/                  # Utility functions
│   ├── models/                 # ML models and algorithms
│   └── visualization/          # Plotting and visualization tools
├── visualizations/             # Final visualization outputs
├── Employee_Sentiment_Analysis_Complete.ipynb  # Main analysis notebook
├── README.md                   # This file
└── requirements.txt           # Dependencies
```

## 🛠️ Installation and Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd employee_sentiment_analysis

# Install required packages
pip install -r requirements.txt
```

### Usage
```bash
# Run the complete analysis
jupyter notebook Employee_Sentiment_Analysis_Complete.ipynb
```

## 📈 Output Files

### Data Files
- `processed_employee_data.csv`: Complete processed dataset
- `employee_analysis.csv`: Employee-level sentiment analysis
- `monthly_rankings.csv`: Monthly employee rankings
- `flight_risk_analysis.csv`: Flight risk assessment results
- `model_results.json`: Predictive model performance metrics
- `summary_statistics.json`: Comprehensive analysis summary

### Visualizations
- `eda_summary.png`: Comprehensive EDA overview
- `sentiment_distribution.png`: Sentiment distribution analysis
- `time_series_analysis.png`: Temporal sentiment patterns
- `monthly_rankings.png`: Employee ranking visualizations
- `flight_risk_analysis.png`: Flight risk analysis charts
- `predictive_modeling.png`: Model performance visualization

## 🔍 Model Performance and Limitations

### Strengths
- **Excellent neutral classification**: 96.1% recall for neutral messages
- **Robust feature engineering**: Uses only observable characteristics
- **Efficient processing**: Handles large message volumes effectively
- **Professional-grade pipeline**: Advanced NLP with RoBERTa transformer model

### Limitations
- **Class imbalance**: Affects positive/negative sentiment detection
- **Limited minority class performance**: Few positive/negative samples
- **Dependency on message quality**: Requires well-structured text
- **Historical data**: Based on 2010-2011 communications

### Future Improvements
- **Resampling techniques**: Implement SMOTE for class balance
- **Feature expansion**: Add linguistic features (punctuation, capitalization)
- **Ensemble methods**: Combine multiple models for improved accuracy
- **Real-time processing**: Implement streaming analysis capabilities

