# HR Attrition Analysis

## 📌 Overview
This project analyzes employee attrition using HR data to identify key drivers and provide HR insights.

## 📊 Data Source
- Kaggle IBM HR Analytics Dataset  
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

## 🔍 Analysis Process
1. Data preprocessing (encoding categorical variables, removing irrelevant features)
2. Correlation analysis to identify candidate variables
3. Multicollinearity check using VIF
4. Logistic regression for significance testing
5. Odds ratio analysis to measure effect size

## 📈 Key Findings
@ Company-wide
- Overtime increases attrition risk by approximately 4.3 times
- Single employees show higher attrition (~2.5x)
- Sales representatives have higher attrition compared to other roles (~2.5x)

@ Sales Role (Deep Dive)
- Sales representatives who work overtime are about 11 times more likely to leave
- Frequent Business travel increases attrition risk by approximately 7 times
- Job involvement is negatively associated with attrition
- Job satisfaction is negatively associated with attrition


## 💡 HR Implications
- Implement overtime control policies
- Improve retention strategies for sales roles
- Focus on engagement and support for sales representatives

## 🛠 Tools
- Python (pandas, numpy, statsmodels)
- VS Code
- Excel
- Power BI

## 📁 Project Scope
- Company-wide attrition analysis
- Sales role-specific attrition analysis
