# LAPD Crime-Data-Analysis (2020 to 2024)

# Introduction
In this project, analysis of the LAPD Crime data spanning from 2020 to 2024 is conducted using pyspark. The selection of this dataset is based on its significance in comprehending and potentially mitigating crime-related issues in Los Angeles. By employing big data analytics, our objective is to unveil insights, patterns, and trends within the data to facilitate informed decision-making and proactive measures in law enforcement and public safety.

# Research Questions
The research questions addressed in this analysis are as follows:

Victim Vulnerability Analysis: Can patterns of victim vulnerability be identified by analyzing the relationship between victim demographics and the types of crimes reported?

Crime Types and Locations: What are the most frequently recorded categories of crimes in the dataset, and are certain crimes more prevalent in specific regions or police districts, allowing for the identification of high-crime zones based on location data?

# Methodology
To analyze the LAPD Crime data, a multi-step methodology leveraging machine learning techniques and exploratory data analysis (EDA) was employed. Two models were utilized to address the research questions:

Model 1: Random Forest
Model 2: Logistic Regression
These methodologies facilitated the extraction of meaningful insights from the LAPD Crime dataset, enabling the resolution of real-world questions posed at the outset of the project. Through rigorous analysis and modeling, the aim was to contribute to the enhancement of crime prevention strategies and informed decision-making in law enforcement and public safety efforts in Los Angeles.

# Analysis Approach
The analysis began with exploratory data analysis (EDA) using Tableau to explore correlations, patterns, and trends within the data. Two interactive dashboards were created to provide comprehensive insights into various aspects of LAPD crime data:

Dashboard 1: Explores crime categories, zones, gender disparities, and age trends.
Dashboard 2: Provides deeper insights into crime trends across police districts, over time, and concerning weapon use and victim demographics.
Following the creation of dashboards, data loading and cleaning procedures were executed to ensure data reliability and accuracy. Exploratory data analysis (EDA) was then conducted to gain insights into the characteristics and distributions of various variables within the dataset.

# Model Performance
Model 1: Random Forest

Accuracy: Approximately 99.04%
Precision, Recall, and F1-score for class 0 (not vulnerable victims) indicated high performance.
Challenges were observed in correctly identifying vulnerable instances, with a significant number of false negatives.
Model 2: Logistic Regression

Similar performance metrics were achieved as Model 1, indicating accurate classification.
Difficulties were highlighted in distinguishing between "Not Common" and "Common" classes.

# Results and Conclusion
The analysis of LAPD Crime data aimed to uncover insights into crime patterns and trends in Los Angeles, with the goal of informing proactive measures in law enforcement and public safety. Despite the high accuracy and precision of both models, challenges were identified, suggesting potential areas for improvement. Continuous refinement and enhancement of crime prevention strategies and law enforcement efforts are essential based on the insights gained from the analysis.

The code for the analysis can be found in the accompanying code files.
