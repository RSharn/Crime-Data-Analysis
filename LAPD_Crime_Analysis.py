#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier



# Load the dataset
crime_data = pd.read_csv("Dataset.csv")
crime_data.head()


# In[5]:


# Data Cleaning
crime_data.dropna(inplace=True)
# EDA
crime_types_freq = crime_data['Crm Cd Desc'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=crime_types_freq.values, y=crime_types_freq.index, palette="viridis")
plt.title('Top 10 Most Frequently Recorded Crimes')
plt.xlabel('Frequency')
plt.ylabel('Crime Type')
plt.show()

# Victim Demographics Analysis
# Analyzing victim demographics vs. crime type using a heatmap
plt.figure(figsize=(12, 6))
crime_type_counts = crime_data['Crm Cd Desc'].value_counts().head(5).index
victim_descent_counts = crime_data['Vict Descent'].value_counts().head(5).index
heatmap_data = crime_data[crime_data['Crm Cd Desc'].isin(crime_type_counts) & crime_data['Vict Descent'].isin(victim_descent_counts)]
heatmap_data = heatmap_data.groupby(['Vict Descent', 'Crm Cd Desc']).size().unstack(fill_value=0)
sns.heatmap(heatmap_data, cmap='Blues', annot=True, fmt='d', linewidths=.5)
plt.title('Top 5 Victim Demographics vs Crime Type')
plt.xlabel('Crime Type')
plt.ylabel('Victim Descent')
plt.show()

# Temporal Analysis of Crime Trends
# Convert date columns to datetime
crime_data['Date Rptd'] = pd.to_datetime(crime_data['Date Rptd'])
crime_data['DATE OCC'] = pd.to_datetime(crime_data['DATE OCC'])

# Extract year and month from 'Date Rptd'
crime_data['Year'] = crime_data['Date Rptd'].dt.year
crime_data['Month'] = crime_data['Date Rptd'].dt.month

# Plotting number of crimes over years
plt.figure(figsize=(10, 6))
sns.countplot(x='Year', data=crime_data)
plt.title('Number of Crimes Reported Over Years')
plt.xlabel('Year')
plt.ylabel('Number of Crimes')
plt.xticks(rotation=45)
plt.show()

# Analyzing victim descent distribution using a pie chart
plt.figure(figsize=(8, 8))
victim_descent_counts = crime_data['Vict Descent'].value_counts().head(5)
plt.pie(victim_descent_counts, labels=victim_descent_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Top 5 Victim Descent Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()


# In[36]:


print(crime_data.columns)


# In[41]:


###########################################  Question 1 ########################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Define the target variable
X = crime_data[['Vict Age', 'Vict Sex', 'Vict Descent']]
y = crime_data['Vulnerable_Victim']

# Perform one-hot encoding for categorical features
X = pd.get_dummies(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, predictions))


# In[50]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
    else:
        title = 'Confusion Matrix'

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# Now you can use this function to plot the confusion matrix
plot_confusion_matrix(y_test, predictions, ['Not Vulnerable', 'Vulnerable'], normalize=True)


# In[34]:


###########################################  Question 2 ########################################################
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression

# Select relevant features
features = ['AREA', 'Crm Cd']  # Adjust based on your dataset and feature engineering

X = crime_data[features]
y = crime_data['Vulnerable_Victim']  # Assuming 'Vulnerable_Victim' indicates whether a crime is common or not

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train, y_train)

# Make predictions
predictions = logistic_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, predictions))


# In[17]:


############################ Confusion Matrix for Logistic Regression ################################

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Train the logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Make predictions
predictions = logistic_model.predict(X_test)

# Calculate probabilities
probs = logistic_model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, probs)

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues, interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks([0, 1], ['Not Common', 'Common'])
plt.yticks([0, 1], ['Not Common', 'Common'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, conf_matrix[i, j], ha='center', va='center', color='white')
plt.show()

