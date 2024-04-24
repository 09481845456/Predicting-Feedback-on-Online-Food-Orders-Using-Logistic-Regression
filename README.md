# Predicting-Feedback-on-Online-Food-Orders-Using-Logistic-Regression
# **Predicting Feedback on Online Food Orders Using Logistic Regression**

# Part 1: Data Loading and Preprocessing

## Import Necessary Libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

# Load the Dataset

df = pd.read_csv('onlinefoods.csv')
print(df.head())


# Handle Missing Values

missing_values = df.isnull().sum()
print("Missing Values:\n",missing_values)
#impute missing values with the mean of the column
df.fillna(df.mean(numeric_only=True), inplace=True)

## Encode Categorical Variables

# Convert categorical variables into one-hot encoded format
df_encoded = pd.get_dummies(df, columns=['Gender', 'Marital Status', 'Occupation', 'Family size'], drop_first=True)
# Label Encoding (for ordinal categories)
Income_mapping = {'Below Rs.10000': 0, '10001 to 25000': 1, '25001 to 50000': 2, 'More than 50000': 3}
df_encoded['Monthly Income'] = df['Monthly Income'].map(Income_mapping)
print(df_encoded.head())

The following features can be considered for inclusion in the model:

Family Size: Family size may influence monthly income as larger families may have higher expenses, which could affect the income level.

Educational Qualifications: Education level often correlates with income, as individuals with higher education qualifications tend to earn more.

These features are relevant for predicting monthly income because they capture demographic and socioeconomic factors that can impact an individual's earning potential. Additionally, these features showed significant associations with the target variable (monthly income) during exploratory data analysis, making them suitable candidates for inclusion in the model.

# Part 2: Exploratory Data Analysis (EDA

## Descriptive Statistics

numeric_summary = df.describe()
print(numeric_summary)

## Visualizations

# Distribution of Age and its impact on Feedback
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', hue='Feedback', multiple='stack', bins=20)
plt.title('Distribution of Age and its impact on Feedback')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(title='Feedback')
plt.show()


# Proportions of Feedback across different levels of Monthly Income
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Monthly Income', hue='Feedback')
plt.title('Proportions of Feedback across different levels of Monthly Income')
plt.xlabel('Monthly Income')
plt.ylabel('Count')
plt.legend(title='Feedback')
plt.show()

# Correlation matrix heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

# Part 3: Logistic Regression Model

## Build the Model

X = df[['Family size','Educational Qualifications']]
X_encoded = pd.get_dummies(X, columns=['Educational Qualifications'], drop_first=True)
y = df['Monthly Income']
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

## Model Evaluation



accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
#Visualization of Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Part 4: Data Analysis and Visualization

## Feature Importance

# Retrieve feature coefficients
feature_importance = model.coef_[0]

# Create a DataFrame to store feature importance
df_importance = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Coefficient': feature_importance
})

# Sort the DataFrame by coefficient values
df_importance = df_importance.sort_values(by='Coefficient', ascending=False)
# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=df_importance, hue='Feature', palette='viridis', legend=False)
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.title('Feature Importance in Logistic Regression Model')
plt.show()


## Prediction Insights

# Get predicted probabilities
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Plot the distribution of predicted probabilities
plt.figure(figsize=(8, 6))
sns.histplot(y_pred_prob, bins=20, kde=True, color='skyblue')
plt.title('Distribution of Predicted Probabilities')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.show()


By examining the spread and concentration of predicted probabilities, we can measure  the model's confidence in its predictions. If the distribution is narrow and peaks at extreme probabilities (close to 0 or 1), it indicates high confidence. Conversely, a broader distribution suggests lower confidence. Understanding the distribution helps assess the reliability of the model's predictions. If the majority of predictions cluster around the true outcome (0 or 1), it suggests the model is making accurate predictions. However, if there's a wide dispersion or unexpected patterns in the distribution, it may indicate areas where the model struggles to make reliable predictions. Comparing the predicted probabilities against the actual outcomes can provide insights into the model's performance. If the distribution aligns well with the observed outcomes, it suggests the model is effectively capturing the underlying patterns in the data. Conversely, discrepancies between predicted and actual probabilities highlight areas for improvement. The distribution can also inform decisions about the threshold for classification. For instance, if the task involves binary classification (e.g., positive/negative feedback), adjusting the threshold based on the distribution can optimize the trade-off between sensitivity and specificity.
