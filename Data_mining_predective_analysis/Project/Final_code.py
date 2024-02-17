import pandas as pd

# Load the dataset to examine its structure
file_path = 'W:\Fau\Data_mining_predictive\Project\dataset\WineQT.csv'
wine_data = pd.read_csv(file_path)

# Display the first few rows of the dataset
wine_data.head()


# Remove the 'Id' column
wine_data.drop('Id', axis=1, inplace=True)

# Calculate the minimum and maximum pH values
min_pH = wine_data['pH'].min()
max_pH = wine_data['pH'].max()

# Add columns for red and white wines based on pH values
wine_data['White Wine'] = (wine_data['pH'] >= min_pH) & (wine_data['pH'] <= 3.4)
wine_data['Red Wine'] = (wine_data['pH'] > 3.4) & (wine_data['pH'] <= max_pH)

# Convert boolean values to integers for clarity
wine_data['White Wine'] = wine_data['White Wine'].astype(int)
wine_data['Red Wine'] = wine_data['Red Wine'].astype(int)

# Display the first few rows of the modified dataset
wine_data.head()


# Checking for missing values in the dataset
missing_values = wine_data.isnull().sum()

missing_values[missing_values > 0]  # Display only columns with missing values, if any


# Calculating IQR for each column
Q1 = wine_data.quantile(0.25)
Q3 = wine_data.quantile(0.75)
IQR = Q3 - Q1

# Determining outliers
outliers = ((wine_data < (Q1 - 1.5 * IQR)) | (wine_data > (Q3 + 1.5 * IQR))).sum()

outliers[outliers > 0]  # Display only columns with outliers, if any


from sklearn.preprocessing import StandardScaler

# Selecting only numeric columns for scaling (excluding binary columns 'White Wine' and 'Red Wine')
numeric_cols = wine_data.columns.difference(['White Wine', 'Red Wine'])

# Applying StandardScaler
scaler = StandardScaler()
wine_data_scaled = wine_data.copy()
wine_data_scaled[numeric_cols] = scaler.fit_transform(wine_data[numeric_cols])

# Display the first few rows of the scaled dataset
wine_data_scaled.head()


import matplotlib.pyplot as plt
import seaborn as sns

# Setting the aesthetics for the plots
sns.set(style="whitegrid")

# Plotting histograms for each numeric variable
plt.figure(figsize=(20, 15))
for i, col in enumerate(numeric_cols):
    plt.subplot(4, 3, i + 1)
    sns.histplot(wine_data[col], kde=True, bins=20)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


# Calculating the correlation matrix
corr_matrix = wine_data[numeric_cols].corr()

# Plotting the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Wine Attributes')
plt.show()


# Creating box plots to compare red and white wines
plt.figure(figsize=(15, 6))

# Alcohol content comparison
plt.subplot(1, 2, 1)
sns.boxplot(x='Red Wine', y='alcohol', data=wine_data)
plt.title('Alcohol Content in Red vs White Wine')
plt.xticks([0, 1], ['White Wine', 'Red Wine'])

# pH level comparison
plt.subplot(1, 2, 2)
sns.boxplot(x='Red Wine', y='pH', data=wine_data)
plt.title('pH Level in Red vs White Wine')
plt.xticks([0, 1], ['White Wine', 'Red Wine'])

plt.tight_layout()
plt.show()


# Scatter plots for wine quality vs other variables
plt.figure(figsize=(18, 6))

# Alcohol vs Quality
plt.subplot(1, 3, 1)
sns.scatterplot(x='quality', y='alcohol', data=wine_data)
plt.title('Alcohol Content vs Quality')

# Volatile Acidity vs Quality
plt.subplot(1, 3, 2)
sns.scatterplot(x='quality', y='volatile acidity', data=wine_data)
plt.title('Volatile Acidity vs Quality')

# Residual Sugar vs Quality
plt.subplot(1, 3, 3)
sns.scatterplot(x='quality', y='residual sugar', data=wine_data)
plt.title('Residual Sugar vs Quality')

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Data distribution for key features
plt.figure(figsize=(15, 7))

# Subplot 1: Distribution of Alcohol
plt.subplot(1, 3, 1)
sns.histplot(wine_data['alcohol'], kde=True)
plt.title('Distribution of Alcohol')

# Subplot 2: Distribution of pH
plt.subplot(1, 3, 2)
sns.histplot(wine_data['pH'], kde=True)
plt.title('Distribution of pH')

# Subplot 3: Quality distribution
plt.subplot(1, 3, 3)
sns.countplot(x='quality', data=wine_data)
plt.title('Quality Distribution')

plt.tight_layout()
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(wine_data.corr(), annot=True, fmt=".2f", cmap='viridis')
plt.title('Correlation Matrix')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Preparing data for the first question
X1 = wine_data.drop(['Red Wine', 'White Wine', 'quality'], axis=1)  # features
y1 = wine_data['Red Wine']  # target

# Splitting the dataset into training and testing sets for the first question
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=42)

# Preparing data for the second question
# Transforming the 'quality' column into three classes: cheap (1-4), average (5-6), expensive (7-10)
wine_data['Quality Class'] = pd.cut(wine_data['quality'], bins=[0, 4, 6, 10], labels=['Cheap', 'Average', 'Expensive'])
X2 = wine_data.drop(['Red Wine', 'White Wine', 'quality', 'Quality Class'], axis=1)  # features
y2 = wine_data['Quality Class']  # target

# Encoding the target variable for the second question
label_encoder = LabelEncoder()
y2_encoded = label_encoder.fit_transform(y2)

# Splitting the dataset into training and testing sets for the second question
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2_encoded, test_size=0.3, random_state=42)

# Outputting the shape of the datasets to confirm the splits
X1_train.shape, X1_test.shape, y1_train.shape, y1_test.shape, X2_train.shape, X2_test.shape, y2_train.shape, y2_test.shape


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Creating and training the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X1_train, y1_train)

# Creating and training the Logistic Regression model
logistic_regression = LogisticRegression(random_state=42, max_iter=1000)
logistic_regression.fit(X1_train, y1_train)

# Evaluating the Random Forest Classifier
rf_pred = rf_classifier.predict(X1_test)
rf_report = classification_report(y1_test, rf_pred)

# Evaluating the Logistic Regression model
log_reg_pred = logistic_regression.predict(X1_test)
log_reg_report = classification_report(y1_test, log_reg_pred)

# Displaying the performance metrics of the Random Forest Classifier
print("Random Forest Classifier Metrics:")
print("-" * 30)
print("Accuracy: {:.2f}".format(accuracy_score(y1_test, rf_pred)))
print("Precision: {:.2f}".format(precision_score(y1_test, rf_pred)))
print("Recall: {:.2f}".format(recall_score(y1_test, rf_pred)))
print("F1 Score: {:.2f}".format(f1_score(y1_test, rf_pred)))
print("\nClassification Report:")
print(classification_report(y1_test, rf_pred))

# Displaying the performance metrics of the Logistic Regression model
print("\nLogistic Regression Metrics:")
print("-" * 30)
print("Accuracy: {:.2f}".format(accuracy_score(y1_test, log_reg_pred)))
print("Precision: {:.2f}".format(precision_score(y1_test, log_reg_pred)))
print("Recall: {:.2f}".format(recall_score(y1_test, log_reg_pred)))
print("F1 Score: {:.2f}".format(f1_score(y1_test, log_reg_pred)))
print("\nClassification Report:")
print(classification_report(y1_test, log_reg_pred))


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# Creating and training the Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(random_state=42)
gb_classifier.fit(X2_train, y2_train)

# Creating and training the Support Vector Machine model
svm_classifier = SVC(random_state=42)
svm_classifier.fit(X2_train, y2_train)

# Evaluating the Gradient Boosting Classifier
gb_pred = gb_classifier.predict(X2_test)
gb_report = classification_report(y2_test, gb_pred, target_names=label_encoder.classes_)

# Evaluating the Support Vector Machine model
svm_pred = svm_classifier.predict(X2_test)
svm_report = classification_report(y2_test, svm_pred, target_names=label_encoder.classes_)

# Displaying the performance metrics of the Gradient Boosting Classifier
print("Gradient Boosting Classifier Metrics:")
print("-" * 30)
print("Accuracy: {:.2f}".format(accuracy_score(y2_test, gb_pred)))
print("Precision: {:.2f}".format(precision_score(y2_test, gb_pred, average='weighted')))
print("Recall: {:.2f}".format(recall_score(y2_test, gb_pred, average='weighted')))
print("F1 Score: {:.2f}".format(f1_score(y2_test, gb_pred, average='weighted')))
print("\nClassification Report:")
print(gb_report)

# Displaying the performance metrics of the Support Vector Machine model
print("\nSupport Vector Machine Metrics:")
print("-" * 30)
print("Accuracy: {:.2f}".format(accuracy_score(y2_test, svm_pred)))
print("Precision: {:.2f}".format(precision_score(y2_test, svm_pred, average='weighted')))
print("Recall: {:.2f}".format(recall_score(y2_test, svm_pred, average='weighted')))
print("F1 Score: {:.2f}".format(f1_score(y2_test, svm_pred, average='weighted')))
print("\nClassification Report:")
print(svm_report)


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Function to calculate evaluation metrics
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return accuracy, precision, recall, f1

# Evaluating Random Forest Classifier
rf_accuracy, rf_precision, rf_recall, rf_f1 = evaluate_model(y1_test, rf_pred)

# Evaluating Logistic Regression
log_reg_accuracy, log_reg_precision, log_reg_recall, log_reg_f1 = evaluate_model(y1_test, log_reg_pred)

# Creating confusion matrices for Random Forest and Logistic Regression
rf_cm = confusion_matrix(y1_test, rf_pred)
log_reg_cm = confusion_matrix(y1_test, log_reg_pred)

# Plotting confusion matrices
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 2, 2)
sns.heatmap(log_reg_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()

# Creating a DataFrame to display the evaluation metrics
metrics_df = pd.DataFrame({
    'Model': ['Random Forest', 'Logistic Regression'],
    'Accuracy': [rf_accuracy, log_reg_accuracy],
    'Precision': [rf_precision, log_reg_precision],
    'Recall': [rf_recall, log_reg_recall],
    'F1 Score': [rf_f1, log_reg_f1]
})

metrics_df


# Evaluating Gradient Boosting Classifier
gb_accuracy, gb_precision, gb_recall, gb_f1 = evaluate_model(y2_test, gb_pred)

# Evaluating SVM
svm_accuracy, svm_precision, svm_recall, svm_f1 = evaluate_model(y2_test, svm_pred)

# Creating confusion matrices for Gradient Boosting and SVM
gb_cm = confusion_matrix(y2_test, gb_pred)
svm_cm = confusion_matrix(y2_test, svm_pred)

# Plotting confusion matrices
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.heatmap(gb_cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Gradient Boosting Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 2, 2)
sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()

# Adding the evaluation metrics for Gradient Boosting and SVM to the DataFrame
metrics_df_2 = pd.DataFrame({
    'Model': ['Gradient Boosting', 'SVM'],
    'Accuracy': [gb_accuracy, svm_accuracy],
    'Precision': [gb_precision, svm_precision],
    'Recall': [gb_recall, svm_recall],
    'F1 Score': [gb_f1, svm_f1]
})

metrics_df = metrics_df.append(metrics_df_2, ignore_index=True)
metrics_df


