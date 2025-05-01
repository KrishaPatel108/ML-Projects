import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
# Load the dataset
dt = pd.read_csv(r'C:\Krisha Patel\PYTHON PROJECTS\Titanic\train.csv')
# Preview the first few rows of the data
ff_lines =dt.head()
print('The data has been loaded successfully and here are the first few lines:\n',ff_lines)
# Check the column names and general info
c_names=dt.info()
print(c_names,'Column names and genral info:\n')
# Check for any missing values
missing_values = dt.isnull().sum()
print('Upon checking for missing values, we find:\n',missing_values)
# Statistical summary of the data
stats_summary=dt.describe()
print('Statistical summary of the data is given below:\n',stats_summary)
# Visualize the distribution of survival
sns.countplot(x='Survived', data=dt)
plt.title('Survival Distribution')
plt.show() 
# Handling missing values (for example, filling Age with the median)
dt['Age'].fillna(dt['Age'].median(), inplace=True)
dt['Embarked'].fillna(dt['Embarked'].mode()[0], inplace=True)
dt['Cabin'].fillna('U', inplace=True)  # Fill missing Cabin values with 'U' for unknown
# Drop columns that won’t be useful for our model
dt.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
# Convert categorical variables to numeric using pandas get_dummies
dt = pd.get_dummies(dt, columns=['Sex', 'Embarked'], drop_first=True)
# Check the cleaned dataset
print('The cleaned dataset is:',dt.head())
# Features (independent variables)
X = dt.drop('Survived', axis=1)
# Label (dependent variable)
y = dt['Survived']
# Split the data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Initialize the logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train_scaled, y_train)
# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.show()
plt.close()
