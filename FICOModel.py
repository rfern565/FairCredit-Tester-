import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from fairlearn.metrics import demographic_parity_difference

# Load the dataset
df = pd.read_csv("/Users/ryanfernandez/Downloads/Credit Score Classification Dataset.csv")

# Encode categorical columns
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Education'] = df['Education'].map({'High School Diploma': 0, "Associate's Degree": 1, 
                                       "Bachelor's Degree": 2, 'Master\'s Degree': 3, 'Doctorate': 4})
df['Marital Status'] = df['Marital Status'].map({'Single': 0, 'Married': 1})
df['Home Ownership'] = df['Home Ownership'].map({'Owned': 1, 'Rented': 0})

# Handle missing values
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Use direct assignment to avoid FutureWarning
for column in df.select_dtypes(include=['object']).columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# Select relevant features
X = df[['Age', 'Gender', 'Income', 'Education', 'Marital Status', 'Number of Children', 'Home Ownership']]
y = df['Credit Score']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a logistic regression model
fico_model = LogisticRegression(max_iter=1000)
fico_model.fit(X_train_scaled, y_train)

# Predictions and evaluation
y_pred_fico = fico_model.predict(X_test_scaled)
accuracy_fico = accuracy_score(y_test, y_pred_fico)
print(f"FICO Model Accuracy: {accuracy_fico:.4f}")

# Fairness check by gender
# Align indices for gender in test set
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Gender-specific accuracy
female_mask = X_test['Gender'] == 0
male_mask = X_test['Gender'] == 1

female_accuracy = accuracy_score(y_test[female_mask], y_pred_fico[female_mask])
male_accuracy = accuracy_score(y_test[male_mask], y_pred_fico[male_mask])

print(f"Accuracy for Females: {female_accuracy:.4f}")
print(f"Accuracy for Males: {male_accuracy:.4f}")

# Accuracy disparity
disparity = abs(female_accuracy - male_accuracy)
print(f"Accuracy disparity between genders: {disparity:.4f}")

# Check demographic parity difference
demographic_parity = demographic_parity_difference(y_test, y_pred_fico, sensitive_features=X_test['Gender'])
print(f"Demographic parity difference: {demographic_parity:.4f}")
