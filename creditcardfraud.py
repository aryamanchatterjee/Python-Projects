import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

# Load dataset
data = pd.read_csv("creditcard.csv")

# Separate fraudulent and valid transactions
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

# Calculate the ratio of fraud to valid transactions (for context on imbalance)
outlierFraction = len(fraud) / len(valid)
print(outlierFraction)

# Prepare features and target variable
X = data.drop(['Class'], axis=1)
Y = data["Class"]

print(X.shape)
print(Y.shape)

# Convert to NumPy arrays
xData = X.values
yData = Y.values

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(
    xData, yData, test_size=0.2, random_state=42)

# Train a Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)

# Make predictions on test set
yPred = rfc.predict(xTest)

# Evaluate the model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

accuracy = accuracy_score(yTest, yPred)
precision = precision_score(yTest, yPred)
recall = recall_score(yTest, yPred)
f1 = f1_score(yTest, yPred)
mcc = matthews_corrcoef(yTest, yPred)

print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")

# Visualize the confusion matrix
conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()

# Save the trained model to disk using pickle (joblib)
import joblib
joblib.dump(rfc, "fraud_model.pkl")
