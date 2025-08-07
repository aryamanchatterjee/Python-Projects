import pandas as pd
import joblib

# Load the trained model
model = joblib.load("fraud_model.pkl")

# Load new transactions
data = pd.read_csv("new_transactions.csv")

# Predict
predictions = model.predict(data)

# Add predictions to the dataframe
data['Prediction'] = predictions

# Ask user what they want to see
print("\nChoose what to display:")
print("1 - Show only fraudulent transactions")
print("2 - Show only valid transactions")
print("3 - Show all transactions")

choice = input("Enter your choice (1/2/3): ")

if choice == '1':
    result = data[data['Prediction'] == 1]
    print("\n🟥 Fraudulent Transactions:")
elif choice == '2':
    result = data[data['Prediction'] == 0]
    print("\n✅ Valid Transactions:")
else:
    result = data
    print("\n📋 All Transactions:")

print(result)