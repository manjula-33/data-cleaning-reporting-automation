import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("sales_data.csv")

# Features
X = df[['Month']]
y = df['Sales']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict future sales
future = [[7]]
prediction = model.predict(future)

print("Predicted Sales:", prediction[0])

# Visualization
plt.scatter(X, y)
plt.plot(X, model.predict(X))

plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Predictive Analytics")

plt.show()