import pickle
import numpy as np

# Load the model
with open('svr_model_Avg_CPM.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare your input data (must have the same features as training data)
# This is just an example - you'd need to process your actual input data
X_new = np.random.random((1, 106))  # Assuming 100 features

# Make a prediction
prediction = model.predict(X_new)
print(f"Predicted value: {prediction[0]}")