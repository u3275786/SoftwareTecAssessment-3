import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib

# Load the saved model and scaler
try:
    model = joblib.load('linear_regression_model.pkl')  # Ensure the model file is in the same directory
    scaler = joblib.load('scaler.pkl')  # Ensure the scaler file is in the same directory
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print("Model or scaler file not found. Ensure the model is retrained and saved correctly.")

# Function to make predictions based on user input
def predict_close(open_price, high_price, low_price, volume):
    try:
        # Prepare the input data for prediction
        input_data = np.array([[open_price, high_price, low_price, volume]])
        print("Raw input data:", input_data)  # Debug print

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        print("Scaled input data:", input_data_scaled)  # Debug print

        # Make the prediction
        prediction = model.predict(input_data_scaled)
        print("Prediction:", prediction)  # Debug print

        return prediction[0]  # Return the predicted close price
    except Exception as e:
        print(f"Error during prediction: {e}")
        messagebox.showerror("Prediction Error", f"An error occurred during prediction: {e}")
        return None

# Function to handle button click and predict the Close price
def predict_button_click():
    try:
        # Get user input
        open_price = float(entry_open.get())
        high_price = float(entry_high.get())
        low_price = float(entry_low.get())
        volume = float(entry_volume.get())
        
        # Use the prediction function to predict the Close price
        predicted_close = predict_close(open_price, high_price, low_price, volume)
        
        # Display the prediction result
        if predicted_close is not None:
            result_label.config(text=f"Predicted Close Price: {predicted_close:.2f}")
    
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")

# Create the Tkinter window
window = tk.Tk()
window.title("Stock Price Predictor")

# Create labels and entry fields for user input
tk.Label(window, text="Open Price").grid(row=0, column=0)
entry_open = tk.Entry(window)
entry_open.grid(row=0, column=1)

tk.Label(window, text="High Price").grid(row=1, column=0)
entry_high = tk.Entry(window)
entry_high.grid(row=1, column=1)

tk.Label(window, text="Low Price").grid(row=2, column=0)
entry_low = tk.Entry(window)
entry_low.grid(row=2, column=1)

tk.Label(window, text="Volume").grid(row=3, column=0)
entry_volume = tk.Entry(window)
entry_volume.grid(row=3, column=1)

# Create a button to trigger the prediction
predict_button = tk.Button(window, text="Predict", command=predict_button_click)
predict_button.grid(row=4, columnspan=2)

# Label to display the result
result_label = tk.Label(window, text="")
result_label.grid(row=5, columnspan=2)

# Start the Tkinter event loop
window.mainloop()
