import onnxruntime as ort
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys

# Load ONNX model
onnx_model_path = "/Users/adarshashok/Downloads/treehacks/healthbot/src/best_model.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# Data preprocessing for a single file
def process_single_file(file_path, window_size=500, overlap=0.75):
    """Process a single CSV file and return windowed data"""
    try:
        # Read data using pandas
        data = pd.read_csv(file_path, header=None)
        
        # Extract numerical columns (skipping the first row)
        numerical_data = data.iloc[1:].values.astype(float)
        
        # Separate features
        sensor_readings = numerical_data[:, :3]  # green, red, IR readings
        accelerations = numerical_data[:, 3:6]   # acc_x, acc_y, acc_z
        
        # Normalize features
        scaler = StandardScaler()
        sensor_normalized = scaler.fit_transform(sensor_readings)
        accelerations_normalized = scaler.fit_transform(accelerations)
        
        # Combine normalized features
        features = np.concatenate([sensor_normalized, accelerations_normalized], axis=1)
        
        # Create overlapping windows
        stride = int(window_size * (1 - overlap))
        windows = [features[i:i + window_size] for i in range(0, len(features) - window_size, stride)]
        
        return np.array(windows).astype(np.float32)
    
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

# Prediction function
def predict_heart_rhythm(file_path, window_size=500):
    """Predict the heart rhythm class using ONNX model"""
    preprocessed_data = process_single_file(file_path, window_size)
    if preprocessed_data is None:
        return None, None
    
    # Get input name for ONNX model
    input_name = ort_session.get_inputs()[0].name
    
    # Run inference
    predictions = ort_session.run(None, {input_name: preprocessed_data})[0]
    
    # Class labels
    class_names = ['AFib', 'Irregular', 'Regular']
    
    # Average predictions across windows
    avg_prediction = predictions.mean(axis=0)
    predicted_class = class_names[np.argmax(avg_prediction)]
    confidence = np.max(avg_prediction)
    
    return predicted_class, confidence

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 heart_predictor.py <csv_file1> <csv_file2> ...")
        sys.exit(1)

    file_paths = sys.argv[1:]
    
    for file_path in file_paths:
        predicted_class, confidence = predict_heart_rhythm(file_path)
        if predicted_class is not None:
            print(predicted_class)
        else:
            print(f"{file_path}: Prediction failed.")

if __name__ == "__main__":
    main()
