import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ============================================
# Section 1: Data Preprocessing & PyTorch Model Prediction
# ============================================
def process_single_file(file_path, window_size=750, overlap=0.5):
    """Process a single CSV file and return windowed data (as numpy array)."""
    try:
        # Read data using pandas (assumes no header or a header row we skip)
        data = pd.read_csv(file_path, header=None)
        numerical_data = data.iloc[1:].values  # Skip header if present
        numerical_data = numerical_data.astype(float)
        
        # Separate features: first 3 columns are PPG signals, next 3 are accelerometer data
        sensor_readings = numerical_data[:, :3]
        accelerations = numerical_data[:, 3:6]
        
        # Normalize features
        scaler = StandardScaler()
        sensor_normalized = scaler.fit_transform(sensor_readings)
        accelerations_normalized = scaler.fit_transform(accelerations)
        
        # Combine normalized features
        features = np.concatenate([sensor_normalized, accelerations_normalized], axis=1)
        
        # Create windows with overlap
        stride = int(window_size * (1 - overlap))
        windows = [features[i:i + window_size] for i in range(0, len(features) - window_size, stride)]
        return np.array(windows)
    
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def predict_heart_rhythm(model, file_path, window_size=500, device='cpu'):
    """
    Predict the heart rhythm class for a new CSV file.
    Returns: predicted class and confidence.
    """
    preprocessed_data = process_single_file(file_path, window_size)
    if preprocessed_data is None:
        return None, None
    # Convert numpy array to torch tensor
    # Expected shape: (num_windows, window_size, num_features)
    input_tensor = torch.tensor(preprocessed_data, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        # Apply softmax to obtain probabilities
        probabilities = F.softmax(outputs, dim=1)
    # Average predictions over windows
    avg_prediction = probabilities.mean(dim=0).cpu().numpy()
    class_names = ['AFib', 'Irregular', 'Regular']
    predicted_class = class_names[np.argmax(avg_prediction)]
    confidence = np.max(avg_prediction)
    return predicted_class, confidence

# ============================================
# Section 2: PyTorch Model Architecture (Equivalent to TensorFlow model)
# ============================================
class HealthModel(nn.Module):
    def __init__(self, input_shape=(500, 6)):
        """
        input_shape: (sequence_length, num_features)
        """
        super(HealthModel, self).__init__()
        # Our input is (batch, seq_len, features). For Conv1d, we'll transpose to (batch, channels, seq_len)
        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        
        # After two poolings, the sequence length becomes input_shape[0] // 4 (e.g., 500//4 = 125)
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(64, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        # Permute to (batch, features, seq_len) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        
        # Permute back for LSTM: (batch, seq_len_new, channels)
        x = x.permute(0, 2, 1)  # Now shape: (batch, seq_len/4, 128)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        # Use the last time step's output
        x = x[:, -1, :]
        x = self.fc1(x)
        # For BatchNorm1d on FC layers, input shape should be (batch, features)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        # No softmax here; it will be applied later in prediction
        return x

# ============================================
# Section 3: PPG Signal Analysis Functions (Unchanged)
# ============================================
def bandpass_filter_np(signal, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def lowpass_filter_np(signal, cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, signal)

def detect_beats(ppg_signal, fs, distance_sec=0.5):
    distance_samples = int(distance_sec * fs)
    peaks, _ = find_peaks(ppg_signal, distance=distance_samples)
    return peaks

def compute_heart_rate(ppg_signal, fs):
    peaks = detect_beats(ppg_signal, fs)
    if len(peaks) < 2:
        return None, peaks
    intervals = np.diff(peaks) / fs
    hr = 60 / np.mean(intervals)
    return hr, peaks

def compute_hrv_metrics(ppg_signal, fs):
    peaks = detect_beats(ppg_signal, fs)
    if len(peaks) < 2:
        return None, None
    intervals = np.diff(peaks) / fs * 1000  # in ms
    sdnn = np.std(intervals)
    rmssd = np.sqrt(np.mean(np.diff(intervals) ** 2))
    return rmssd, sdnn

def detect_arrhythmia(ppg_signal, fs, rmssd_threshold=50):
    rmssd, sdnn = compute_hrv_metrics(ppg_signal, fs)
    if rmssd is None:
        return "Insufficient Data"
    if rmssd > rmssd_threshold * 1.5:
        return "AFib"
    elif rmssd > rmssd_threshold:
        return "Irregular"
    else:
        return "Regular"

def estimate_spo2(red_signal, ir_signal, fs=100, A=110, B=25):
    ac_lowcut, ac_highcut = 0.5, 5
    dc_cutoff = 0.5
    red_ac = bandpass_filter_np(red_signal, ac_lowcut, ac_highcut, fs)
    ir_ac = bandpass_filter_np(ir_signal, ac_lowcut, ac_highcut, fs)
    red_dc = lowpass_filter_np(red_signal, dc_cutoff, fs)
    ir_dc = lowpass_filter_np(ir_signal, dc_cutoff, fs)
    red_ac_rms = np.sqrt(np.mean(red_ac ** 2))
    ir_ac_rms = np.sqrt(np.mean(ir_ac ** 2))
    red_dc_mean = np.mean(red_dc)
    ir_dc_mean = np.mean(ir_dc)
    R = (red_ac_rms / red_dc_mean) / (ir_ac_rms / ir_dc_mean)
    spo2 = A - B * R
    return spo2

def check_spo2(spo2, normal_range=(95, 100)):
    lower, upper = normal_range
    if spo2 < lower:
        return f"Low SpO₂: {spo2:.2f}%"
    elif spo2 > upper:
        return f"High SpO₂: {spo2:.2f}%"
    else:
        return f"Normal SpO₂: {spo2:.2f}%"

def extract_respiratory_rate(ppg_signal, fs, lowcut=0.1, highcut=0.5):
    resp_signal = bandpass_filter_np(ppg_signal, lowcut, highcut, fs)
    peaks, _ = find_peaks(resp_signal, distance=fs*1.5)
    num_peaks = len(peaks)
    time_span = len(ppg_signal) / fs
    rr = (num_peaks / time_span) * 60 if time_span > 0 else 0
    return rr, resp_signal, peaks

def check_respiratory_rate(rr, normal_range=(12, 20)):
    lower, upper = normal_range
    if rr < lower:
        return f"Low respiratory rate: {rr:.2f} bpm"
    elif rr > upper:
        return f"High respiratory rate: {rr:.2f} bpm"
    else:
        return f"Normal respiratory rate: {rr:.2f} bpm"

def estimate_stress(ppg_signal, fs, rmssd_threshold=50):
    rmssd, _ = compute_hrv_metrics(ppg_signal, fs)
    if rmssd is None:
        return "Insufficient data"
    if rmssd < rmssd_threshold * 0.75:
        return "High stress"
    elif rmssd < rmssd_threshold:
        return "Moderate stress"
    else:
        return "Low stress"

def estimate_sleep_quality(ppg_signal_night, fs):
    hr, _ = compute_heart_rate(ppg_signal_night, fs)
    rmssd, _ = compute_hrv_metrics(ppg_signal_night, fs)
    if hr is None or rmssd is None:
        return "Insufficient data"
    if hr < 60 and rmssd > 50:
        return "Good sleep quality"
    elif hr < 70 and rmssd > 40:
        return "Moderate sleep quality"
    else:
        return "Poor sleep quality"

def pulse_wave_analysis(ppg_signal, fs):
    peaks = detect_beats(ppg_signal, fs)
    if len(peaks) < 2:
        return None
    rise_times = []
    for i in range(1, len(peaks)):
        segment = ppg_signal[peaks[i-1]:peaks[i]]
        if len(segment) == 0:
            continue
        peak_index = np.argmax(segment)
        rise_time = peak_index / fs
        rise_times.append(rise_time)
    return np.mean(rise_times) if rise_times else None

def check_warnings(hr, rmssd, spo2, rr, arrhythmia_status, stress_level, sleep_quality):
    warnings = []
    if hr is None:
        warnings.append("Insufficient heart rate data.")
    else:
        if hr < 50:
            warnings.append(f"Warning: Heart Rate too low: {hr:.2f} bpm")
        elif hr > 100:
            warnings.append(f"Warning: Heart Rate too high: {hr:.2f} bpm")
    
    if rmssd is None:
        warnings.append("Insufficient HRV data.")
    else:
        if rmssd < 30:
            warnings.append(f"Warning: Low HRV (RMSSD: {rmssd:.2f} ms) may indicate high stress.")
        elif rmssd > 100:
            warnings.append(f"Warning: High HRV (RMSSD: {rmssd:.2f} ms) - unusual variability.")
    
    if spo2 < 95:
        warnings.append(f"Warning: Low SpO₂: {spo2:.2f}%")
    elif spo2 > 100:
        warnings.append(f"Warning: High SpO₂: {spo2:.2f}%")
    
    if rr < 12:
        warnings.append(f"Warning: Low Respiratory Rate: {rr:.2f} bpm")
    elif rr > 20:
        warnings.append(f"Warning: High Respiratory Rate: {rr:.2f} bpm")
    
    if arrhythmia_status != "Regular":
        warnings.append(f"Warning: Abnormal heart rhythm detected: {arrhythmia_status}")
    
    if stress_level == "High stress":
        warnings.append("Warning: High stress level detected.")
    
    if sleep_quality == "Poor sleep quality":
        warnings.append("Warning: Poor sleep quality detected.")
    
    return warnings

# ============================================
# Section 4: Main Functionality
# ============================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ---- Part A: PyTorch Model Prediction from CSV File ----
    model_path = 'best_model.keras'
    if os.path.exists(model_path):
        # Instantiate the model architecture and load weights
        model = HealthModel(input_shape=(500, 6)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        predicted_class, confidence = predict_heart_rhythm(model, "/Users/adarshashok/Downloads/treehacks/healthbot/src/test_1.csv", window_size=500, device=device)
        print("Model Prediction:")
        print(f"Heart Rhythm: {predicted_class}, Confidence: {confidence:.2f}")
    else:
        print(f"Model file '{model_path}' not found. Skipping model prediction.")

    # ---- Part B: Simulate a PPG Signal and Compute Health Metrics ----
    fs = 100  # Sampling frequency in Hz
    duration = 60  # seconds
    t = np.linspace(0, duration, fs * duration)
    
    # Simulate a PPG signal:
    heartbeat = 100 * np.sin(2 * np.pi * 1.2 * t)           # ~72 BPM heartbeat
    respiratory = 10 * np.sin(2 * np.pi * 0.25 * t)           # ~15 breaths/min respiratory modulation
    baseline = 1000
    noise = 5 * np.random.randn(len(t))
    ppg_signal = baseline + heartbeat + respiratory + noise
    
    # For SpO₂: simulate red and IR signals (with slight phase shift and noise)
    red_signal = baseline + 100 * np.sin(2 * np.pi * 1.2 * t) + 50 * np.random.randn(len(t))
    ir_signal  = baseline + 100 * np.sin(2 * np.pi * 1.2 * t + 0.1) + 50 * np.random.randn(len(t))
    
    # Compute Health Metrics:
    hr, beat_peaks = compute_heart_rate(ppg_signal, fs)
    rmssd, sdnn = compute_hrv_metrics(ppg_signal, fs)
    arrhythmia_status = detect_arrhythmia(ppg_signal, fs)
    spo2 = estimate_spo2(red_signal, ir_signal, fs)
    spo2_status = check_spo2(spo2)
    rr, resp_component, resp_peaks = extract_respiratory_rate(ppg_signal, fs)
    rr_status = check_respiratory_rate(rr)
    stress_level = estimate_stress(ppg_signal, fs)
    sleep_quality = estimate_sleep_quality(ppg_signal, fs)  # using same signal for simulation
    systolic_rise_time = pulse_wave_analysis(ppg_signal, fs)
    
    # Print Simulated Metrics:
    print("\nSimulated Health Metrics:")
    if hr is not None:
        print(f"Heart Rate: {hr:.2f} bpm")
    else:
        print("Heart Rate: Insufficient data")
    print(f"HRV - RMSSD: {rmssd:.2f} ms, SDNN: {sdnn:.2f} ms")
    print(f"Arrhythmia Status: {arrhythmia_status}")
    print(f"SpO₂: {spo2:.2f}%, Status: {spo2_status}")
    print(f"Respiratory Rate: {rr:.2f} bpm, Status: {rr_status}")
    print(f"Estimated Stress Level: {stress_level}")
    print(f"Sleep Quality: {sleep_quality}")
    if systolic_rise_time is not None:
        print(f"Average Systolic Rise Time: {systolic_rise_time:.3f} seconds")
    else:
        print("Insufficient data for pulse wave analysis.")
    
    # Check and print warnings for abnormal metrics:
    warnings = check_warnings(hr, rmssd, spo2, rr, arrhythmia_status, stress_level, sleep_quality)
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(" - " + w)
    else:
        print("\nNo abnormal metrics detected.")
    
    # Plot the simulated signals:
    plt.figure(figsize=(14, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(t, ppg_signal, label="Simulated PPG Signal")
    plt.scatter(t[beat_peaks], ppg_signal[beat_peaks], color='red', label="Detected Beats")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("PPG Signal with Detected Heart Beats")
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(t, resp_component, label="Respiratory Component", color='green')
    plt.scatter(t[resp_peaks], resp_component[resp_peaks], color='orange', label="Respiratory Peaks")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Respiratory Rate Extraction from PPG Signal")
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(t, red_signal, label="Simulated Red PPG", alpha=0.7)
    plt.plot(t, ir_signal, label="Simulated IR PPG", alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("PPG Signals for SpO₂ Estimation")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()