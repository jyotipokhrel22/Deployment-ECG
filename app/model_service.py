"""
Handles model loading, preprocessing, and predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from scipy import signal as scipy_signal
import sys
import os

# MODEL ARCHITECTURE 


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, 
                               stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7,
                               stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetEncoder(nn.Module):
    def __init__(self, num_blocks=[2, 2, 2, 2]):
        super(ResNetEncoder, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

class ClinicalAttention(nn.Module):
    def __init__(self, in_channels):
        super(ClinicalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        att_weights = self.attention(x)
        return x * att_weights, att_weights

class TemporalAttention(nn.Module):
    def __init__(self, in_channels):
        super(TemporalAttention, self).__init__()
        self.query = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        batch_size, C, length = x.size()
        proj_query = self.query(x).view(batch_size, -1, length).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, length)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value(x).view(batch_size, -1, length)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, length)
        out = self.gamma * out + x
        return out, attention

class GuidedAttention(nn.Module):
    def __init__(self, in_channels):
        super(GuidedAttention, self).__init__()
        self.clinical_attention = ClinicalAttention(in_channels)
        self.temporal_attention = TemporalAttention(in_channels)
        
    def forward(self, x):
        clinical_out, clinical_weights = self.clinical_attention(x)
        temporal_out, temporal_weights = self.temporal_attention(clinical_out)
        return temporal_out, clinical_weights, temporal_weights

class ECGReconstructor(nn.Module):
    def __init__(self, in_channels=512, target_length=360):
        super(ECGReconstructor, self).__init__()
        self.target_length = target_length
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),
        )
    
    def forward(self, x):
        out = self.decoder(x)
        if out.size(2) != self.target_length:
            out = F.interpolate(out, size=self.target_length, mode='linear', align_corners=False)
        return out

class MultiAttentionECGModel(nn.Module):
    def __init__(self, num_classes=5, input_length=360):
        super(MultiAttentionECGModel, self).__init__()
        self.encoder = ResNetEncoder()
        self.guided_attention = GuidedAttention(in_channels=512)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        self.reconstructor = ECGReconstructor(in_channels=512, target_length=input_length)
    
    def forward(self, x):
        features = self.encoder(x)
        attended_features, clinical_att, temporal_att = self.guided_attention(features)
        pooled = self.global_pool(attended_features).squeeze(-1)
        class_output = self.classifier(pooled)
        reconstructed = self.reconstructor(attended_features)
        return class_output, reconstructed, clinical_att, temporal_att

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def butter_bandpass_filter(data, lowcut=0.5, highcut=50, fs=360, order=4):
    """Apply Butterworth bandpass filter"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = scipy_signal.butter(order, [low, high], btype='band')
    return scipy_signal.filtfilt(b, a, data)

def notch_filter(data, freq=60, fs=360, quality=30):
    """Apply notch filter for powerline interference"""
    nyquist = 0.5 * fs
    freq_normalized = freq / nyquist
    b, a = scipy_signal.iirnotch(freq_normalized, quality)
    return scipy_signal.filtfilt(b, a, data)

def preprocess_ecg(raw_signal):
    """Complete ECG signal preprocessing"""
    # Apply filters
    filtered = butter_bandpass_filter(raw_signal)
    filtered = notch_filter(filtered)
    
    # Normalize
    mean = np.mean(filtered)
    std = np.std(filtered)
    normalized = (filtered - mean) / (std + 1e-8)
    
    return normalized

# ============================================================================
# MODEL SERVICE CLASS
# ============================================================================

class ModelService:
    """
    Service class for model loading and predictions
    """
    def __init__(self, model_path, metadata_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load metadata
        try:
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            self.scaler = self.metadata['scaler']
            self.class_names = self.metadata['class_names']
            print(f"✅ Metadata loaded: {self.class_names}")
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
            raise
        
        # Load model
        try:
            # checkpoint = torch.load(model_path, map_location=self.device)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            num_classes = len(self.class_names)
            
            self.model = MultiAttentionECGModel(
                num_classes=num_classes,
                input_length=360
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"✅ Model loaded successfully!")
            print(f"   Classes: {num_classes}")
            print(f"   Validation Accuracy: {checkpoint.get('val_acc', 'N/A')}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict(self, ecg_signal):
        """
        Make prediction on ECG signal
        
        Parameters:
        -----------
        ecg_signal: np.array
            Raw ECG signal (360 samples)
        
        Returns:
        --------
        dict with prediction results and attention maps
        """
        try:
            # Preprocess
            preprocessed = preprocess_ecg(ecg_signal)

            # Convert to 1D float array first
            preprocessed = np.asarray(preprocessed, dtype=np.float32).flatten()

            if len(preprocessed) != 360:
                preprocessed = np.interp(
                np.linspace(0, len(preprocessed) - 1, 360),
                np.arange(len(preprocessed)),
                preprocessed
            )
            
            # Scale
            scaled = self.scaler.transform(preprocessed.reshape(1, -1))
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(scaled).unsqueeze(1).to(self.device)
            
            # Predict
            with torch.no_grad():
                class_output, reconstructed, clinical_att, temporal_att = self.model(input_tensor)
                probabilities = F.softmax(class_output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
            
            # Get class name
            class_name = self.class_names[predicted_class]
            
            # Get all probabilities
            all_probs = {
                self.class_names[i]: probabilities[0, i].item() 
                for i in range(len(self.class_names))
            }
            
            # Calculate reconstruction error
            reconstructed_signal = reconstructed.cpu().numpy()[0, 0, :]
            reconstruction_error = np.mean((scaled[0] - reconstructed_signal)**2)
            
            return {
                'class': class_name,
                'confidence': confidence,
                'all_probabilities': all_probs,
                'reconstructed': reconstructed_signal,
                'clinical_attention': clinical_att.cpu().numpy()[0, :, :],
                'temporal_attention': temporal_att.cpu().numpy()[0, :, :],
                'original': scaled[0],
                'preprocessed': preprocessed,
                'reconstruction_error': reconstruction_error
            }
        
        except Exception as e:
            print(f" Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_model_info(self):
        """Get model information"""
        return {
            'num_classes': len(self.class_names),
            'class_names': self.class_names.tolist(),
            'input_length': 360,
            'sampling_rate': self.metadata.get('sampling_rate', 360),
            'window_size': self.metadata.get('window_size', 360),
            'device': str(self.device),
            'model_parameters': sum(p.numel() for p in self.model.parameters())
        }
    
    def batch_predict(self, ecg_signals):
        """
        Predict on multiple ECG signals
        
        Parameters:
        -----------
        ecg_signals: list of np.array
            List of raw ECG signals
        
        Returns:
        --------
        list of prediction dicts
        """
        results = []
        for ecg in ecg_signals:
            result = self.predict(ecg)
            if result:
                results.append(result)
        return results