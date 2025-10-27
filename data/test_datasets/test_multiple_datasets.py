import sys
import os

# ===== Add project root to sys.path =====
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ===== Imports =====
from app.model_service import ModelService
from load_ptbxl_data import load_ptbxl_dataset
from sklearn.metrics import classification_report

# ===== Paths =====
model_path = os.path.join(PROJECT_ROOT, 'models', 'best_model.pth')
metadata_path = os.path.join(PROJECT_ROOT, 'data', 'processed_data', 'metadata.pkl')  # updated path

# ===== Initialize model service =====
model_service = ModelService(model_path=model_path, metadata_path=metadata_path)

# ===== Load ECG data =====
X, y_true = load_ptbxl_dataset()

# ===== Make predictions =====
y_pred = []
for ecg in X:
    result = model_service.predict(ecg)
    y_pred.append(result['class'])

# ===== Print classification report =====
print(classification_report(y_true, y_pred))
