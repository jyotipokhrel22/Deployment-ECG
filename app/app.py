from flask import Flask, render_template, request, jsonify
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.model_service import ModelService 
from app.visualization import create_visualization_html
# from visualization import create_visualization
import numpy as np
import json
from datetime import datetime


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
# app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change in production

# Initialize model service
print("Loading model service...")
model_service = ModelService(
    model_path='models/best_model.pth',
    metadata_path='data/processed_data/metadata.pkl'
)
print(" Model service loaded successfully!")

# Class information for UI
CLASS_INFO = {
    'N': {
        'name': 'Normal Sinus Rhythm',
        'description': 'Normal heartbeat with regular rhythm and no abnormalities detected.',
        'color': '#28a745',
        'icon': '',
        'recommendation': 'No immediate action required. Continue routine cardiac monitoring.',
        'severity': 'Low',
        'clinical_notes': 'P wave, QRS complex, and T wave are within normal limits. No arrhythmia detected.'
    },
    'S': {
        'name': 'Supraventricular Ectopic Beat',
        'description': 'Premature beat originating from the upper chambers of the heart (atria).',
        'color': '#ffc107',
        'icon': '⚠️',
        'recommendation': 'Monitor frequency. If symptoms persist or worsen, consult a cardiologist.',
        'severity': 'Medium',
        'clinical_notes': 'Early P wave with abnormal morphology. May cause palpitations. Usually benign if infrequent.'
    },
    'V': {
        'name': 'Ventricular Ectopic Beat',
        'description': 'Premature beat originating from the lower chambers of the heart (ventricles).',
        'color': '#fd7e14',
        'icon': '⚠️',
        'recommendation': 'Medical evaluation recommended. Frequent VEBs may require treatment.',
        'severity': 'Medium-High',
        'clinical_notes': 'Wide QRS complex (>0.12s) without preceding P wave. Requires follow-up if frequent or symptomatic.'
    },
    'F': {
        'name': 'Fusion Beat',
        'description': 'Combined beat resulting from simultaneous normal and ventricular activation.',
        'color': '#e83e8c',
        'icon': '🔴',
        'recommendation': 'Specialist consultation advised. Requires detailed cardiac evaluation.',
        'severity': 'High',
        'clinical_notes': 'QRS morphology shows characteristics of both normal and ventricular beats. Indicates ventricular irritability.'
    },
    'Q': {
        'name': 'Unknown/Paced Beat',
        'description': 'Unclassifiable beat or pacemaker-induced cardiac activity.',
        'color': '#6c757d',
        'icon': '❓',
        'recommendation': 'Further analysis needed. Check pacemaker function if applicable.',
        'severity': 'Variable',
        'clinical_notes': 'Cannot definitively classify. May indicate pacemaker artifact or noise. Manual review recommended.'
    }
}

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', class_info=CLASS_INFO)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle ECG prediction request"""
    try:
        ecg_data = None
        
        # Check if file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Read CSV file
            try:
                content = file.read().decode('utf-8')
                lines = content.strip().split('\n')
                ecg_data = []
                for line in lines:
                    try:
                        value = float(line.strip().split(',')[0])
                        ecg_data.append(value)
                    except:
                        continue
                ecg_data = np.array(ecg_data)
            except Exception as e:
                return jsonify({'error': f'Error reading CSV file: {str(e)}'}), 400
        
        # Check if JSON data
        elif request.is_json:
            data = request.get_json()
            if 'ecg_data' in data:
                ecg_data = np.array(data['ecg_data'])
        
        # Check if form data
        elif 'ecg_data' in request.form:
            ecg_data = np.array(json.loads(request.form['ecg_data']))
        
        if ecg_data is None:
            return jsonify({'error': 'No ECG data provided'}), 400
        
        # Validate and adjust length
        if len(ecg_data) < 360:
            ecg_data = np.pad(ecg_data, (0, 360 - len(ecg_data)), mode='constant')
        elif len(ecg_data) > 360:
            ecg_data = ecg_data[:360]
        
        # Get patient info from request
        # patient_info = {
        #     'id': request.form.get('patient_id', 'N/A'),
        #     'age': request.form.get('patient_age', 'N/A'),
        #     'gender': request.form.get('patient_gender', 'N/A')
        # }
        
        # Make prediction
        result = model_service.predict(ecg_data)
        
        if result is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Create visualization
        visualization_html = create_visualization_html(
            original=result['original'],
            reconstructed=result['reconstructed'],
            clinical_attention=result['clinical_attention'],
            predicted_class=result['class'],
            class_info=CLASS_INFO
        )
        
        # Get class information
        class_info = CLASS_INFO[result['class']]
        
        # Prepare response
        response = {
            'prediction': result['class'],
            'class_name': class_info['name'],
            'confidence': result['confidence'] * 100,
            'description': class_info['description'],
            'recommendation': class_info['recommendation'],
            'clinical_notes': class_info['clinical_notes'],
            'severity': class_info['severity'],
            'color': class_info['color'],
            'icon': class_info['icon'],
            'all_probabilities': {k: v*100 for k, v in result['all_probabilities'].items()},
            'visualization': visualization_html,
            'reconstruction_error': float(result['reconstruction_error']),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'attention_stats': {
                'clinical_attention_max': float(np.max(result['clinical_attention'])),
                'clinical_attention_mean': float(np.mean(result['clinical_attention'])),
                'temporal_attention_max': float(np.max(result['temporal_attention'])),
                'temporal_attention_mean': float(np.mean(result['temporal_attention']))
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        print(f"Error in prediction: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/demo', methods=['GET'])
def demo():
    """Load demo ECG data (from CSV files if available, else synthetic fallback)."""
    try:
        base_path = 'data/test_datasets/csv_ecg_signal'
        csv_files = {
            'fusion': os.path.join(base_path, 'fusion.csv'),
            'pvc': os.path.join(base_path, 'pvc.csv'),
            'normal': os.path.join(base_path, 'normal.csv'),
            'supra': os.path.join(base_path, 'supra.csv'),
        }

        ecg_data = {}
        messages = []

        # ✅ Load first 4 ECGs from CSV files if available
        csv_available = all(os.path.exists(p) for p in csv_files.values())

        if csv_available:
            for name, path in csv_files.items():
                try:
                    df = pd.read_csv(path, header=None)
                    # Flatten and convert to list (1D ECG signal)
                    ecg_signal = df.values.flatten().tolist()
                    ecg_data[name] = ecg_signal
                    messages.append(f'Loaded ECG from {name}.csv ({len(ecg_signal)} samples)')
                except Exception as e:
                    messages.append(f'Failed to load {name}.csv: {e}')
        else:
            # ⚙️ Fallback to synthetic data generation
            t = np.linspace(0, 1, 360)

            normal_ecg = (
                np.sin(2 * np.pi * 1.2 * t) +
                0.3 * np.sin(2 * np.pi * 2.4 * t) +
                0.5 * np.sin(2 * np.pi * 60 * t) * np.exp(-((t - 0.3)**2) / 0.01) +
                0.05 * np.random.randn(360)
            )

            pvc = (
                np.sin(2 * np.pi * 1.0 * t) +
                0.8 * np.sin(2 * np.pi * 45 * t) * np.exp(-((t - 0.25)**2) / 0.005) +
                0.2 * np.random.randn(360)
            )

            fusion = (
                np.sin(2 * np.pi * 1.5 * t) +
                0.5 * np.sin(2 * np.pi * 3 * t) +
                0.5 * np.sin(2 * np.pi * 70 * t) * np.exp(-((t - 0.45)**2) / 0.008) +
                0.1 * np.random.randn(360)
            )

            supra = (
                np.sin(2 * np.pi * 1.2 * t + 0.5 * np.random.randn(360)) +
                0.2 * np.random.randn(360)
            )

            ecg_data = {
                'fusion': fusion.tolist(),
                'pvc': pvc.tolist(),
                'normal': normal_ecg.tolist(),
                'supra': supra.tolist()
            }

            messages = [
                'Synthetic fusion ECG generated',
                'Synthetic PVC ECG generated',
                'Synthetic normal ECG generated',
                'Synthetic supra ECG generated'
            ]

        return jsonify({
            'ecg_data': ecg_data,
            'messages': messages
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_service.model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        info = model_service.get_model_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    print("=" * 80)
    print("🫀 CARDIOAI ECG CLASSIFICATION WEB APP")
    print("=" * 80)
    print(f"\n Model loaded: {model_service.model is not None}")
    print(f" Classes: {model_service.class_names}")
    print(f"\n Starting Flask server...")
    print(f" Open your browser and go to: http://localhost:5000")
    print("=" * 80)
    
    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000)