import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

def create_visualization_html(original, reconstructed, clinical_attention, 
                              predicted_class, class_info):
    """
    Create comprehensive visualization of ECG with attention maps
    
    Parameters:
    -----------
    original: np.array
        Original preprocessed ECG signal
    reconstructed: np.array
        Reconstructed ECG signal from model
    clinical_attention: np.array
        Clinical attention weights
    predicted_class: str
        Predicted class label
    class_info: dict
        Dictionary containing class information
    
    Returns:
    --------
    str: Base64 encoded image
    """
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8)) 
    
    time_axis = np.arange(len(original)) / 360  # Convert to seconds
    
    # Get class color
    class_color = class_info[predicted_class]['color']
    
    # 1. Original ECG Signal
    ax1 = axes[0]
    ax1.plot(time_axis, original, linewidth=2, color='#2c3e50', label='Original ECG')
    ax1.set_title('Original ECG Signal', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel('Amplitude (normalized)', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.set_xlim([0, time_axis[-1]])
 
    
    # Add subtle background color based on prediction
    ax1.axhspan(ax1.get_ylim()[0], ax1.get_ylim()[1], 
                alpha=0.05, color=class_color, zorder=0)
    
    # 2. Reconstructed ECG Signal
    ax2 = axes[0]
    ax2.plot(time_axis, original, linewidth=2, alpha=0.6, 
            color='#2c3e50', linestyle='--', label='Original')
    ax2.plot(time_axis, reconstructed, linewidth=2, 
            color='#e74c3c', label='Reconstructed')
    
    # Calculate and display MSE
    mse = np.mean((original - reconstructed)**2)
    ax2.set_title(f'Reconstructed ECG Signal (MSE: {mse:.6f})', 
                 fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylabel('Amplitude (normalized)', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.set_xlim([0, time_axis[-1]])
    
    # 3. ECG with Clinical Attention Overlay
    ax3 = axes[1]
    
    # Upsample clinical attention to match signal length
    clinical_att_avg = np.mean(clinical_attention, axis=0)
    clinical_att_upsampled = np.interp(
        np.linspace(0, len(clinical_att_avg)-1, len(original)),
        np.arange(len(clinical_att_avg)),
        clinical_att_avg
    )
    
    # Normalize attention for better visualization
    clinical_att_normalized = (clinical_att_upsampled - clinical_att_upsampled.min()) / \
                             (clinical_att_upsampled.max() - clinical_att_upsampled.min() + 1e-8)
    
    # Plot ECG
    ax3.plot(time_axis, original, linewidth=2, color='#2c3e50', 
            label='ECG Signal', zorder=2)
    
    # Create twin axis for attention
    ax3_twin = ax3.twinx()
    ax3_twin.fill_between(time_axis, clinical_att_normalized, 
                          alpha=0.4, color='#3498db', label='Clinical Attention',
                          zorder=1)
    ax3_twin.set_ylabel('Attention Weight', fontsize=12, color='#3498db')
    ax3_twin.tick_params(axis='y', labelcolor='#3498db')
    ax3_twin.set_ylim([0, 1.2])
    
    ax3.set_title('ECG Signal with Clinical Attention Overlay', 
                 fontsize=14, fontweight='bold', pad=15)
    ax3.set_xlabel('Time (seconds)', fontsize=12)
    ax3.set_ylabel('Amplitude (normalized)', fontsize=12)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='upper left', fontsize=11)
    ax3.set_xlim([0, time_axis[-1]])
    
    # Add prediction info as text box
    prediction_text = f"Prediction: {class_info[predicted_class]['name']}"
    ax3.text(0.02, 0.98, prediction_text,
            transform=ax3.transAxes,
            fontsize=12,
            fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=class_color, alpha=0.3))
    
    # Highlight regions with high attention
    threshold = 0.7
    high_attention_regions = clinical_att_normalized > threshold
    for i in range(len(high_attention_regions) - 1):
        if high_attention_regions[i]:
            ax3.axvline(x=time_axis[i], color='red', alpha=0.1, linewidth=0.5)
    
    # Overall figure styling
    fig.suptitle(f'ECG Analysis Report - {class_info[predicted_class]["name"]}',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_base64


def create_attention_heatmap(clinical_attention, temporal_attention):
    """
    Create attention heatmap visualizations
    
    Parameters:
    -----------
    clinical_attention: np.array
        Clinical attention weights [channels, time]
    temporal_attention: np.array
        Temporal attention weights [time, time]
    
    Returns:
    --------
    str: Base64 encoded image
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Clinical Attention Heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(clinical_attention, aspect='auto', cmap='hot', 
                     interpolation='bilinear')
    ax1.set_title('Clinical Attention\n(Feature Channel Importance)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax1.set_xlabel('Temporal Position', fontsize=11)
    ax1.set_ylabel('Feature Channel', fontsize=11)
    cbar1 = plt.colorbar(im1, ax=ax1, label='Attention Weight')
    cbar1.ax.tick_params(labelsize=10)
    
    # Add grid
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Temporal Attention Heatmap
    ax2 = axes[1]
    im2 = ax2.imshow(temporal_attention, aspect='auto', cmap='viridis',
                     interpolation='bilinear')
    ax2.set_title('Temporal Attention\n(Time Dependencies)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax2.set_xlabel('Key Position', fontsize=11)
    ax2.set_ylabel('Query Position', fontsize=11)
    cbar2 = plt.colorbar(im2, ax=ax2, label='Attention Weight')
    cbar2.ax.tick_params(labelsize=10)
    
    # Add grid
    ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_base64


def create_probability_bar_chart(probabilities, class_info):
    """
    Create bar chart for class probabilities
    
    Parameters:
    -----------
    probabilities: dict
        Dictionary of class names and their probabilities
    class_info: dict
        Dictionary containing class information
    
    Returns:
    --------
    str: Base64 encoded image
    """
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    classes = list(probabilities.keys())
    probs = [probabilities[c] * 100 for c in classes]
    colors = [class_info[c]['color'] for c in classes]
    
    bars = ax.barh(classes, probs, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
               f'{prob:.2f}%',
               ha='left', va='center', fontweight='bold', fontsize=11)
    
    ax.set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Arrhythmia Class', fontsize=12, fontweight='bold')
    ax.set_title('Class Probability Distribution', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim([0, 105])
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # Add class names as y-labels
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([f"{c} - {class_info[c]['name']}" for c in classes])
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_base64


def create_comparison_plot(original, reconstructed):
    """
    Create side-by-side comparison of original vs reconstructed
    
    Parameters:
    -----------
    original: np.array
        Original ECG signal
    reconstructed: np.array
        Reconstructed ECG signal
    
    Returns:
    --------
    str: Base64 encoded image
    """
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    time_axis = np.arange(len(original)) / 360
    
    # Original
    axes[0].plot(time_axis, original, linewidth=2, color='#2c3e50')
    axes[0].set_title('Original ECG Signal', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Amplitude', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, time_axis[-1]])
    
    # Reconstructed
    axes[1].plot(time_axis, reconstructed, linewidth=2, color='#e74c3c')
    axes[1].set_title('Reconstructed ECG Signal', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Time (seconds)', fontsize=11)
    axes[1].set_ylabel('Amplitude', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, time_axis[-1]])
    
    # Calculate and display reconstruction quality
    mse = np.mean((original - reconstructed)**2)
    correlation = np.corrcoef(original, reconstructed)[0, 1]
    
    fig.text(0.5, 0.02, 
            f'Reconstruction Quality - MSE: {mse:.6f} | Correlation: {correlation:.4f}',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_base64


def create_attention_overlay_plot(original, clinical_attention):
    """
    Create plot showing where model focused attention
    
    Parameters:
    -----------
    original: np.array
        Original ECG signal
    clinical_attention: np.array
        Clinical attention weights
    
    Returns:
    --------
    str: Base64 encoded image
    """
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    time_axis = np.arange(len(original)) / 360
    
    # Upsample attention
    clinical_att_avg = np.mean(clinical_attention, axis=0)
    clinical_att_upsampled = np.interp(
        np.linspace(0, len(clinical_att_avg)-1, len(original)),
        np.arange(len(clinical_att_avg)),
        clinical_att_avg
    )
    
    # Normalize
    clinical_att_normalized = (clinical_att_upsampled - clinical_att_upsampled.min()) / \
                             (clinical_att_upsampled.max() - clinical_att_upsampled.min() + 1e-8)
    
    # Create color map based on attention
    colors = plt.cm.RdYlGn_r(clinical_att_normalized)
    
    # Plot ECG with colored segments based on attention
    for i in range(len(original) - 1):
        ax.plot(time_axis[i:i+2], original[i:i+2], 
               color=colors[i], linewidth=3, alpha=0.8)
    
    ax.set_title('ECG Signal Colored by Attention Intensity\n(Red = High Attention, Green = Low Attention)',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Amplitude (normalized)', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, time_axis[-1]])
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, 
                               norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Attention Weight')
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_base64


def create_comprehensive_report(result, class_info):
    """
    Create comprehensive visualization report
    
    Parameters:
    -----------
    result: dict
        Prediction result from model_service
    class_info: dict
        Dictionary containing class information
    
    Returns:
    --------
    dict: Dictionary of base64 encoded images
    """
    
    visualizations = {}
    
    # 1. Main ECG analysis
    visualizations['main_analysis'] = create_visualization_html(
        result['original'],
        result['reconstructed'],
        result['clinical_attention'],
        result['class'],
        class_info
    )
    
    # 2. Attention heatmaps
    visualizations['attention_heatmaps'] = create_attention_heatmap(
        result['clinical_attention'],
        result['temporal_attention']
    )
    
    # 3. Probability distribution
    visualizations['probability_chart'] = create_probability_bar_chart(
        result['all_probabilities'],
        class_info
    )
    
    # 4. Comparison plot
    visualizations['comparison'] = create_comparison_plot(
        result['original'],
        result['reconstructed']
    )
    
    # 5. Attention overlay
    visualizations['attention_overlay'] = create_attention_overlay_plot(
        result['original'],
        result['clinical_attention']
    )
    
    return visualizations