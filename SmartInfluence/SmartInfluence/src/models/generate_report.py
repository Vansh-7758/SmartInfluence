import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# Setup paths to ensure we can import the model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, 'src'))

print("Loading model and calculating metrics. This might take a few seconds...")
# Import metrics and data from the model script
from models.xgboost_model import (
    test_accuracy, cv_mean, cv_std, 
    conf_matrix, feat_importances, target_classes, 
    FEATURES, class_dist, top_feature_names, corr_matrix
)

# Output path
out_dir = os.path.join(BASE_DIR, 'outputs', 'reports')
os.makedirs(out_dir, exist_ok=True)
pdf_path = os.path.join(out_dir, 'xgboost_analysis_report.pdf')

print(f"Generating PDF report at: {pdf_path}")

with PdfPages(pdf_path) as pdf:
    
    # ---------------------------------------------------------
    # Page 1: Title and Summary Metrics
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    
    title_text = "SmartInfluence\nXGBoost Model Analysis Report"
    ax.text(0.5, 0.9, title_text, ha='center', va='center', fontsize=20, fontweight='bold', color='#1e3a8a')
    
    metrics_text = (
        f"Model Performance Metrics:\n\n"
        f"Test Set Accuracy : {test_accuracy * 100:.2f}%\n"
        f"Cross-Val Mean    : {cv_mean * 100:.2f}%\n"
        f"Cross-Val Std Dev : {cv_std * 100:.2f}%\n\n"
        f"Target Classes:\n"
        f" - High Performer (Sales > $10K)\n"
        f" - Mid Performer  (Sales $1K - $10K)\n"
        f" - Low Performer  (Sales < $1K)\n"
    )
    
    ax.text(0.1, 0.6, metrics_text, ha='left', va='top', fontsize=14, family='monospace')
    pdf.savefig(fig)
    plt.close()

    # ---------------------------------------------------------
    # Page 2: Class Distribution
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = list(class_dist.keys())
    sizes = list(class_dist.values())
    colors = ['#dc2626', '#d97706', '#059669'] # Red, Orange, Green
    
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 12})
    ax.axis('equal')
    plt.title('Training Data: Sales Class Distribution', fontsize=16, pad=20)
    pdf.savefig(fig)
    plt.close()

    # ---------------------------------------------------------
    # Page 3: Confusion Matrix
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_classes, yticklabels=target_classes, ax=ax)
    plt.title('Confusion Matrix (Test Data)', fontsize=16, pad=20)
    plt.ylabel('Actual Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # ---------------------------------------------------------
    # Page 4: Feature Importances
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sort features
    indices = np.argsort(feat_importances)
    sorted_features = [FEATURES[i] for i in indices]
    sorted_importances = [feat_importances[i] for i in indices]
    
    # Take top 15
    sorted_features = sorted_features[-15:]
    sorted_importances = sorted_importances[-15:]
    
    # Clean names
    clean_features = [f.replace('INSTAGRAM_', '') for f in sorted_features]
    
    ax.barh(clean_features, sorted_importances, color='#2563eb')
    plt.title('Top 15 Feature Importances', fontsize=16, pad=20)
    plt.xlabel('Relative Importance', fontsize=12)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

    # ---------------------------------------------------------
    # Page 5: Top Features Correlation Heatmap
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    clean_top_features = [f.replace('INSTAGRAM_', '') for f in top_feature_names]
    
    # Convert back to numpy array for seaborn
    corr_array = np.array(corr_matrix)
    
    # Mask upper triangle for cleaner look
    mask = np.triu(np.ones_like(corr_array, dtype=bool))
    
    sns.heatmap(corr_array, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                xticklabels=clean_top_features, yticklabels=clean_top_features, 
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    
    plt.title('Correlation Matrix (Top Features)', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print("Report successfully generated!")
