import json
import pandas as pd

# Load evaluations
with open('evaluation/model_evaluations.json', 'r') as f:
    evaluations = json.load(f)

print("="*80)
print("DETAILED MODEL EVALUATION METRICS")
print("="*80)

# Detailed Metrics Table
detailed_data = []
for model_name, eval_data in evaluations.items():
    detailed_data.append({
        'Model': model_name,
        'Accuracy': f"{eval_data['accuracy']:.4f}",
        'Precision': f"{eval_data['precision']:.4f}",
        'Recall': f"{eval_data['recall']:.4f}",
        'F1-Score': f"{eval_data['f1_score']:.4f}",
        'ROC-AUC': f"{eval_data['roc_auc']:.4f}"
    })

detailed_df = pd.DataFrame(detailed_data)
print("\n" + detailed_df.to_string(index=False))

# Confusion Matrices
print("\n" + "="*80)
print("CONFUSION MATRICES")
print("="*80)

for model_name, eval_data in evaluations.items():
    cm = eval_data['confusion_matrix']
    print(f"\n{model_name}:")
    print(f"[[{cm[0][0]:>4}  {cm[0][1]:>4}]")  # TN, FP
    print(f" [{cm[1][0]:>4}  {cm[1][1]:>4}]]") # FN, TP
    print(f"   Actual: [Not Fraud, Fraud]")
    print(f"Predicted: [Not Fraud, Fraud]")

# Best Model Identification
print("\n" + "="*80)
print("BEST MODEL IDENTIFICATION")
print("="*80)

best_f1 = max(evaluations.items(), key=lambda x: x[1]['f1_score'])
best_auc = max(evaluations.items(), key=lambda x: x[1]['roc_auc'])
best_precision = max(evaluations.items(), key=lambda x: x[1]['precision'])
best_recall = max(evaluations.items(), key=lambda x: x[1]['recall'])

print(f"\n🏆 BEST MODEL BY F1-Score: {best_f1[0]} (F1 = {best_f1[1]['f1_score']:.4f})")
print(f"🏆 BEST MODEL BY ROC-AUC: {best_auc[0]} (AUC = {best_auc[1]['roc_auc']:.4f})")
print(f"🏆 BEST MODEL BY Precision: {best_precision[0]} (Precision = {best_precision[1]['precision']:.4f})")
print(f"🏆 BEST MODEL BY Recall: {best_recall[0]} (Recall = {best_recall[1]['recall']:.4f})")

# Performance Summary
print("\n" + "="*80)
print("PERFORMANCE SUMMARY")
print("="*80)

print(f"\nTotal Models Evaluated: {len(evaluations)}")
print(f"Dataset Type: {'Highly Imbalanced' if any(eval_data['roc_auc'] > 0.9 for eval_data in evaluations.values()) else 'Balanced'}")
print(f"Best Overall Model: {best_f1[0]} (based on F1-Score for imbalanced data)")

# Model Rankings
print("\n" + "="*80)
print("MODEL RANKINGS")
print("="*80)

# Rank by F1-Score (most important for imbalanced data)
ranked_by_f1 = sorted(evaluations.items(), key=lambda x: x[1]['f1_score'], reverse=True)
print("\nRanked by F1-Score (Best for Imbalanced Data):")
for i, (model_name, metrics) in enumerate(ranked_by_f1, 1):
    print(f"{i}. {model_name}: {metrics['f1_score']:.4f}")

# Rank by ROC-AUC
ranked_by_auc = sorted(evaluations.items(), key=lambda x: x[1]['roc_auc'], reverse=True)
print("\nRanked by ROC-AUC:")
for i, (model_name, metrics) in enumerate(ranked_by_auc, 1):
    print(f"{i}. {model_name}: {metrics['roc_auc']:.4f}")