"""
Trial 3: Fit Classification
Target: Existing Fit_Classification column (High Fit / Moderate Fit / Low Fit)

Uses the KMeans-derived cluster labels as supervised classification target.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data_loader import load_data, engineer_features, create_target_fit_classification, prepare_data
from common.evaluation import run_all_models


def main():
    print("\n" + "="*70)
    print("  TRIAL 3: Fit Classification")
    print("  Target: Fit_Classification (High Fit / Moderate Fit / Low Fit)")
    print("="*70)

    df = load_data()
    df = engineer_features(df)
    df = create_target_fit_classification(df)

    print("\n  Class Distribution:")
    print(df['target_class'].value_counts().to_string(header=False))

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    X_train, X_test, y_train, y_test, le, features = prepare_data(df, 'target_class')
    print(f"\n  Features: {len(features)}")
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    results = run_all_models(X_train, X_test, y_train, y_test, le,
                             'Trial 3: Fit Classification', output_dir)
    print(f"\n  ✓ Results saved to: {output_dir}")
    return results


if __name__ == '__main__':
    main()
