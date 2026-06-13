"""
Trial 1: Sales Performance Class (BASELINE)
Target: TOTAL_SALES -> High (>$10K) / Mid ($1K-$10K) / Low (<$1K)

This is the baseline trial matching the existing model's target variable.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data_loader import load_data, engineer_features, create_target_sales_class, prepare_data
from common.evaluation import run_all_models


def main():
    print("\n" + "="*70)
    print("  TRIAL 1: Sales Performance Class (Baseline)")
    print("  Target: TOTAL_SALES → High (>$10K) / Mid ($1K-$10K) / Low (<$1K)")
    print("="*70)

    df = load_data()
    df = engineer_features(df)
    df = create_target_sales_class(df)

    print("\n  Class Distribution:")
    print(df['target_class'].value_counts().to_string(header=False))

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    X_train, X_test, y_train, y_test, le, features = prepare_data(df, 'target_class')
    print(f"\n  Features: {len(features)}")
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    results = run_all_models(X_train, X_test, y_train, y_test, le,
                             'Trial 1: Sales Class', output_dir)
    print(f"\n  ✓ Results saved to: {output_dir}")
    return results


if __name__ == '__main__':
    main()
