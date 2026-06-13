"""
Trial 4: Orders Class
Target: TOTAL_ORDERS -> High (>100) / Mid (10-100) / Low (<10)

Tests commercial volume prediction capability.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data_loader import load_data, engineer_features, create_target_orders_class, prepare_data
from common.evaluation import run_all_models


def main():
    print("\n" + "="*70)
    print("  TRIAL 4: Orders Class")
    print("  Target: TOTAL_ORDERS → High (>100) / Mid (10-100) / Low (<10)")
    print("="*70)

    df = load_data()
    df = engineer_features(df)
    df = create_target_orders_class(df)

    print("\n  Class Distribution:")
    print(df['target_class'].value_counts().to_string(header=False))

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    X_train, X_test, y_train, y_test, le, features = prepare_data(df, 'target_class')
    print(f"\n  Features: {len(features)}")
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    results = run_all_models(X_train, X_test, y_train, y_test, le,
                             'Trial 4: Orders Class', output_dir)
    print(f"\n  ✓ Results saved to: {output_dir}")
    return results


if __name__ == '__main__':
    main()
