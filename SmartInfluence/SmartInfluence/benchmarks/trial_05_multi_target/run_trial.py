"""
Trial 5: Multi-Target Composite
Target: Combined Sales rank + Engagement rank → Top / Mid / Low

Tests a holistic influencer quality score combining both commercial
performance and social engagement.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data_loader import load_data, engineer_features, create_target_multi, prepare_data
from common.evaluation import run_all_models


def main():
    print("\n" + "="*70)
    print("  TRIAL 5: Multi-Target Composite")
    print("  Target: Sales rank + Engagement rank → Top / Mid / Low")
    print("="*70)

    df = load_data()
    df = engineer_features(df)
    df = create_target_multi(df)

    print("\n  Class Distribution:")
    print(df['target_class'].value_counts().to_string(header=False))

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    X_train, X_test, y_train, y_test, le, features = prepare_data(df, 'target_class')
    print(f"\n  Features: {len(features)}")
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    results = run_all_models(X_train, X_test, y_train, y_test, le,
                             'Trial 5: Multi-Target', output_dir)
    print(f"\n  ✓ Results saved to: {output_dir}")
    return results


if __name__ == '__main__':
    main()
