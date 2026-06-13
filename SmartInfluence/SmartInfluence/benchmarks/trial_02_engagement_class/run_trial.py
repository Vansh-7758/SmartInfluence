"""
Trial 2: Engagement Class
Target: INSTAGRAM_ENGAGEMENT_RATE -> High (>5%) / Medium (2-5%) / Low (<2%)

Tests how well social metrics predict engagement tiers.
INSTAGRAM_ENGAGEMENT_RATE is excluded from features since it's the target.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data_loader import load_data, engineer_features, create_target_engagement_class, prepare_data
from common.evaluation import run_all_models


def main():
    print("\n" + "="*70)
    print("  TRIAL 2: Engagement Class")
    print("  Target: ENGAGEMENT_RATE → High (>5%) / Medium (2-5%) / Low (<2%)")
    print("="*70)

    df = load_data()
    df = engineer_features(df)
    df = create_target_engagement_class(df)

    print("\n  Class Distribution:")
    print(df['target_class'].value_counts().to_string(header=False))

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    # Exclude engagement rate from features since it IS the target
    X_train, X_test, y_train, y_test, le, features = prepare_data(
        df, 'target_class',
        exclude_cols=['INSTAGRAM_ENGAGEMENT_RATE', 'engagement_x_growth',
                      'absolute_engagement', 'eng_x_clicks', 'eng_rate_sq',
                      'engagement_density']
    )
    print(f"\n  Features: {len(features)} (engagement-derived columns excluded)")
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    results = run_all_models(X_train, X_test, y_train, y_test, le,
                             'Trial 2: Engagement Class', output_dir)
    print(f"\n  ✓ Results saved to: {output_dir}")
    return results


if __name__ == '__main__':
    main()
