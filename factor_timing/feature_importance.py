# feature_importance.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from config import FEATURES


def plot_feature_importance(rf_model, X_test_scaled, y_test_bin):
    # Method A: RF built-in feature importance (impurity-based, in-sample)
    importances = rf_model.feature_importances_
    feat_imp    = pd.Series(importances, index=FEATURES).sort_values(ascending=True)

    # Method B: Permutation Importance (more reliable, out-of-sample)
    perm     = permutation_importance(
        rf_model, X_test_scaled, y_test_bin,
        n_repeats=30, random_state=42, scoring='accuracy'
    )
    perm_imp = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    feat_imp.plot(kind='barh', ax=axes[0], color='steelblue')
    axes[0].set_title('RF Feature Importance\n(Impurity-based, In-sample)', fontsize=12)
    axes[0].set_xlabel('Importance Score')
    axes[0].axvline(1/len(FEATURES), color='red', linestyle='--', label='Uniform baseline')
    axes[0].legend()

    perm_imp.plot(kind='barh', ax=axes[1], color='darkorange')
    axes[1].set_title('Permutation Importance\n(Out-of-sample)', fontsize=12)
    axes[1].set_xlabel('Accuracy Drop when Feature Shuffled')
    axes[1].axvline(0, color='red', linestyle='--')

    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()

    summary = pd.DataFrame({
        'Impurity_Imp':    pd.Series(rf_model.feature_importances_, index=FEATURES),
        'Permutation_Imp': pd.Series(perm.importances_mean,          index=FEATURES),
        'Permutation_Std': pd.Series(perm.importances_std,           index=FEATURES)
    }).sort_values('Permutation_Imp', ascending=False)

    print("\nFeature importance ranking (out-of-sample permutation as primary):")
    print(summary.round(5))
    return summary
