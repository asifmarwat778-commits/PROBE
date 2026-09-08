"""
SHAP-Equivalent Feature Importance for GBR
Permutation Importance + Partial Dependence Plot
Font: Times New Roman, size 28, DPI 600
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import partial_dependence, permutation_importance
import matplotlib.pyplot as plt

# Load data
file_path = r"D:\HIT Research\Research Paper 1\Updated paper\ML\DATA.csv"
df = pd.read_csv(file_path)
target_col = 'Ct/C0 (output)'
df = pd.get_dummies(df, columns=['System / Notes'], drop_first=False)
y = df[target_col].values
X_df = df.drop(columns=[target_col])
feature_names = X_df.columns.tolist()
X = X_df.values

# Train GBR on full data
gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gbr.fit(X, y)

# --- 1. Permutation Importance ---
perm = permutation_importance(gbr, X, y, n_repeats=10, random_state=42, n_jobs=-1, scoring='r2')
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': perm.importances_mean,
    'Std': perm.importances_std
}).sort_values('Importance', ascending=False)

print("=" * 60)
print("=== Top 10 Features by Permutation Importance ===")
print("=" * 60)
print(importance_df.head(10).to_string(index=False))
print("=" * 60)

# --- 2. Feature Importance Plot (font size 28, DPI 600) ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(8, 8))
top10 = importance_df.head(10)
y_pos = np.arange(len(top10))
ax.barh(y_pos, top10['Importance'], xerr=top10['Std'], color='steelblue', edgecolor='k', alpha=0.85, capsize=3)
ax.set_yticks(y_pos)
ax.set_yticklabels(top10['Feature'], fontsize=14, fontname='Times New Roman')
ax.invert_yaxis()
ax.set_xlabel('Permutation Importance (R² decrease)', fontsize=28, fontname='Times New Roman')
ax.set_title('GBR Feature Importance (Full Dataset)', fontsize=28, fontname='Times New Roman', pad=15)
ax.tick_params(axis='both', labelsize=14)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname('Times New Roman')
ax.grid(True, axis='x', linestyle='--', alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('GBR_Feature_Importance.png', dpi=600, bbox_inches='tight')
plt.show()
print("Saved: GBR_Feature_Importance.png")

# --- 3. pH Partial Dependence Plot (font size 28, DPI 600) ---
pH_idx = feature_names.index('pH')
pd_result = partial_dependence(gbr, X, features=[pH_idx], kind='average')
pd_values = pd_result['average'][0]
pd_grid = pd_result['grid_values'][0]

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(pd_grid, pd_values, 'b-', lw=2.5, label='Partial Dependence')
ax.set_xlabel('pH', fontsize=28, fontname='Times New Roman')
ax.set_ylabel('Partial Dependence (Ct/C$_0$)', fontsize=28, fontname='Times New Roman')
ax.set_title('Partial Dependence of Ct/C$_0$ on pH (GBR)', fontsize=28, fontname='Times New Roman', pad=15)
ax.tick_params(axis='both', labelsize=14)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname('Times New Roman')
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_axisbelow(True)

# Mark tested pH values
for p in [3, 5, 7, 9]:
    ax.axvline(x=p, color='gray', linestyle='--', alpha=0.5, lw=1)

# Annotation
ax.annotate('HOCl → OCl⁻ shift\nFe(III) hydrolysis', 
            xy=(5, pd_values[np.argmin(np.abs(pd_grid - 5))]), 
            xytext=(6.5, min(pd_values) + 0.05),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=13, fontname='Times New Roman', color='red')

plt.tight_layout()
plt.savefig('GBR_pH_Partial_Dependence.png', dpi=600, bbox_inches='tight')
plt.show()
print("Saved: GBR_pH_Partial_Dependence.png")