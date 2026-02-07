# Run this and share the output
import pandas as pd
import numpy as np

df = pd.read_csv("../data/Imputing Rationales.csv")
rationales = ['diversity', 'indep', 'tenure', 'busyness', 'combined_ceo_chairman']

# Get labeled and unlabeled
dissent = df[df['ind_dissent'] == 1]
labeled = dissent[dissent[rationales].notna().any(axis=1)]
unlabeled = dissent[dissent[rationales].isna().all(axis=1)]

print("=" * 80)
print("DATA DISTRIBUTION ANALYSIS")
print("=" * 80)

print(f"\nSample sizes:")
print(f"  Total dissent: {len(dissent):,}")
print(f"  Labeled: {len(labeled):,}")
print(f"  Unlabeled: {len(unlabeled):,}")

print(f"\nInvestor overlap:")
print(f"  Unique investors (labeled): {labeled['investor_id'].nunique()}")
print(f"  Unique investors (unlabeled): {unlabeled['investor_id'].nunique()}")
print(f"  Overlap: {len(set(labeled['investor_id']) & set(unlabeled['investor_id']))}")

print(f"\nTemporal distribution:")
print(f"  Labeled years: {labeled['ProxySeason'].min()}-{labeled['ProxySeason'].max()}")
print(f"  Unlabeled years: {unlabeled['ProxySeason'].min()}-{unlabeled['ProxySeason'].max()}")

print(f"\nKey feature comparison:")
key_features = ['Per_female', 'AvTenure', 'Per_Independent', 'frac_vote_against']
for feat in key_features:
    if feat in df.columns:
        print(f"\n{feat}:")
        print(f"  Labeled:   {labeled[feat].mean():.4f} (std: {labeled[feat].std():.4f})")
        print(f"  Unlabeled: {unlabeled[feat].mean():.4f} (std: {unlabeled[feat].std():.4f})")
        
        # Check missing rates
        print(f"  Missing - Labeled: {labeled[feat].isna().mean():.1%}, Unlabeled: {unlabeled[feat].isna().mean():.1%}")

print(f"\nPositive rates (labeled data only):")
for rat in rationales:
    pos_rate = labeled[rat].mean()
    print(f"  {rat}: {pos_rate:.1%} ({labeled[rat].sum():.0f} / {labeled[rat].notna().sum():.0f})")