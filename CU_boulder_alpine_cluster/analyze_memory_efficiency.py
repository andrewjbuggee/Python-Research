#!/usr/bin/env python3
"""
Analyze memory efficiency distribution from Alpine jobs.
Reads the CSV output from memory_efficiency_analysis.sh and creates visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
df = pd.read_csv('memory_efficiency_data_25096241.txt')

# Convert to numeric (in case there are any parsing issues)
df['Memory_Efficiency_Percent'] = pd.to_numeric(df['Memory_Efficiency_Percent'], errors='coerce')

# Remove any NaN values
df_clean = df.dropna()

eff = df_clean['Memory_Efficiency_Percent'].values

# Compute statistics
stats = {
    'Mean': np.mean(eff),
    'Median': np.median(eff),
    'Std Dev': np.std(eff),
    'Min': np.min(eff),
    'Max': np.max(eff),
    'Q1': np.percentile(eff, 25),
    'Q3': np.percentile(eff, 75),
}

# Print statistics
print("=" * 60)
print("MEMORY EFFICIENCY STATISTICS")
print("=" * 60)
print(f"Number of jobs analyzed: {len(eff)}")
print(f"Mean:                    {stats['Mean']:.2f}%")
print(f"Median:                  {stats['Median']:.2f}%")
print(f"Std Dev:                 {stats['Std Dev']:.2f}%")
print(f"Min:                     {stats['Min']:.2f}%")
print(f"Max:                     {stats['Max']:.2f}%")
print(f"Q1 (25th percentile):    {stats['Q1']:.2f}%")
print(f"Q3 (75th percentile):    {stats['Q3']:.2f}%")
print("=" * 60)

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram
ax = axes[0, 0]
ax.hist(eff, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(stats['Mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['Mean']:.1f}%")
ax.axvline(stats['Median'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats['Median']:.1f}%")
ax.set_xlabel('Memory Efficiency (%)', fontsize=11)
ax.set_ylabel('Number of Jobs', fontsize=11)
ax.set_title('Distribution of Memory Efficiency', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Box plot
ax = axes[0, 1]
bp = ax.boxplot(eff, vert=True, patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
ax.set_ylabel('Memory Efficiency (%)', fontsize=11)
ax.set_title('Box Plot of Memory Efficiency', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Cumulative distribution
ax = axes[1, 0]
sorted_eff = np.sort(eff)
cumulative = np.arange(1, len(sorted_eff) + 1) / len(sorted_eff) * 100
ax.plot(sorted_eff, cumulative, linewidth=2, color='steelblue')
ax.fill_between(sorted_eff, cumulative, alpha=0.3, color='steelblue')
ax.set_xlabel('Memory Efficiency (%)', fontsize=11)
ax.set_ylabel('Cumulative Percentage of Jobs (%)', fontsize=11)
ax.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# Efficiency ranges
ax = axes[1, 1]
ranges = ['0-25%', '25-50%', '50-75%', '75-100%']
counts = [
    ((eff >= 0) & (eff < 25)).sum(),
    ((eff >= 25) & (eff < 50)).sum(),
    ((eff >= 50) & (eff < 75)).sum(),
    ((eff >= 75) & (eff <= 100)).sum(),
]
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
bars = ax.bar(ranges, counts, color=colors, edgecolor='black', alpha=0.7)
ax.set_ylabel('Number of Jobs', fontsize=11)
ax.set_title('Jobs by Efficiency Range', fontsize=12, fontweight='bold')
# Add count labels on bars
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(count)}', ha='center', va='bottom', fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('memory_efficiency_distribution.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved to: memory_efficiency_distribution.png")
plt.show()
