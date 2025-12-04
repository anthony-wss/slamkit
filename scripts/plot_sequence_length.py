#!/usr/bin/env python3
"""
Plot sequence length distribution from JSONL file.
Memory-efficient streaming implementation for large files.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_sequence_lengths(jsonl_path, max_lines=None):
    """
    Stream through JSONL file and extract sequence lengths.

    Args:
        jsonl_path: Path to JSONL file
        max_lines: Optional limit on number of lines to process

    Returns:
        List of sequence lengths
    """
    lengths = []

    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break

            if i % 10000 == 0:
                print(f"Processed {i:,} lines...")

            data = json.loads(line)
            length = len(data['input_ids'])
            lengths.append(length)

    return lengths

def plot_distribution(lengths, output_path='sequence_length_distribution.png'):
    """
    Create histogram and statistics plot for sequence lengths.

    Args:
        lengths: List of sequence lengths
        output_path: Path to save the plot
    """
    lengths_array = np.array(lengths)

    # Calculate statistics
    stats = {
        'count': len(lengths_array),
        'mean': np.mean(lengths_array),
        'median': np.median(lengths_array),
        'std': np.std(lengths_array),
        'min': np.min(lengths_array),
        'max': np.max(lengths_array),
        'p25': np.percentile(lengths_array, 25),
        'p75': np.percentile(lengths_array, 75),
        'p95': np.percentile(lengths_array, 95),
        'p99': np.percentile(lengths_array, 99),
    }

    # Print statistics
    print("\nSequence Length Statistics:")
    print(f"  Total sequences: {stats['count']:,}")
    print(f"  Mean:   {stats['mean']:.2f}")
    print(f"  Median: {stats['median']:.2f}")
    print(f"  Std:    {stats['std']:.2f}")
    print(f"  Min:    {stats['min']}")
    print(f"  Max:    {stats['max']}")
    print(f"  25th percentile: {stats['p25']:.2f}")
    print(f"  75th percentile: {stats['p75']:.2f}")
    print(f"  95th percentile: {stats['p95']:.2f}")
    print(f"  99th percentile: {stats['p99']:.2f}")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Full histogram
    ax1.hist(lengths_array, bins=100, edgecolor='black', alpha=0.7)
    ax1.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.0f}")
    ax1.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats['median']:.0f}")
    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Sequence Length Distribution (n={stats["count"]:,})', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Zoomed histogram (exclude outliers beyond 99th percentile)
    max_length_zoom = stats['p99']
    lengths_zoomed = lengths_array[lengths_array <= max_length_zoom]
    ax2.hist(lengths_zoomed, bins=100, edgecolor='black', alpha=0.7, color='orange')
    ax2.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.0f}")
    ax2.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats['median']:.0f}")
    ax2.set_xlabel('Sequence Length', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Sequence Length Distribution (Zoomed to 99th percentile: {max_length_zoom:.0f})',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    return stats

def main():
    # Configuration
    jsonl_file = 'example_slamomni/sft_data/train_all.jsonl'
    output_file = 'sequence_length_distribution.png'

    print(f"Analyzing sequence lengths in: {jsonl_file}")
    print("This may take a few minutes for large files...\n")

    # Process file
    lengths = analyze_sequence_lengths(jsonl_file)

    # Create plot
    plot_distribution(lengths, output_file)

    print("\nDone!")

if __name__ == '__main__':
    main()
