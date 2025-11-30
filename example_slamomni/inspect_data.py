#!/usr/bin/env python3
"""
Script to inspect the parquet data structure.
"""

from datasets import load_dataset

# Load first parquet file
dataset = load_dataset("parquet", data_files="/data/train-00000-of-00853.parquet", split='train')

print(f"Dataset size: {len(dataset)}")
print(f"\nDataset features: {dataset.features}")
print(f"\nFirst sample:")
print(dataset[0])
print(f"\nColumn names: {dataset.column_names}")

# Print first 3 samples to understand structure
print(f"\n=== First 3 samples ===")
for i in range(min(3, len(dataset))):
    print(f"\n--- Sample {i} ---")
    sample = dataset[i]
    for key, value in sample.items():
        if isinstance(value, (str, int, float)):
            print(f"{key}: {value[:200] if isinstance(value, str) else value}")
        else:
            print(f"{key}: {type(value)} (length: {len(value) if hasattr(value, '__len__') else 'N/A'})")
