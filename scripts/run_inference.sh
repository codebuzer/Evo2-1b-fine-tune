#!/bin/bash

# Create country names file if it doesn't exist
mkdir -p /app/preprocessed_data
python -c "
import pandas as pd
df = pd.read_csv('/app/data/train_spike_data.csv')
country_names = [col for col in df.columns if col != 'sequence']
with open('/app/preprocessed_data/country_names.txt', 'w') as f:
    f.write('\n'.join(country_names))
"

# Run inference
python /app/scripts/inference.py
