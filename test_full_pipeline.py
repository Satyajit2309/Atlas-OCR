"""Test the refined app.py on Sample.pdf — expect 5 dimensions: 10, 410, 133, 169, 122"""
import os
os.environ['USE_TORCH'] = '1'
import sys
sys.path.insert(0, '.')
from app import process_drawing
import json

result, error, excel, csv_f = process_drawing('Sample.pdf', 'Sample.pdf')
if error:
    print(f'ERROR: {error}')
else:
    print(f"\n{'='*60}")
    print(f"Total dimensions extracted: {len(result['data'])}")
    print(f"{'='*60}")
    for row in result['data']:
        print(f"  [{row.get('confidence','?')}] {row['feature']}: {row['value']} {row['unit']} ({row['type']})")
    
    expected = {'10', '410', '133', '169', '122'}
    found = {row['value'] for row in result['data']}
    print(f"\nExpected: {expected}")
    print(f"Found:    {found}")
    print(f"Match:    {expected & found}")
    print(f"Missing:  {expected - found}")
    print(f"Extra:    {found - expected}")
    print(f"\nFiles: {excel}, {csv_f}")
