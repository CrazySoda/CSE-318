#!/usr/bin/env python3

import sys
import csv
import os
from collections import Counter

def detect_missing_values(data):
    missing_indicators = set()
    for row in data:
        for cell in row:
            cell_clean = cell.strip()
            if cell_clean in ['?', '', 'NA', 'na', 'NULL', 'null', 'NaN', 'nan', 'missing']:
                missing_indicators.add(cell_clean)
    return missing_indicators

def calculate_majority_values(data, missing_indicators):
    if not data:
        return []
    
    num_columns = len(data[0])
    majority_values = []
    
    for col_idx in range(num_columns):
        column_values = []
        for row in data:
            if col_idx < len(row):
                cell_value = row[col_idx].strip()
                if cell_value not in missing_indicators and cell_value:
                    column_values.append(cell_value)
        
        if column_values:
            value_counts = Counter(column_values)
            majority_value = value_counts.most_common(1)[0][0]
            majority_values.append(majority_value)
        else:
            majority_values.append("Unknown")
    
    return majority_values

def impute_missing_values(data, majority_values, missing_indicators):
    imputed_data = []
    total_missing = 0
    
    for row in data:
        new_row = []
        for col_idx in range(len(majority_values)):
            if col_idx < len(row):
                cell_value = row[col_idx].strip()
                if cell_value in missing_indicators or not cell_value:
                    new_row.append(majority_values[col_idx])
                    total_missing += 1
                else:
                    new_row.append(cell_value)
            else:
                new_row.append(majority_values[col_idx])
                total_missing += 1
        imputed_data.append(new_row)
    
    print(f"Total missing values imputed: {total_missing}")
    return imputed_data

def convert_data_to_csv_with_imputation(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
        
        if not lines:
            print("Error: Input file is empty")
            return False
        
        lines = [line.strip() for line in lines if line.strip()]
        
        if not lines:
            print("Error: No valid data found")
            return False
        
        raw_data = []
        num_columns = 0
        
        for line in lines:
            row = [cell.strip() for cell in line.split(',')]
            raw_data.append(row)
            num_columns = max(num_columns, len(row))
        
        if not raw_data:
            print("Error: No valid data rows found")
            return False
        
        print(f"Processing {len(lines)} rows with {num_columns} columns")
        
        missing_indicators = detect_missing_values(raw_data)
        majority_values = calculate_majority_values(raw_data, missing_indicators)
        imputed_data = impute_missing_values(raw_data, majority_values, missing_indicators)
        
        headers = [f"attr{i+1}" for i in range(num_columns)]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            
            for row in imputed_data:
                while len(row) < num_columns:
                    row.append(majority_values[len(row)] if len(row) < len(majority_values) else "Unknown")
                row = row[:num_columns]
                writer.writerow(row)
        
        print(f"Successfully converted to {output_file}")
        return True
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_data.py <input_file.data> [output_file.csv]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        base_name = os.path.splitext(input_file)[0]
        output_file = base_name + '_clean.csv'
    
    success = convert_data_to_csv_with_imputation(input_file, output_file)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()