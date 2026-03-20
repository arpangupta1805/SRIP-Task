import pyarrow.parquet as pq
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

def data_analysis(file_path):
    try:
        parquet_file = pq.ParquetFile(file_path)
        print("Schema")
        print(parquet_file.schema)
        print(f"Num row groups: {parquet_file.num_row_groups}")
        
        print("\n HEAD (from first row group) ")
        df_rg = parquet_file.read_row_group(0).to_pandas()
        pd.set_option('display.max_columns', None)
        print(df_rg.head(10))
        print("\nTOPIC VALUE COUNTS")
        print(df_rg['TOPIC'].value_counts())
    except Exception as e:
        print(f"Error reading Parquet: {e}")

if __name__ == "__main__":
    data_analysis('dataset_10M.parquet')
