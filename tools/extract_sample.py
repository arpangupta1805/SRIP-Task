import pyarrow.parquet as pq
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

def sample_data(input_file, output_file, size):
    try:
        parquet_file = pq.ParquetFile(input_file)
        df_list = []
        for i in range(2):
            df_list.append(parquet_file.read_row_group(i).to_pandas())

        df = pd.concat(df_list, ignore_index=True)
        df = df.head(size)

        print(f"Saving to {output_file}")
        df.to_parquet(output_file, engine='pyarrow')
        print("Shape:", df.shape)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    sample_data('dataset_10M.parquet', 'dataset_sample_2M.parquet', 2000000)