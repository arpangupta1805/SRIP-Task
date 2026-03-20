import pyarrow.parquet as pq
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

def sample_data(input_file, output_file, size):
    try:
        parquet_file = pq.ParquetFile(input_file)
        df_list = []
        total_rows = 0
        for i in range(parquet_file.num_row_groups):
            chunk = parquet_file.read_row_group(i).to_pandas()
            df_list.append(chunk)
            total_rows += len(chunk)
            if total_rows >= size:
                break

        df = pd.concat(df_list, ignore_index=True)
        df = df.head(size)

        print(f"Saving to {output_file}")
        df.to_parquet(output_file, engine='pyarrow')
        print("Shape:", df.shape)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    sample_data('dataset_10M.parquet', 'dataset_sample_500k.parquet', 500000)