# https://www.engreitzlab.org/resources?
# wget ftp://ftp.broadinstitute.org/outgoing/lincRNA/ABC/AllPredictions.AvgHiC.ABC0.015.minus150.ForABCPaperV3.txt.gz

import pandas as pd
import gzip
from pyliftover import LiftOver
from tqdm import tqdm

print("Initializing LiftOver (hg19 -> hg38)...")
lo = LiftOver('hg19', 'hg38')

print("Reading input file...")
input_file = "data/AllPredictions.AvgHiC.ABC0.015.minus150.ForABCPaperV3.txt.gz"
df = pd.read_csv(input_file, sep='\t', compression='gzip')

print(f"Loaded {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")

def liftover_coordinates(row):
    chrom = row['chr']
    start = row['start']
    end = row['end']
    
    new_start = lo.convert_coordinate(chrom, start)
    new_end = lo.convert_coordinate(chrom, end)

    if new_start and new_end and len(new_start) > 0 and len(new_end) > 0:
        if len(new_start[0]) == 2:
            new_chrom_start, new_pos_start = new_start[0]
            new_chrom_end, new_pos_end = new_end[0]
        else:
            new_chrom_start, new_pos_start = new_start[0][:2]
            new_chrom_end, new_pos_end = new_end[0][:2]
        
        if new_chrom_start == new_chrom_end:
            return pd.Series({
                'chr_hg38': new_chrom_start,
                'start_hg38': int(new_pos_start),  # convert to integer
                'end_hg38': int(new_pos_end),      # convert to integer
                'liftover_status': 'success'
            })

    return pd.Series({
        'chr_hg38': None,
        'start_hg38': None,
        'end_hg38': None,
        'liftover_status': 'failed'
    })

print("\nPerforming liftover...")

results = []

batch_size = 10000
num_batches = (len(df) + batch_size - 1) // batch_size

for i in tqdm(range(num_batches), desc="Processing batches"):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(df))
    batch = df.iloc[start_idx:end_idx]
    
    batch_results = []
    for _, row in batch.iterrows():
        chrom = row['chr']
        start = row['start']
        end = row['end']
        
        new_start = lo.convert_coordinate(chrom, start)
        new_end = lo.convert_coordinate(chrom, end)

        if new_start and new_end and len(new_start) > 0 and len(new_end) > 0:
            if len(new_start[0]) == 2:
                new_chrom_start, new_pos_start = new_start[0]
                new_chrom_end, new_pos_end = new_end[0]
            else:
                new_chrom_start, new_pos_start = new_start[0][:2]
                new_chrom_end, new_pos_end = new_end[0][:2]
            
            if new_chrom_start == new_chrom_end:
                batch_results.append({
                    'chr_hg38': new_chrom_start,
                    'start_hg38': int(new_pos_start),  # convert to integer
                    'end_hg38': int(new_pos_end),      # convert to integer
                    'liftover_status': 'success'
                })
            else:
                batch_results.append({
                    'chr_hg38': None,
                    'start_hg38': None,
                    'end_hg38': None,
                    'liftover_status': 'failed'
                })
        else:
            batch_results.append({
                'chr_hg38': None,
                'start_hg38': None,
                'end_hg38': None,
                'liftover_status': 'failed'
            })
    
    results.extend(batch_results)

liftover_results = pd.DataFrame(results)

df_hg38 = pd.concat([df.reset_index(drop=True), liftover_results], axis=1)

success_count = (df_hg38['liftover_status'] == 'success').sum()
failed_count = (df_hg38['liftover_status'] == 'failed').sum()

print(f"\n=== LiftOver Results ===")
print(f"Success: {success_count} ({success_count/len(df)*100:.2f}%)")
print(f"Failed: {failed_count} ({failed_count/len(df)*100:.2f}%)")

df_success = df_hg38[df_hg38['liftover_status'] == 'success'].copy()

df_success['start_hg38'] = df_success['start_hg38'].astype(int)
df_success['end_hg38'] = df_success['end_hg38'].astype(int)

output_file = "data/AllPredictions.AvgHiC.ABC0.015.minus150.ForABCPaperV3.hg38.txt.gz"
df_success.to_csv(output_file, sep='\t', index=False, compression='gzip')
print(f"\nSaved to: {output_file}")

df_replaced = df_success.copy()
df_replaced['chr'] = df_replaced['chr_hg38']
df_replaced['start'] = df_replaced['start_hg38'].astype(int)
df_replaced['end'] = df_replaced['end_hg38'].astype(int)
df_replaced = df_replaced.drop(columns=['chr_hg38', 'start_hg38', 'end_hg38', 'liftover_status'])

output_file_replaced = "data/AllPredictions.AvgHiC.ABC0.015.minus150.ForABCPaperV3.hg38_replaced.txt.gz"
df_replaced.to_csv(output_file_replaced, sep='\t', index=False, compression='gzip')
print(f"Saved (coordinates replaced): {output_file_replaced}")

if failed_count > 0:
    df_failed = df_hg38[df_hg38['liftover_status'] == 'failed']
    failed_file = "data/AllPredictions.failed_liftover.txt.gz"
    df_failed.to_csv(failed_file, sep='\t', index=False, compression='gzip')
    print(f"Failed records saved to: {failed_file}")

print("\n=== Sample of converted data ===")
print(df_success[['chr', 'start', 'end', 'chr_hg38', 'start_hg38', 'end_hg38']].head(10))
