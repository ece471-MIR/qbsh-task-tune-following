from data_loader import MIRQBSHDataset
from preprocessing import preprocess_query
from dtw_wrapper import DTWWrapper
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

args = sys.argv[1:]
do_we_warp = '--warp' in args
do_we_tune = '--tune' in args
fill_temp = '--fill_templates' in args

print(f'Do we warp: {do_we_warp}')
print(f'Do we tune: {do_we_tune}')
print(f'Do we fill template unvoiced sections: {fill_temp}')

dataset = MIRQBSHDataset("./data/MIR-QBSH")
query_paths = dataset.query_files
query_count = len(query_paths)

top_ten_count = 0
best_hit_count = 0
for query_path in tqdm(query_paths):
    raw_query = dataset.load_query_pv(query_path)
    processed = preprocess_query(raw_query)
    template = query_path.stem

    dtw_computer = DTWWrapper(dataset)
    predicted_template = dtw_computer.match_query_in_database(processed,
                                                                warp=do_we_warp,
                                                                tuned=do_we_tune,
                                                                fill_temp=fill_temp,
                                                                prog_bar=False)

    if template == predicted_template[0]:
        best_hit_count += 1
        top_ten_count += 1
    elif query_path.stem in predicted_template:
        top_ten_count += 1

print('\nResults:')
print(f' Top Ten Score:  {(100.0 * top_ten_count) / query_count}%')
print(f' Best Hit Score: {(100.0 * best_hit_count) / query_count}%')
