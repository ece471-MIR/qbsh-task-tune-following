from data_loader import MIRQBSHDataset
from preprocessing import preprocess_query
from dtw_wrapper import DTWWrapper
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

args = sys.argv[1:]
uni_w = '-u' in args or '--uni'  in args
bi_w  = '-b' in args or '--bi'   in args
assert (not uni_w) or (not bi_w), "Warp cannot be uni- and bi-directional!"
tune = '-t' in args or '--tune' in args
fill = '-f' in args or '--fill' in args

print(f'Unidirectional DTW: {uni_w}')
print(f'Bidirectional DTW: {bi_w}')
print(f'Tune-Following: {tune}')
print(f'Template Unvoiced Filling: {fill}')

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
    predicted_template = dtw_computer.match_query_in_database(
        query_in=processed,
        uni_w=uni_w,
        bi_w=bi_w,
        tune=tune,
        fill=fill,
        prog_bar=False
    )

    if template == predicted_template[0]:
        best_hit_count += 1
        top_ten_count += 1
    elif template in predicted_template:
        top_ten_count += 1

print('\nResults:')
print(f' Top Ten Score:  {(100.0 * top_ten_count) / query_count}%')
print(f' Best Hit Score: {(100.0 * best_hit_count) / query_count}%')
