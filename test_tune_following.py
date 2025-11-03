from data_loader import MIRQBSHDataset
from preprocessing import preprocess_query, fill_unvoiced
from dtw_wrapper import DTWWrapper
import librosa
import numpy as np
import matplotlib.pyplot as plt
import sys

args = sys.argv[1:]
if '-q' in args: # bad arg checking
    query_file = int(args[args.index('-q') + 1])
else:
    query_file = 7

dataset = MIRQBSHDataset("./data/MIR-QBSH")
dtw_computer = DTWWrapper(dataset)

query_path = dataset.query_files[query_file]
raw_query = dataset.load_query_pv(query_path)
processed = preprocess_query(raw_query)

for fill in [True, False]:

    query_tuned, template_subseq, query_warped = dtw_computer.fit_template(
        query_in=processed,
        template_in=dataset.load_template_midi(query_path.stem),
        uni_w = True, bi_w = False, tune = True, fill = True
    )

    q_len = query_warped.size
    plt.figure(figsize=(12, 4))
    # plt.plot(template[:q_len], label='Template', alpha=0.7)
    plt.plot(template_subseq[:q_len], label='Template Subsequence', alpha=0.7)
    # plt.plot(processed[:q_len], label='Query before DTW', alpha=0.7)
    plt.plot(query_warped[:q_len], label='Query after DTW, before TF', alpha=0.7)
    plt.plot(query_tuned[:q_len], label='Query after TF', alpha=0.7)
    plt.legend()
    plt.xlabel('Frame')
    plt.ylabel('MIDI Note')
    plt.title(f'Effect of Tune Following Procedure, Template Unvoiced Filling = {fill}')
    plt.savefig(f'tune_following_effect_q{query_file}_fill_{fill}.png')
    print(f"Saved tune_following_effect_q{query_file}_fill_{fill}.png")
