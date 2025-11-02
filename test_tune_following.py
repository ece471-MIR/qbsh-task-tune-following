from data_loader import MIRQBSHDataset
from preprocessing import preprocess_query, fill_unvoiced
from dtw_rewrapper import DTWWrapper
import librosa
import numpy as np
import matplotlib.pyplot as plt
import sys


args = sys.argv[1:]
if '-q' in args: # bad arg checking
    query_file = int(args[args.index('-q') + 1])
else:
    query_file = 10

dataset = MIRQBSHDataset("./data/MIR-QBSH")
dtw_computer = DTWWrapper(dataset)

query_path = dataset.query_files[query_file]
raw_query = dataset.load_query_pv(query_path)
processed = preprocess_query(raw_query)

for fill_temp in [True, False]:
    template = dataset.load_template_midi(query_path.stem)
    if fill_temp:
        template = fill_unvoiced(template)

    ### warping, tuning

    d_beg = dtw_computer._compute_d_beg(processed, template)
    processed -= d_beg

    D, wp = librosa.sequence.dtw(
        Y=processed,
        X=template[0:len(processed)],
        band_rad=0.5
    )
    query_warped: np.ndarray = processed[wp[:,0][::-1]]
    query_tuned: np.ndarray = dtw_computer._tune_follow(
        query_warped,
        template
    )

    q_len = query_warped.size
    plt.figure(figsize=(12, 4))
    plt.plot(template[:q_len], label='Template', alpha=0.7)
    plt.plot(query_warped[:q_len], label='Query before Tune Following', alpha=0.7)
    plt.plot(query_tuned[:q_len], label='Query after Tune Following', alpha=0.7)
    plt.legend()
    plt.xlabel('Frame')
    plt.ylabel('MIDI Note')
    plt.title(f'Effect of Tune Following Procedure, Template Unvoiced Filling = {fill_temp}')
    plt.savefig(f'tune_following_effect_q{query_file}_fill_temp_{fill_temp}.png')
    print(f"Saved tune_following_effect_q{query_file}_fill_temp_{fill_temp}.png")
