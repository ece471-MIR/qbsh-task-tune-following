from data_loader import MIRQBSHDataset
from preprocessing import preprocess_query
from dtw_wrapper import DTWWrapper
import matplotlib.pyplot as plt

dataset = MIRQBSHDataset("./data/MIR-QBSH")

query_path = dataset.query_files[7]
raw_query = dataset.load_query_pv(query_path)

template = dataset.load_template_midi(query_path.stem)

print(f"Query: {query_path}")
print(f"Raw length: {len(raw_query)}")
print(f"Raw range: [{raw_query.min():.1f}, {raw_query.max():.1f}]")

processed = preprocess_query(raw_query)

dtw_computer = DTWWrapper(dataset)

for do_we_tune in [False, True]:
    print(f"Do we tune: {do_we_tune}")
    predicted_template = dtw_computer.match_query_in_database(processed,
                                                              tuned=do_we_tune)

    print(f"Predicted: {predicted_template}, actual: {query_path.stem}")

    template_guessed = dataset.load_template_midi(predicted_template[0])

    processed -= dtw_computer._compute_d_beg(
        processed, template_guessed
    )
    if do_we_tune:
        tune_processed = dtw_computer._tune_follow(processed, template_guessed)
    else:
        tune_processed = processed

    q_len = tune_processed.size
    plt.figure(figsize=(12, 4))
    plt.plot(template[:q_len], label='Actual', alpha=0.7)
    plt.plot(template_guessed[:q_len], label=f'Guessed {predicted_template[0]}', alpha=0.7)
    plt.plot(tune_processed, label='Query', alpha=0.7)
    plt.legend()
    plt.xlabel('Frame')
    plt.ylabel('MIDI Note')
    plt.title(f'Templates: Predicted vs Actual Source, did we tune: {do_we_tune}')
    plt.savefig(f'dtw_base_{do_we_tune}.png')
    print(f"Saved visualization to dtw_base_{do_we_tune}.png")

