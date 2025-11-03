from data_loader import MIRQBSHDataset
from preprocessing import preprocess_query
import matplotlib.pyplot as plt

dataset = MIRQBSHDataset("./data/MIR-QBSH")

query_path = dataset.query_files[0]
raw_query = dataset.load_query_pv(query_path)

template = dataset.load_template_midi(query_path.stem)

print(f"Query: {query_path}")
print(f"Raw length: {len(raw_query)}")
print(f"Raw range: [{raw_query.min():.1f}, {raw_query.max():.1f}]")

processed = preprocess_query(raw_query)

print(f"Processed length: {len(processed)}")
print(f"Processed range: [{processed.min():.1f}, {processed.max():.1f}]")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

ax1.plot(raw_query, label='Raw', alpha=0.7)
ax1.plot(processed, label='Processed', alpha=0.7)
ax1.legend()
ax1.set_xlabel('Frame')
ax1.set_ylabel('MIDI Note')
ax1.set_title('Preprocessing Effect')

ax2.plot(template[0:237], label='Template', alpha=0.7)
ax2.plot(processed, label='Processed', alpha=0.7)
ax2.legend()
ax2.set_xlabel('Frame')
ax2.set_ylabel('MIDI Note')
ax2.set_title('Preprocessed vs template')

plt.savefig('plots/test_preprocessing.png')
print("Saved visualization to plots/test_preprocessing.png")
