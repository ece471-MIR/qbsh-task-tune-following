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

plt.figure(figsize=(12, 4))
plt.plot(raw_query, label='Raw', alpha=0.7)
plt.plot(processed, label='Processed', alpha=0.7)
plt.legend()
plt.xlabel('Frame')
plt.ylabel('MIDI Note')
plt.title('Preprocessing Effect')
plt.savefig('preprocessing_test.png')
print("Saved visualization to preprocessing_test.png")

plt.figure(figsize=(12, 4))
plt.plot(template[0:237], label='Template', alpha=0.7)
plt.plot(processed, label='Processed', alpha=0.7)
plt.legend()
plt.xlabel('Frame')
plt.ylabel('MIDI Note')
plt.title('Preprocessed vs template')
plt.savefig('preprocessing_vs_template.png')
print("Saved visualization to preprocessing_vs_template.png")

