from data_loader import MIRQBSHDataset

dataset = MIRQBSHDataset("./data/MIR-QBSH/")

print("\n" + "="*60)
print("SONGS (from songList.txt):")
print("="*60)
for i, (key, info) in enumerate(list(dataset.song_list.items())[:5]):
    print(f"{key}: {info}")

print("\n" + "="*60)
print("QUERIES (first 10):")
print("="*60)
for query_path in dataset.query_files[:10]:
    print(query_path)

print("\n" + "="*60)
print("GROUND TRUTH MAPPING (sample):")
print("="*60)
ground_truth = dataset.get_ground_truth_mapping()
print(f"Total mappings: {len(ground_truth)}")
for i, (query, song) in enumerate(list(ground_truth.items())[:10]):
    print(f"{query} -> {song}")

print("\n" + "="*60)
print("TEST: Load a query")
print("="*60)
if dataset.query_files:
    test_query = dataset.query_files[0]
    print(f"Loading: {test_query}")
    pv = dataset.load_query_pv(test_query)
    print(f"  Length: {len(pv)} frames")
    print(f"  Range: [{pv.min():.1f}, {pv.max():.1f}]")
    print(f"  First 10 values: {pv[:10]}")

print("\n" + "="*60)
print("TEST: Load a template MIDI")
print("="*60)
try:
    template = dataset.load_template_midi("00001")  # "I'm the teapot"
    print(f"  Length: {len(template)} frames")
    print(f"  Range: [{template.min():.1f}, {template.max():.1f}]")
    print(f"  First 10 values: {template[:10]}")
except Exception as e:
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()