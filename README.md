# qbsh-task-tune-following
Re-Implementation of Tune Following procedure for MIREX 2016s QbSH task
Replication of Stasiak (2014) "Follow That Tune" paper using librosa's DTW implementation.

## Setup

1. **Create virtual environment and install dependencies:**
```bash
uv venv
uv pip install -r requirements.txt
```

2. **Download the dataset:**
```bash
./get_data.sh
```

This downloads the MIR-QBSH corpus (4431 queries, 48 ground-truth songs).

## Dataset Structure

After running `get_data.sh`, data is in `data/MIR-QBSH/`:
- `midiFile/` - 48 ground-truth MIDI files and songList.txt
- `waveFile/` - 4431 query files organized by year/person
    - Each query has a `.pv` file (manually labeled pitch vector)
    - Format: MIDI note numbers, one per line (0 = unvoiced)

## Testing Data Loading
```bash
uv run python test_data_loader.py
```

Expected output:
- 48 ground-truth songs loaded
- 4431 query files found
- Ground truth mapping verified

## Testing Preprocessing
```bash
uv run python test_preprocessing.py

This tests the 5-step preprocessing pipeline from Section 3 of the 2014 paper:
1. Removes leading/trailing silence
2. Removes outliers (>24 semitones from median)
3. Limits jumps (>14 semitones between frames)
4. Fill unvoiced frames
5. Apply median filter (order 9)
