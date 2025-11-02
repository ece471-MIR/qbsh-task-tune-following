# qbsh-task-tune-following
Re-Implementation of Tune Following procedure for MIREX 2016s QbSH task
Replication of Stasiak (2014) "Follow That Tune" paper using librosa's DTW implementation.

## Prerequisite

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) for
your platform

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
```

This tests the 5-step preprocessing pipeline from Section 3 of the 2014 paper:
1. Removes leading/trailing silence
2. Removes outliers (>24 semitones from median)
3. Limits jumps (>14 semitones between frames)
4. Fill unvoiced frames
5. Apply median filter (order 9)

## Testing Dynamic Time Warping and Tune Following Together
```bash
uv run python test_dtw.py
```

Infers the ground-truth template of a fixed query with and without DTW & TF (strictly both or neither).

Produces the plots `dtw_base_True.png` and `dtw_base_False.png` showing query vector, actual template vector and inferred template vector.

## Testing Tune Following
```bash
uv run python test_tune_following.py [-q NUMBER]
```

Passing in `-q NUMBER` allows the specification of a query.

Plots the query vector after DTW, query vector after DTW & tone following to the correct template and the correct template vector with and without filling unvoiced sections of the template vector. Saves the plots to `tune_following_effect_q{NUMBER}_fill_temp_True.png` and `tune_following_effect_q{NUMBER}_fill_temp_False.png`.

## Evaluation
```bash
uv run python eval_loop.py [--warp] [--tune] [--fill_templates] 
```

Loops over all queries and infers a template from all templates.  
`--warp` enables DTW.  
`--tune` enables TF.  
`--fill_templates` fills unvoiced sections of the template vectors.

Outputs percentage of queries for which the top inference was correct (Best Hit Score) and for which the correct template was in the top 10 inferences (Top Ten Score).

Results are provided in [eval/](eval/).
