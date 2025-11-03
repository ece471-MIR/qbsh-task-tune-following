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

## Test Data Loading
```bash
uv run python test_data_loader.py
```

Expected output:
- 48 ground-truth songs loaded
- 4431 query files found
- Ground truth mapping verified

## Test Preprocessing
```bash
uv run python test_preprocessing.py
```

This tests the 5-step preprocessing pipeline from Section 3 of the 2014 paper:
1. Removes leading/trailing silence
2. Removes outliers (>24 semitones from median)
3. Limits jumps (>14 semitones between frames)
4. Fill unvoiced frames
5. Apply median filter (order 9)

## Inference
```bash
uv run python infer.py [-u] [-b] [-t] [-f] [-q NUMBER]
```

Infers the template of a provided query.  
`-u`: enables unidirectional DTW (restricts template warping). Do not specify -b  
`-b`: enables bidirectional DTW (allows template warping). Do not specify -u  
`-t` or `--tune`: enables TF.  
`-f` or `--fill`: fills unvoiced sections of the template vectors.
`-q NUMBER`: specify query

Plots the (optionally warped or tune-fitted) query against its true template (optionally filled or warped) and the top incorrect guess (same conditions) in separate subplots. Saves the plots to `inference_q{NUMBER}_[u][b][t][f].png`

## Evaluation
```bash
uv run python eval_loop.py [-u] [-b] [-t] [-f]
```

Loops over all queries and infers a template from all templates.  
`-u`: enables unidirectional DTW (restricts template warping). Do not specify -b  
`-b`: enables bidirectional DTW (allows template warping). Do not specify -u
`-t` or `--tune`: enables TF.  
`-f` or `--fill`: fills unvoiced sections of the template vectors.

Outputs percentage of queries for which the top inference was correct (Best Hit Score) and for which the correct template was in the top 10 inferences (Top Ten Score).

Results are saved to and provided in [eval/](eval/).

## Evaluation Plot
```bash
uv run python eval_plot.py
```

Plot evaluation results across all algorithm varieties.

Plot is saved to `plots/eval_plot.png`