from data_loader import MIRQBSHDataset
from preprocessing import preprocess_query
from dtw_wrapper import DTWWrapper
import matplotlib.pyplot as plt
import sys

args = sys.argv[1:]

uni_w = '-u' in args or '--uni'  in args
bi_w  = '-b' in args or '--bi'   in args
assert (not uni_w) or (not bi_w), "Warp cannot be uni- and bi-directional!"

tune  = '-t' in args or '--tune' in args
fill  = '-f' in args or '--fill' in args
if '-q' in args: # bad arg checking
    assert args.size >= args.index('-q'), "Must pass a query number after '-q'!"
    query_file = int(args[args.index('-q') + 1])
else:
    query_file = 15

print(f'Unidirectional DTW: {uni_w}')
print(f'Bidirectional DTW: {bi_w}')
print(f'Tune-Following: {tune}')
print(f'Template Unvoiced Filling: {fill}')

dataset = MIRQBSHDataset("./data/MIR-QBSH")
dtw_computer = DTWWrapper(dataset)

query_path = dataset.query_files[query_file]
raw_query = dataset.load_query_pv(query_path)
query_processed = preprocess_query(raw_query)

template = dataset.load_template_midi(query_path.stem)

predicted_template = dtw_computer.match_query_in_database(
    query_in=query_processed,
    uni_w=uni_w,
    bi_w=bi_w,
    tune=tune,
    fill=fill
)

print(f"Predicted: {predicted_template}, Actual: {query_path.stem}")

# nice figure strings
f_suffix = ''
q_suffix = ''
i_suffix = ''
t_suffix = ''
if uni_w:
    q_suffix += '_warped'
    i_suffix += '_warped'
    f_suffix += 'u'
elif bi_w:
    q_suffix += '_warped'
    t_suffix += '_warped'
    i_suffix += '_warped'
    f_suffix += 'b'
if tune:
    q_suffix += '_tuned'
    f_suffix += 't'
if fill:
    t_suffix = '_filled'
    f_suffix += 'f'
if f_suffix != '':
    f_suffix = '_' + f_suffix

if query_path.stem == predicted_template[0]:
    t1_type = '(Best Hit Guess)'
    t2_type = '(Guess #2)'
elif query_path.stem in predicted_template:
    t1_type = f'(Guess #{1 + predicted_template.index(query_path.stem)})'
    t2_type = '(Best Hit Guess)'
else:
    t1_type = '(Not Guessed)'
    t2_type = '(Best Hit Guess)'


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

# top subplot: processed query and correct template (whether guessed or not)
q1, t1, i1 = dtw_computer.fit_template(
    query_in=query_processed,
    template_in=template,
    uni_w=uni_w, bi_w=bi_w, tune=tune, fill=fill)

ax1.plot(t1, label=f'Template{t_suffix}', alpha=0.7)
if tune:
    ax1.plot(i1, label=f'Query{i_suffix}', alpha=0.7)
ax1.plot(q1, label=f'Query{q_suffix}', alpha=0.7)
ax1.legend()
ax1.set_xlabel('Frame')
ax1.set_ylabel('MIDI Note')
ax1.set_title(f'Query Matched to Correct Template {t1_type}')

# bottom subplot: processed query and top incorrect guess
q2, t2, i2 = dtw_computer.fit_template(
    query_in=query_processed,
    template_in=dataset.load_template_midi(
        predicted_template[
            1 if query_path.stem == predicted_template[0] else 0
        ]),
    uni_w=uni_w, bi_w=bi_w, tune=tune, fill=fill)

ax2.plot(t2, label=f'Template{t_suffix}', alpha=0.7)
if tune:
    ax2.plot(i2, label=f'Query{i_suffix}', alpha=0.7)
ax2.plot(q2, label=f'Query{q_suffix}', alpha=0.7)
ax2.legend()
ax2.set_xlabel('Frame')
ax2.set_ylabel('MIDI Note')
ax2.set_title(f'Query Matched to Best Incorrect Template Guess {t2_type}')

plt.savefig(f'inference_q{query_file}{f_suffix}.png')
print(f"Saved visualization to inference_q{query_file}{f_suffix}.png")
plt.show()
