import matplotlib.pyplot as plt
import numpy as np

alg = "DTWTF"

eval_stem = 'eval/eval_'
alg_types = [
    "btf", "bf",
    "utf", "uf",
    "tf",  "f",
    "bt",  "b",
    "ut",  "u",
    "t",   ""
]

alg_labels = [
    "Bi-DTW Tune Fill",  "Bi-DTW Fill",
    "Uni-DTW Tune Fill", "Uni-DTW Fill",
    "Tune Fill",         "Fill",
    "Bi-DTW Tune",       "Bi-DTW",
    "Uni-DTW Tune",      "Uni-DTW",
    "Tune",              "Base"
]

top_ten = []
best_hit = []

for idx in range(len(alg_types)):
    alg_type = alg_types[idx]
    alg_label = alg_labels[idx]
    eval_path = eval_stem + alg_type + '.txt'

    eval_file = open(eval_path, 'r') 
    eval_txt = eval_file.readlines()

    top_ten.append(float(eval_txt[10]))
    best_hit.append(float(eval_txt[12]))

    eval_file.close()

fig, ax = plt.subplots(figsize=(12, 9))

height = 0.7
offset = 0.35
y_pos = np.arange(len(alg_labels)) * 2
hbars_tt = ax.barh(y_pos - offset, top_ten, height, label='Top Ten Score')
hbars_bh = ax.barh(y_pos + offset, best_hit, height, label='Best Hit Score')
ax.set_yticks(y_pos, labels=alg_labels)
ax.invert_yaxis()

ax.set_title('Algorithm Evaluation Scores')
ax.set_xlabel('% Score')
ax.legend()

ax.bar_label(hbars_tt, fmt='%.3f')
ax.bar_label(hbars_bh, fmt='%.3f')
ax.set_xlim(right=100)

plt.savefig(f'plots/eval_plot.png')
print("Saved visualization to plots/eval_plot.png")
plt.show()