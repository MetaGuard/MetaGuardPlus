# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
key = pd.read_csv('./usability/answer-key.txt', header=None, names=['Question', 'Group', 'Type', 'Map', 'Answer'])
res = pd.read_csv('./usability/results.csv', skiprows=[1,2])

# Print results for a subset
def get_res(res):
    # Determine correct answers
    answers = {}
    accuracy = {}
    groups = {}
    totals = {}
    for index, row in key.iterrows():
        answers[row['Question']] = row['Answer']
        accuracy[row['Question']] = 0
        totals[row['Question']] = 0
        if (not row['Group'] in groups): groups[row['Group']] = []
        groups[row['Group']].append(row['Question'])

    # Measure accuracy
    rows = 0
    for index, row in res.iterrows():
        for i in range(12):
            choice = row[str(i+1)]
            if (not pd.isna(choice)):
                totals[i] += 1
                if ("Recording A" in choice):
                    ans = 0
                elif ("Recording B" in choice):
                    ans = 1
                elif ("Recording C" in choice):
                    ans = 2
                elif ("Recording D" in choice):
                    ans = 3
                else:
                    ans = -1
                if (ans == answers[i]):
                    accuracy[i] += 1
        rows += 1

    # Print results
    print("N =", rows)
    res = {}
    for group in groups:
        questions = groups[group]
        correct, total = 0, 0
        for question in questions:
            correct += accuracy[question]
            total += totals[question]
        pct = 100 * (correct / total)
        res[group] = pct
        print(group, "\t", str(round(pct, 2)) + "%")
    return res, rows

print("-- All --")
all, Na = get_res(res)

print("\n-- Novices --")
novices, Nn = get_res(res.loc[res['Beat Saber Playtime'] != "Yes, I have played Beat Saber for more than 100 hours."])

print("\n-- Experts --")
experts, Ne = get_res(res.loc[res['Beat Saber Playtime'] == "Yes, I have played Beat Saber for more than 100 hours."])

bs, be = '$\\bf{', '}$'
sets = { f"{bs}All{be} (N={Na})": all, f"{bs}Novices{be} (N={Nn})": novices, f"{bs}Experts{be} (N={Ne})": experts }
fig, ax = plt.subplots(layout='constrained')
x = np.arange(len(all.keys()))
w = 0.25
for i, set in enumerate(sets.keys()):
    vals = sets[set].values()
    rects = ax.bar(x + w * i, vals, w, label=set)
    ax.bar_label(rects, padding=3, labels=[str(round(p)) + "%" for p in vals], fontsize=9)

ax.set_ylabel('Distinguishability (%)')
ax.set_title(f'{bs}Motion~Distinguishability~vs.~Anonymization~Technique{be}')
p = '{+}'
keys = {
    "control": f"{bs}Unmodified{be}\n(Neg. Control)",
    "artificial": f"{bs}Artificial{be}\n(Pos. Control)",
    "metaguard": f"{bs}MetaGuard{be}\n(Treatment 1)",
    "metaguardplus": f"{bs}MetaGuard{p}{p}{be}\n(Treatment 2)",
}
ax.set_xticks(x + w, [keys[key] for key in all.keys()])
ax.legend(loc='upper left')
ax.set_ylim(0, 100)

plt.savefig("./usability/results.pdf",bbox_inches='tight')
plt.savefig("./usability/results.png",bbox_inches='tight')
plt.show()
