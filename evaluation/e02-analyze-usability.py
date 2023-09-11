# Import libraries
import numpy as np
import pandas as pd

# Load data
key = pd.read_csv('./usability/answer-key.txt', header=None, names=['Question', 'Group', 'Type', 'Map', 'Answer'])
res = pd.read_csv('./usability/results.csv', skiprows=[1,2])

# Determine correct answers
answers = {}
accuracy = {}
groups = {}
for index, row in key.iterrows():
    answers[row['Question']] = row['Answer']
    accuracy[row['Question']] = 0
    if (not row['Group'] in groups): groups[row['Group']] = []
    groups[row['Group']].append(row['Question'])

# Measure accuracy
rows = 0
for index, row in res.iterrows():
    for i in range(12):
        choice = row[str(i+1)]
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
for group in groups:
    questions = groups[group]
    correct, total = 0, 0
    for question in questions:
        correct += accuracy[question]
        total += rows
    pct = 100 * (correct / total)
    print(group.capitalize(), "\t", str(round(pct, 2)) + "%")
