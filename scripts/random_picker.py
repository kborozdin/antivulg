#!/usr/bin/python3

import numpy as np

CUT = 20

lines = []
with open('filtered_comments.csv') as f:
    for line in f:
        if len(line) >= CUT:
            lines.append(line)

CNT = 100
#CNT = 10000
BLOCK = 100
#BLOCK = 100500
np.random.seed(31415 + 1234)

for fid in ['_train2']:
    fname = 'val{}.txt'.format(str(fid))
    #fname = 'unlabeled10k.txt'
    picked_lines = np.random.choice(lines, size=CNT)
    with open(fname, 'w') as f:
        for raw_line in picked_lines:
            line = raw_line.strip()
            for start in range(0, len(line), BLOCK):
                length = min(BLOCK, len(line) - start)
                f.write(line[start:start+length] + '\n')
                f.write('0' * length + '\n')
            f.write('\n')
