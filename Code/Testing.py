import random
import re
import math

sloop = "GCGGGUUAUGAGGCUCCAGAGCAAUUGAGACUUAACUU"
sloop_db = "(((((..(.(((....))).)..)))))......."
bases = 'AUCG'
char_list = ['(', ')']

print(sloop)

bounds_list = []
for char in char_list:
    repeats = re.findall(r'\{}+'.format(char), sloop_db)
    max_reps = len(max(repeats))
    max_pat = max(repeats)
    for pattern in re.finditer('\{}'.format(char) * max_reps, sloop_db):
        bounds_list.append((pattern.start(), pattern.end()))

ran_bounds = random.choice(bounds_list)
mid_bound = math.floor((ran_bounds[0] + ran_bounds[1]) / 2)

sloop = sloop[:mid_bound] + random.choice(bases) + sloop[mid_bound:]
print(sloop, sloop[mid_bound])

# x = [(m.start(), m.stop()) for m in re.finditer('(' * max_reps, sloop_db)]
# print(x)
# numString = '1234555325146555'
# fives = re.findall(r'5+', numString)
# print(len(max(fives)))          # most repetitions
# print(fives.count(max(fives)))  # number of times most repetitions occurs
