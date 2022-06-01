import numpy as np
import sys


file_name = sys.argv[1]

data = np.loadtxt(file_name, delimiter=',')
data = np.random.permutation(data)

with open(file_name + '.rand', 'w') as output:
    for line in data:
        for value in line[:-1]:
            print("%s," % str(value), file=output, end='')
        print("%s" % str(line[-1]), file=output)
