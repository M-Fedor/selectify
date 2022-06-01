import numpy as np
import sys


input_name = sys.argv[1]
output_name = sys.argv[2]

data = np.loadtxt(input_name, delimiter=',')
data = np.save(output_name, data)
