import sys
import numpy as np
filename = sys.argv[1];
the_arr = np.load(filename)
print(the_arr)
print(np.sum(the_arr))
