from firedrake import *

# constants
T = 1         # final time
dt = 0.001     # timestepping length
theta = 1/2     # theta constant
N = 10          # mesh size

Re = Constant(0.01)    # Reynold's num for viscosity

# -------- 
# For MMS 
# --------

P = Constant(10.0)      # pressure constant
H = 1.0                 # height of rectangle, with length = 3H