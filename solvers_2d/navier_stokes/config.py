from firedrake import *

# constants
T = 2           # final time
dt = 0.1        # timestepping length
theta = 1/2     # theta constant
N = 10          # mesh size

Re = Constant(1.0)    # Reynold's num for viscosity

# functions
ufl_v = as_vector([1, 0])           # velocity ic
ufl_p = Constant(0.0)               # pressure ic
ufl_f = as_vector([0, 0])           # source term f
ufl_g = as_vector([0, 0])           # bdy condition g