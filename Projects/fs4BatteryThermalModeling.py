import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from scipy.integrate import solve_ivp

# Constants Parameters for the differential equation
R = 12.6316
m = 49.9
c = 1000

# Storing constants into a single variable
k = R/(m * c)

df = pl.read_csv("Docs/Columns.csv")

df.select([a for a in df.columns if a.startswith("ACC") and a[9] == "T"]).mean_horizontal()

#Initital conditions
T0 = 22 # Tnitial temperature of the cell in °C
t_span = (0, 60)
t_eval = np.linspace(0, 60, 600)

#differential equation
def thermal_ode(t,I, k):
    """"
    This method returns the differential
    equation given current draw I from
    colums
    """
    I = df.select([a for a in df.columns if a.startswith("ACC") and a[9] == "T"]).mean_horizontal()
    dT_dt = k * (I ** 2)
    return dT_dt

# Numerical Integration using Runge Kutta method (RK45)
solution = solve_ivp(
    thermal_ode, 
    T0,
    method = 'RK45',
    t_eval = t_eval,
    args = (k,),
    rtol = 1e-6, 
    atol= 1e-9
    )
