import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import coint, adfuller


sns.set(style="whitegrid")

# Function to generate random data
def generate_data(params):
    mu = params[0]
    sigma = params[1]
    return np.random.normal(mu, sigma)

# Generate stationary and non-stationary series
params = [0, 1]
T = 100

A = pd.Series(index=range(T), dtype=float)
A.name = "A"

for t in range(T):
    A[t] = generate_data(params)

B = pd.Series(index=range(T), dtype=float)
B.name = "B"

for t in range(T):
    params = [t * 0.1, 1]
    B[t] = generate_data(params)

# Plot results
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

ax1.plot(A)
ax2.plot(B)
ax1.legend(['Series A'])
ax2.legend(['Series B'])
ax1.set_title('Stationary')
ax2.set_title('Non-Stationary')


# Augmented Dickey Fuller (ADF) test for stationarity 
def stationarity_test(X, cutoff = 0.01):
    # H_0 in adfuller is unit root exists (non-stationary)
    pvalue = adfuller(X)[1]
    if pvalue > cutoff:
        print(f"There is not sufficient evidence to reject H0 since {pvalue} > {cutoff}. Therefore, likely non stationary.\n")
    else:
        print(f"There is sufficient evidence to reject H0 since {pvalue} < {cutoff}. Therefore, likely stationary.")

stationarity_test(A)
stationarity_test(B)


## Cointegeration
