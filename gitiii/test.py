import numpy as np
from scipy.stats import linregress, t, nct


def calculate_power(x, y, beta1_alt=1, alpha=0.05):
    """
    Calculate the power of a test for H0: beta1 = 0 vs H1: beta1 = beta1_alt
    given two numpy arrays x and y representing the predictor and response variables
    for a simple linear regression without an intercept.

    Parameters:
    - x: numpy array of predictor values
    - y: numpy array of response values
    - beta1_alt: float, alternative hypothesis value for beta1 (default is 1)
    - alpha: significance level (default is 0.05)

    Returns:
    - z: float, the z-score based on the correlation coefficient (r)
    - power: float, the power of the test
    """
    # Estimate beta1 directly without an intercept
    beta1_hat = np.sum(x * y) / np.sum(x ** 2)

    # Calculate residuals and standard error of beta1_hat
    residuals = y - beta1_hat * x
    sse = np.sum(residuals ** 2)
    std_err = np.sqrt(sse / (len(x) - 1)) / np.sqrt(np.sum(x ** 2))

    # Calculate the correlation coefficient (r)
    r_value = np.corrcoef(x, y)[0, 1]

    # Return (0, 0) if r is extremely high, indicating perfect correlation
    if abs(r_value) >= 0.999999:
        return 0, 0

    # Calculate the z-score based on the correlation coefficient
    z = r_value * ((len(x) - 1) ** 0.5) / (1 - r_value ** 2) ** 0.5

    # Degrees of freedom (n - 1 for no intercept)
    df = len(x) - 1

    # Calculate the non-centrality parameter (ncp) for the alternative hypothesis
    ncp = beta1_alt / std_err

    # Calculate the critical t value for a two-tailed test
    t_critical = t.ppf(1 - alpha / 2, df)

    # Calculate the power of the test using the non-central t-distribution
    power = 1 - (nct.cdf(t_critical, df, ncp) - nct.cdf(-t_critical, df, ncp))

    return z, power

# Sample data
x = np.array([1, 2, 3, 4, 5,6])
y = np.array([2, 4, 6, 8, 12,-13])

# Calculate power
power = calculate_power(x, y)
print(f"Power of the test: {power}")