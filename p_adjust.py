""" Functions to adjust p-values """

################################################################################
# 1: Setup
################################################################################

# Import necessary modules
import numpy as np
import pandas as pd
from lmpy.ols import cvec

################################################################################
# 2: Define functions
################################################################################

################################################################################
# 2.1: Bonferroni correction
################################################################################


# Define the Bonferroni correction
def bonferroni(p, varnames=None, gen_name='X'):
    """ Run Bonferroni correction

    Inputs
    p: k by 1 vector-like; p-values to adjust
    varnames: k by 1 vector-like or None; variable names
    gen_name: string; generic variable name prefix, used if varnames is None

    Outputs
    padjs: k by 1 vector; adjusted p-values
    """
    # Get the number of hypotheses M
    M = len(p)

    # Calculate adjusted p-values
    padj = cvec(p * M)

    # Make sure these are no larger than one
    toolarge = padj[:, 0] > 1
    padj[toolarge, 0] = 1

    # Check whether varnames are missing, but p is a pandas object
    if (varnames is None) and isinstance(p, (pd.Series, pd.DataFrame)):
        # If so, use its index as variable names
        varnames = p.index
    elif varnames is None:
        # Otherwise, make a generic index
        varnames = [gen_name+str(i+1) for i in np.arange(M)]

    # Convert the results to a DataFrame
    padj = pd.DataFrame(padj, index=varnames, columns=['Bonferroni p-value'])

    # Return the results
    return padj

################################################################################
# 2.1: Holm-Bonferroni correction
################################################################################


# Define the Holm-Bonferroni correction
def holm_bonferroni(p, varnames=None, gen_name='X'):
    """ Run Holm-Bonferroni correction

    Inputs
    p: k by 1 vector-like; p-values to adjust
    varnames: k by 1 vector-like or None; variable names
    gen_name: string; generic variable name prefix, used if varnames is None

    Outputs
    padjs: k by 1 vector; adjusted p-values
    """
    # Get number of hypotheses M
    M = len(p)

    # Set up the adjusted p-values, as a proper column vector based on p
    padj = cvec(p)

    # Get an index which would sort the p-values
    sortidx = np.argsort(padj[:, 0])

    # Go through all p-values
    for k in np.arange(M):
        # Get the position of the k-th smallest p-value in the original vector
        # of p-values
        idxk = sortidx[k]

        # Adjust the p-value
        padj[idxk, 0] = np.amin([padj[idxk, 0] * (M - k), 1])

        # Check whether this is anyone but the smallest p-value
        if k > 0:
            # Get the index of the preceding p-value
            idxlast = sortidx[k-1]

            # Enforce monotonicity
            padj[idxk, 0] = np.amax([padj[idxk, 0], padj[idxlast, 0]])

    # Check whether varnames are missing, but p is a pandas object
    if (varnames is None) and isinstance(p, (pd.Series, pd.DataFrame)):
        # If so, use its index as variable names
        varnames = p.index
    elif varnames is None:
        # Otherwise, make a generic index
        varnames = [gen_name+str(i+1) for i in np.arange(len(p))]

    # Convert the results to a DataFrame
    padj = pd.DataFrame(
        padj, index=varnames, columns=['Holm-Bonferroni p-value']
    )

    # Return the results
    return padj
