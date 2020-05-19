################################################################################
### Define preprocessing functions
################################################################################

################################################################################
### 1: Setup
################################################################################

# Import necessary packages
import copy as cp
import joblib as jbl
import multiprocessing as mp
import numpy as np
import pandas as pd
from lmpy.ols import ols
from scipy.stats import randint

################################################################################
### 2: Preprocessing functions
################################################################################

################################################################################
### 2.1: Variance inflation factors (VIFs)
################################################################################


# Define a function to calculate the VIF for a given variable (auxiliary)
def vif_i(X, vlhs, add_intercept=True):
    """ Calculate the VIF for a single variable

    Inputs
    X: pandas DataFrame, input data
    vlhs: string, name of the variable in X for which to calculate VIF
    add_intercept: boolean, if True, adds an intercept to X before calculating
                   VIF

    Outputs
    vlhs: string, same as vlhs; makes it easier to run this in parallel
    VIF: float, VIF for specified variable
    """
    # Set up an OLS model
    model = ols(coef_only=True, add_intercept=add_intercept)

    # Get data for column i only, and data for all other columns
    Xi = X[vlhs]
    Xnoti = X.drop(vlhs, axis=1)

    # Regress variable i on all other variables
    model.fit(Xnoti, Xi)

    # Get the R-squared of that regression
    Ri = model.score(Xnoti, Xi)

    # Store the VIF
    if np.isnan(Ri):
        VIF = np.inf
    else:
        VIF = 1 / (1 - Ri)

    # Return the index and associated VIF
    return vlhs, VIF


# Define a function to calculate the VIFs for a group of variables in a data set
# (auxiliary, uses vif_i())
def vif(X, cols=None, add_intercept=True, corecap=np.inf):
    """ Calculate VIFs for a group of variables in a data set

    Inputs
    X: pandas DataFrame, input data
    cols: list or None, columns in X for which to calculate VIFs; if None, uses
          all columns
    add_intercept: boolean, if True, adds an intercept to X before calculating
                   VIFs
    corecap: integer or np.inf, maximum number of cores to use; if np.inf, uses
             all available cores

    Output
    VIF: len(cols) by 1 DataFrame, VIFs for all specified variables
    """
    # Use all columns of X is cols is None
    if cols is None:
        cols = X.columns

    # Check how many cores can be used
    cores = np.int(np.amin([mp.cpu_count(), corecap]))

    # Get all VIFs in parallel
    res = (
        jbl.Parallel(n_jobs=cores)(
            jbl.delayed(vif_i)(X, c, add_intercept)
            for c in cols
        )
    )

    # Convert the result to a Numpy array
    res = np.array(res)

    # Split it into the indices (variable names) and VIFs
    idx, VIF = res[:,0], res[:,1]

    # Make a DataFrame of VIFs, with the variable names as its index
    VIF = pd.DataFrame(VIF, index=idx, dtype=np.float)

    # Order the variables as in the original cols
    VIF = VIF.loc[cols, :]

    # Return the VIFs
    return VIF


# Define a function to decide which variables to retain, based on their VIFs
# (uses vif())
def retain_vif(X, maxVIF=10, add_intercept=True, fix_seed=False,
               random_state=9969, name_gen='V_', corecap=np.inf, copy=True):
    """ Decides which variables to retain based on their VIFs

    Inputs
    X: n by k array-like, input data (has to be a pandas DataFrame, or something
       which can be cast to a pandas DataFrame)
    maxVIF: float, maximum VIF; retain_vif() will drop variables from X until
            all VIFs are at most maxVIF
    add_intercept: boolean, if True, adds an intercept to X before dropping
                   variables
    fix_seed: boolean, if True, fixes a seed when dropping variables. Since
              retain_vif() drops the variable with the highest VIF at every
              iteration, if two variables have the same (highest) VIF, the
              decision which of them to drop is arbitrary. If fix_seed is True,
              the same one will be dropped every time.
    random_state: integer, seed to use if fix_seed is True
    name_gen: string, generic variable name to use if X is not a pandas
              DataFrame
    corecap: integer or np.inf, maximum number of cores to use; if np.inf, uses
             all available cores
    copy: boolean, if True, copies X before dropping variables

    Outputs
    drop_varnames: list, names of variables to drop from X
    retain_varnames: list, names of variables to retain from X (complement of
                     drop_varnames in the column space of X)
    retain_bool: length k list of booleans; all True elements should be retained
                 (helpful if X was passed as anything but a pandas DataFrame, so
                 variable names are somewhat meaningless)
    """
    # Check whether to copy the input data, and do so if necessary
    if copy:
        X = cp.copy(X)

    # Check whether the input data are a pandas DataFrame
    if not isinstance(X, pd.DataFrame):
        # If not, set them up as one, with generic column names
        X = pd.DataFrame(X)
        X.columns = [name_gen + str(i) for i in np.arange(X.shape[1])]

    # Make a list of elements of X to retain, starting with all of them
    fullcols = X.columns

    # Set up a list of candidate columns to drop
    cands = fullcols

    # Set up a list of all safe columns (columns with VIFs below maxVIF)
    safecols = []

    # Calculate VIFs
    VIF = vif(X=X, add_intercept=add_intercept, corecap=corecap)

    # Check whether any VIFs exceed tolerance
    if any(VIF.values > maxVIF):
        # If so, set up a convergence flag, as False
        converged = False
    else:
        # Otherwise, set that flag to True
        converged = True

    # Iterate until all VIFs are below tolerance
    while not converged:
        # Get the largest elements
        sup = VIF.loc[VIF.iloc[:,0] == VIF.iloc[:,0].max(), 0]

        # Pick one of them at random to kick out. (The .sample() picks an
        # element, the .index returns a column name (because that's how vif()
        # indexes its results), and the list() just helps ensure the result is a
        # plain old list, not a pandas index.)
        if fix_seed:
            drop = list(sup.sample(n=1, random_state=random_state).index)
        else:
            drop = list(sup.sample(n=1).index)

        # Drop the chosen column
        X = X.drop(drop, axis=1)

        # Mark all safe columns
        safe = [c for i, c in enumerate(cands) if VIF.iloc[i,0] <= maxVIF]

        # Add them to the list of safe columns
        safecols = safecols + safe

        # Save the remaining candidates
        cands = [c for c in cands if c not in drop + safecols]

        # Check whether there are candidates left. (This will not be True if the
        # last variable was dropped on this iteration. Without this check, that
        # would result in cands being empty, and that will cause an error in
        # vif(), so the check is necessary.)
        if len(cands) > 0:
            # Recalculate VIFs for the remaining variables
            VIF = (
                vif(X=X,cols=cands,add_intercept=add_intercept,corecap=corecap)
            )
        else:
            # Otherwise, set the convergence indicator to True
            converged = True

        # Check whether convergence has been achieved
        if all(VIF.values <= maxVIF):
            # Set the convergence flag
            converged = True

    # Make a list of all variables which should be dropped. (This specific list
    # comprehension preserves the order of the variable names, i.e. the order
    # will be the same as for the original X input.)
    drop_varnames = [c for c in fullcols if c not in X.columns]

    # Get the names of all surviving variables
    retain_varnames = [c for c in X.columns]

    # Also get a boolean indicator for which variables survived (I usually find
    # retain_varnames easier to work with, but this can be handy if variables do
    # not really have informative names.)
    retain_bool = [c in X.columns for c in fullcols]

    # Return the retain list
    return drop_varnames, retain_varnames, retain_bool
