################################################################################
### 1: Setup
################################################################################

# Import necessary packages
import numpy as np
import pandas as pd
import scipy.linalg
from hdmpy import cvec
from scipy.stats import norm

################################################################################
### 2: Auxiliary functions
################################################################################

# Currently, there are none

################################################################################
### 3: Regression models
################################################################################


# Define OLS model
class ols():
    """ Runs OLS regressions """


    # Define initialization function
    def __init__(self, name_gen_X='X', name_gen_y='y', add_intercept=True,
                 name_gen_icept='(Intercept)', coef_only=False,
                 covariance_estimator=None, level=.05, verbose=True,
                 fprec=np.float32, nround=4):
        """ Initialize ols() class

        Inputs
        name_gen_x: String, generic name prefix for X (RHS) variables, used if
                    no variable names are provided in ols.fit()
        name_gen_y: String, generic name prefix for y (LHS) variable, used if no
                    variable name is provided in ols.fit()
        add_intercept: Boolean, if True, an intercept will be added to the model
        name_gen_icept: String, name to use for the intercept variable
        coef_only: Boolean, if True, the model only calculates coefficients (and
                   not the covariance matrix, which is computationally costly)
        covariance_estimator: String or None, covariance estimator to use.
                              Possible choices are:

                              'homoskedastic': The homoskedastic covariance
                                               estimator
                              'hc1': The heteroskedasticity robust covariance
                                     estimator of MacKinnon and White (1985)
                              'cluster': The cluster robust covariance estimator
                                         proved to be unbiased in Williams
                                         (2000)

                              If None, uses the default provided in
                              ols.ols_cov(), which is either 'hc1', if cluster
                              IDs were not provided in ols.fit(), or 'cluster',
                              if cluster IDs were provided
        level: Scalar in [0,1], level for confidence intervals
        verbose: Boolean, if True, some notes and warnings are printed (e.g. if
                 X'X is not invertible, and a pseudo inverse needs to be used)
        fprec: Float data type, all floats will be cast to this type. The
               default value is np.float32, which can speed up NumPy's matrix
               multiplication in some settings.
        nround: Integer, results of ols.summarize() will be rounded to this
                number of decimal points
        """
        # Instantiate parameters
        #
        # Instantiate verbose flag
        self.verbose = verbose

        # Instantiate float precision data type
        self.fprec = fprec

        # Instantiate parameters for ols.fit()
        self.name_gen_X = name_gen_X
        self.name_gen_y = name_gen_y
        self.add_icept = add_intercept
        self.name_gen_icept = name_gen_icept
        self.coef_only = coef_only

        # Parameters for self.ols_ci()
        self.level = level

        # Parameters for self.summarize()
        self.nround = nround

        # Instantiate variables created by various methods
        #
        # Variables from ols_fit()
        self.X = None
        self.y = None
        self.clustvar = None
        self.names_X = None
        self.name_y = None
        self.XX = None
        self.XXinv = None
        self.coef = None
        self.est = None

        # Variables created by self.ols_cov()
        self.cov_est = covariance_estimator
        self.U_hat = None
        self.V_hat = None
        self.se = None

        # Variables created by self.ols_t()
        self.t = None

        # Variables created by self.ols_ci()
        self.ci = None
        self.names_ci = (
            ['{}%'.format(q*100) for q in [self.level/2, 1 - self.level/2]]
        )

        # Variables created by self.ols_p()
        self.p = None

        # Variables created by self.summarize()
        self.regtable = None

        # Variables created by self.simulate()
        self.sim_coef = None
        self.sim_X = None
        self.sim_residuals = None
        self.sim_y = None


    # Define a function to fit the model
    def fit(self, y, X, clusters=None, names_X=None, name_y=None,
            name_gen_X=None, name_gen_y=None, add_intercept=None,
            coef_only=None):
        """ Fit OLS model

        Inputs
        y: n by 1 vector-like, outcome variable
        X: n by k matrix-like, RHS variables
        cluster: n by 1 vector-like or None, cluster IDs
        names_X: length n list or None, names for variables in X. If names_X is
                 None and X is a pandas DataFrame or Series, column names will
                 be used.
        name_y: length n list or None, name for outcome variable. As with X,
                this can be inferred if y is a pandas Series or DataFrame.
        name_gen_x: String or None, see __init()__; if None, uses the value
                    provided in __init()__
        name_gen_y: String or None, see __init()__; if None, uses the value
                    provided in __init()__
        add_intercept: Boolean or None, see __init()__; if None, uses the value
                       provided in __init()__
        coef_only: Boolean or None, see __init()__; if None, uses the value
                   provided in __init()__
        """
        # Check whether the generic name for X variables was changed
        if name_gen_X is not None:
            # If so, adjust self.name_gen_X
            self.name_gen_X = name_gen_X

        # Check whether the generic name for the y variable was changed
        if name_gen_y is not None:
            # If so, adjust self.name_gen_y
            self.name_gen_y = name_gen_y

        # Check whether the intercept paramter was changed
        if add_intercept is not None:
            # If so, adjust it
            self.add_icept = add_intercept

        # Check whether the coefficients only paramter was changed
        if coef_only is not None:
            # If so, adjust it
            self.coef_only = coef_only

        # Variables names
        #
        # Check whether names for X were provided
        if names_X is not None:
            # If so, adjust names_X
            self.names_X = names_X

        # Alternatively, check whether any data were provided as pandas objects,
        # and get their names if necessary; otherwise, instantiate names
        # automatically
        #
        # Check whether X is a pandas DataFrame
        elif isinstance(X, pd.DataFrame):
            # If so, set names_X to the the column names
            self.names_X = X.columns

        # Check whether X is a pandas Series
        elif isinstance(X, pd.Series):
            # If so, set names_X to the Series name
            self.names_X = X.name

        # If all else fails...
        else:
            # ... use generic names
            self.names_X = [self.name_gen_X+str(i+1)
                            for i in np.arange(X.shape[1])]

        # Check whether names for y were provided
        if name_y is not None:
            # If so, adjust name_y
            self.name_y = name_y

        # Check whether y is a pandas Data Frame and name_y was not provided
        elif isinstance(y, pd.DataFrame):
            # If use, set name_y to the column name
            self.name_y = y.columns[0]
        # Check whether y is a pandas Series and name_y was not provided
        elif isinstance(y, pd.Series):
            # If so, set name_y to the Series name
            self.name_y = y.name
        # If all else fails...
        else:
            # ... use generic names
            self.name_y = self.name_gen_y

        # Instantiate data matrices
        #
        # Check whether to add an intercept
        if self.add_icept:
            # If so, set up an intercept
            cons = np.ones(shape=(X.shape[0], 1))

            # Add it to the data matrix
            self.X = np.concatenate([cons, X], axis=1)

            # Add the intercept to the variables names
            self.names_X = [self.name_gen_icept] + list(self.names_X)
        else:
            # Otherwise, just instantiate the X data as is
            self.X = X

        # For speed considerations, make sure these are self.fprec types
        self.X = np.array(self.X).astype(self.fprec)

        # Get number of observations n and variables p
        self.n, self.k = self.X.shape

        # Instantiate y data elements
        self.y = cvec(y).astype(self.fprec)

        # Check whether a cluster variable was provided
        if clusters is not None:
            # If so, adjust the cluster variable (here, using integers makes
            # sense, because these might be used as an indexer at some point)
            self.clustvar = cvec(clusters).astype(np.int)

        # Calculate X'X
        self.XX = self.X.T @ self.X

        # Calculate (X'X)^(-1)
        #
        # This may fail if that matrix is not invertible, so use try-except
        try:
            # Get the inverse
            self.XXinv = scipy.linalg.inv(self.XX)
        # Catch linear algebra errors
        except np.linalg.LinAlgError as e:
            # Check whether the error was due to a singular X'X matrix
            if 'singular matrix' in str(e):
                # If so, use the Moore-Penrose pseudo inverse for real Hermitian
                # matrices
                self.XXinv = scipy.linalg.pinvh(self.XX)

                # Check whether to be talkative
                if self.verbose:
                    print("\nNote in ols(): X'X matrix was not invertible,",
                          'Moore-Penrose pseudo-inverse had to be used')

        # Calculate coefficient vector
        self.coef = self.XXinv @ (self.X.T @ self.y)

        # Get residuals
        self.U_hat = self.y - self.X @ self.coef

        # Check whether to calculate anything besides the coefficients
        if not self.coef_only:
            # Get the covariance
            self.ols_cov()

            # Get t-statistics
            self.ols_t()

            # Get confidence intervals
            self.ols_ci()

            # Get p-values
            self.ols_p()
        else:
            # Otherwise, set all other results to NAN. (This is important for
            # coef_only=True to provided a speed-up. If this does not happen,
            # some of the pandas DataFrames containing results are created
            # automatically, and automatically inferring their shape and size
            # takes a while. Without pre-setting these here, coef_only=True
            # actually slows down the program.)
            self.se = np.zeros(shape=(self.k,1)) * np.nan
            self.t = np.zeros(shape=(self.k,1)) * np.nan
            self.ci = np.zeros(shape=(self.k,2)) * np.nan
            self.p = np.zeros(shape=(self.k,1)) * np.nan
            self.V_hat = np.zeros(shape=(self.k,self.k)) * np.nan

        # Combine results into a dictionary
        self.est = {
            'coefficients': pd.DataFrame(self.coef, index=self.names_X,
                                         columns=['Estimated coefficient']),
            'se': pd.DataFrame(self.se, index=self.names_X,
                               columns=['Standard error']),
            'covariance estimator': self.cov_est,
            't': pd.DataFrame(self.t, index=self.names_X,
                              columns=['t-statistic']),
            'ci': pd.DataFrame(self.ci, index=self.names_X,
                               columns=self.names_ci),
            'level': self.level,
            'p': pd.DataFrame(self.p, index=self.names_X, columns=['p-value']),
            'covariance matrix': pd.DataFrame(self.V_hat, index=self.names_X,
                                              columns=self.names_X),
            'residuals': pd.DataFrame(self.U_hat, columns=['Residuals']),
            'clusters': pd.DataFrame(self.clustvar, columns=['Cluster ID'])
        }


    # Define a function to calculate the covariance matrix plus standard errors
    def ols_cov(self, covariance_estimator=None):
        """ Calculate covariance matrix and standard errors

        Input
        covariance_estimator: String or None, see __init()__; if None, uses the
                              value provided in __init()__
        """
        # Check whether covariance_estimator was changed from the default None
        if covariance_estimator is not None:
            # If so, set cov_est to the specified covariance estimator
            self.cov_est = covariance_estimator

        # Otherwise, check whether the original value provided to __init__() was
        # left at the default None, and no cluster IDs were provided
        elif (self.cov_est is None) and (self.clustvar is None):
            # If so, use HC1 as the default covariance estimator
            self.cov_est = 'hc1'

        # Otherwise, check whether the original value provided to __init__() was
        # left at the default None, and cluster IDs were provided
        elif (self.cov_est is None) and (self.clustvar is not None):
            # If so, use clustered standard errors
            self.cov_est = 'cluster'

        # Check whether clusters were provided, but a non-clustered covariance
        # estimator is being used, and the class is set to be talkative
        if (
                (self.clustvar is not None)
                and (self.cov_est != 'cluster')
                and self.verbose
        ):
            print('\nNote in ols(): Cluster IDs were provided, but a',
                  'non-clustered covariance estimator is being used')

        # Check which covariance estimator to use
        #
        # Homoskedastic
        if self.cov_est.lower() == 'homoskedastic':
            # For the homoskedastic estimator, just calculate the standard
            # variance
            self.V_hat = (
                (1 / (self.n - self.k))
                * self.XXinv * (self.U_hat.T @ self.U_hat)
            )

        # HC1
        elif self.cov_est.lower() == 'hc1':
            # Calculate component of middle part of EHW sandwich,
            # S_i = X_i u_i, which makes it very easy to calculate
            # sum_i X_i X_i' u_i^2 = S'S
            S = (self.U_hat @ np.ones(shape=(1,self.k))) * self.X

            # Calculate EHW variance/covariance matrix
            self.V_hat = (
                (self.n / (self.n - self.k))
                * self.XXinv @ (S.T @ S) @ self.XXinv
            )

        # Clustered errors
        elif self.cov_est.lower() == 'cluster':
            # Calculate number of clusters
            J = len(np.unique(self.clustvar[:,0]))

            # Same thing as S above, but needs to be a DataFrame, because pandas
            # has the groupby method, which is needed in the next step
            S = pd.DataFrame((self.U_hat @ np.ones(shape=(1,self.k))) * self.X)

            # Sum all covariates within clusters
            S = S.groupby(self.clustvar[:,0], axis=0).sum()

            # Convert back to a NumPy array
            S = np.array(S).astype(np.float32)

            # Calculate cluster-robust variance estimator
            self.V_hat = (
                (self.n / (self.n - self.k) ) * (J / (J - 1))
                * self.XXinv @ (S.T @ S) @ self.XXinv
            )

        # Some other unknown method
        else:
            # Print an error message
            raise ValueError('Error in ols.fit(): The specified covariance '
                             + 'estimator could not be recognized; please '
                             + 'specify a valid estimator')

        # Replace NaNs as zeros (happens if division by zero occurs)
        #self.V_hat[np.isnan(V_hat)] = 0

        # Calculate the standard errors for all coefficients
        self.se = cvec(np.sqrt(np.diag(self.V_hat))).astype(self.fprec)


    # Define a function to calculate t-statistics
    def ols_t(self):
        """ Calculate t-statistics """
        # Calculate t-statistics (I like having them as a column vector, but
        # to get that, I have to convert the square root of the diagonal
        # elements of V_hat into a proper column vector first)
        self.t = self.coef / self.se


    # Define a function to calculate confidence intervals
    def ols_ci(self, level=None):
        """ Calculate confidence intervals

        Input
        level: Scalar in [0,1] or None, see __init()__; if None, uses the value
               provided in __init()__
        """
        # Check whether level was changed from the default None
        if level is not None:
            # If so, adjust the level
            self.level = level

        # See which quantiles are needed for critical values
        quants = [self.level/2, 1 - self.level/2]

        # Make a row vector of critical values
        cval = cvec(norm.ppf(quants)).astype(self.fprec).T

        # Calculate confidence intervals
        self.ci = (
            self.coef @ np.ones(shape=(1,2))
            + (self.se @ np.ones(shape=(1,2)))
            * (np.ones(shape=(self.k,1)) @ cval)
        )

        # Make a two element list of names for the upper and lower bound
        self.names_ci = ['{}%'.format(q*100) for q in quants]


    # Define a function to calculate p-values
    def ols_p(self):
        """ Calculate p-values """
        # Calculate p-values
        self.p = 2 * (1 - norm.cdf(np.abs(self.t)))


    # Define a function to return a 'classic' regression table
    def summarize(self):
        """ Produce a pandas DataFrame containing model results """
        # Make a regression table DataFrame
        self.regtable = (
            pd.concat(
                [self.est['coefficients'],
                 self.est['se'],
                 self.est['ci'],
                 self.est['t'],
                 self.est['p'],
                ], axis=1
            )
        )

        # Round the results
        self.regtable = self.regtable.round(self.nround)

        # Add a name to the DataFrame
        self.regtable.name = 'Outcome variable: ' + self.name_y

        # Return the result, to make it easily printable
        return self.regtable


    # Define a function which generates a new draw of data (for bootstrapping)
    def simulate(self, residuals, X=None, coef=None):
        # Check whether coef was left at the default None
        if coef is None:
            # If so, set the simulation coefficients to the fitted coefficients
            self.sim_coef = self.coef
        else:
            # Otherwise, set the simulation coefficients to a proper column
            # vector based on coef
            self.sim_coef = cvec(coef).astype(self.fprec)

        # Check whether X was left at the default None
        if X is None:
            # If so, set the simulation X to the existing data
            self.sim_X = self.X
        else:
            # Otherwise, set the simulation X to the provided data, but make
            # sure they're a 32 bit NumPy array
            self.sim_X = np.array(X).astype(self.fprec)

        # Adjust the simulation residuals
        self.sim_residuals = cvec(residuals).astype(self.fprec)

        # Generate simulated data
        self.sim_y = self.sim_X @ self.sim_coef + self.sim_residuals

        # Return the result
        return self.sim_y
