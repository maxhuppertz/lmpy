################################################################################
### Define OLS class
################################################################################

################################################################################
### 1: Setup
################################################################################

# Import necessary packages
import numpy as np
import pandas as pd
import scipy.linalg as scl
import scipy.stats as scs
from hdmpy import cvec
from scipy.stats import norm

################################################################################
### 2: Auxiliary functions
################################################################################

# Currently, there are none

################################################################################
### 3: Define OLS class
################################################################################


# Define OLS model
class ols():
    """ Runs OLS regressions """


    # Define initialization function
    def __init__(self, name_gen_X='X', name_gen_y='y', add_intercept=True,
                 name_gen_icept='(Intercept)', coef_only=False,
                 no_joint=False, covariance_estimator=None, level=.05,
                 verbose=True, fprec=np.float64, nround=4):
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
        no_joint: Boolean, if True, the model does not (by default) calculate a
                  Wald test of joint significane of all coefficients
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
               default value is np.float64, to ensure high precision. Using
               np.float32 instead can speed up NumPy's linear algebra in some
               settings.
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
        self.no_joint = no_joint

        # Parameters for self.ols_ci()
        self.level = level

        # Parameters for self.summarize()
        self.nround = nround

        # Instantiate variables created by various methods
        #
        # Variables from ols_fit()
        self.names_X = None
        self.name_y = None
        self.coef = None
        self.est = None

        # Variables created by self.ols_cov()
        self.cov_est = covariance_estimator
        self.V_hat = None
        self.se = None

        # Variables created by self.ols_t()
        self.t = None

        # Variables created by self.ols_ci()
        self.ci = None

        # Variables created by self.ols_p()
        self.p = None

        # Variables created by self.summarize()
        self.regtable = None

        # Variables created by self.score()
        self.R2 = None

        # Variables created by self.wald()
        self.W = None
        self.pW = None
        self.waldtable = None


    # Define a function to fit the model
    def fit(self, X, y, clusters=None, names_X=None, name_y=None,
            name_gen_X=None, name_gen_y=None, add_intercept=None,
            coef_only=None, no_joint=None, copy=True, **kwargs_wald):
        """ Fit OLS model

        Inputs
        y: n by 1 vector-like, outcome variable
        X: n by k matrix-like, RHS variables
        clusters: n by 1 vector-like or None, cluster IDs
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
        no_joint: Boolean or None, see __init()__; if None, use the value
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

        # Check whether the joint significance parameter was changed
        if no_joint is not None:
            # If so, adjust it
            self.no_joint = no_joint

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
            self.names_X = [X.name]

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

        if copy:
            X = X.copy()

        # Instantiate data matrices
        #
        # Start by instantiating the X data as is
        if np.ndim(X) == 1:
            # For one dimensional X, make sure this is a proper vector
            X = cvec(X)
        else:
            # Otherwise, make sure it's a proper NumPy array
            X = np.array(X)

        # Check whether to add an intercept
        if self.add_icept:
            # If so, set up an intercept
            cons = np.ones(shape=(X.shape[0], 1))

            # Add it to the data matrix
            X = np.concatenate([cons, X], axis=1)

            # Add the intercept to the variables names
            self.names_X = [self.name_gen_icept] + list(self.names_X)

        # For speed considerations, make sure these are self.fprec types
        X = X.astype(self.fprec)

        # Get number of observations n and variables p
        self.n, self.k = X.shape

        # Instantiate y data elements
        fit_y = cvec(y).astype(self.fprec)

        # Check whether clusters were not provided, but clustered errors are
        # supposed to be used
        if clusters is None and self.cov_est == 'cluster':
            raise ValueError(
                "Error in ols.fit(): The covariance estimator was set to"
                + " 'cluster', but clusters were not provided; please provide"
                + " cluster IDs, or change the covariance estimator"
            )

        # Check whether a cluster variable was provided
        elif clusters is not None:
            # If so, adjust the cluster variable
            clustvar = cvec(clusters)

        # Calculate coefficient vector
        #self.coef = self.XXinv @ (X @ fit_y)
        self.coef = cvec(scl.lstsq(X, fit_y)[0]).astype(self.fprec)

        # Check whether to calculate anything besides the coefficients
        if not self.coef_only:
            # Calculate some inputs for covariance estimators
            #
            # Get residuals
            U_hat = fit_y - X @ self.coef

            # Get other elements of the OLS model
            #
            # Get the covariance
            sedf, Vdf = self.ols_cov(X=X, residuals=U_hat, clusters=clustvar)

            # Get t-statistics
            tdf = self.ols_t()

            # Get confidence intervals
            cidf = self.ols_ci()

            # Get p-values
            pdf = self.ols_p()

            # Check whether to do a Wald test
            if not self.no_joint:
                # Do a Wald test
                walddf = self.wald(**kwargs_wald)
                waldstat = walddf.loc['Wald statistic', :]
                waldp = walddf.loc['p-value', :]
            else:
                waldstat = None
                waldp = None
        else:
            # Otherwise, set all other results to NAN. (This is important for
            # coef_only=True to provide a speed-up. If this does not happen,
            # some of the pandas DataFrames containing results are created
            # automatically, and automatically inferring their shape and size
            # takes a while. Without pre-setting these here, coef_only=True
            # actually slows down the program.)
            sedf = None
            tdf = None
            cidf = None
            pdf = None
            waldstat = None
            waldp = None
            Vdf = None

        # Combine results into a dictionary
        self.est = {
            'coefficients': pd.DataFrame(self.coef, index=self.names_X,
                                         columns=['Estimated coefficient']),
            'se': sedf,
            'covariance estimator': self.cov_est,
            't': tdf,
            'ci': cidf,
            'level': self.level,
            'p': pdf,
            'wald': waldstat,
            'wald p': waldp,
            'covariance matrix': Vdf#,
            #'residuals': pd.DataFrame(U_hat, columns=['Residuals']),
            #'clusters': pd.DataFrame(clustvar, columns=['Cluster ID'])
        }


    # Define a function to calculate the covariance matrix plus standard errors
    def ols_cov(self, X, residuals, clusters=None, covariance_estimator=None):
        """ Calculate covariance matrix and standard errors

        Input
        covariance_estimator: String or None, see __init()__; if None, uses the
                              value provided in __init()__
        """

        # Calculate (X'X)^(-1)
        XXinv = scl.pinv(X.T @ X)

        # Get residuals and clusters
        U_hat = residuals
        clustvar = clusters

        # Check whether covariance_estimator was changed from the default None
        if covariance_estimator is not None:
            # If so, set cov_est to the specified covariance estimator
            cov_est = covariance_estimator

        # Otherwise, check whether the original value provided to __init__() was
        # left at the default None, and no cluster IDs were provided
        elif (self.cov_est is None) and (clustvar is None):
            # If so, use HC1 as the default covariance estimator
            cov_est = 'hc1'

        # Otherwise, check whether the original value provided to __init__() was
        # left at the default None, and cluster IDs were provided
        elif (self.cov_est is None) and (clustvar is not None):
            # If so, use clustered standard errors
            cov_est = 'cluster'

        else:
            # Otherwise, use the specified covariance estimator
            cov_est = self.cov_est

        # Check whether clusters were provided, but a non-clustered covariance
        # estimator is being used, and the class is set to be talkative
        if (
                (clustvar is not None)
                and (cov_est != 'cluster')
                and self.verbose
        ):
            print('\nNote in ols(): Cluster IDs were provided, but a',
                  'non-clustered covariance estimator is being used')

        # Check which covariance estimator to use
        #
        # Homoskedastic
        if cov_est.lower() == 'homoskedastic':
            # For the homoskedastic estimator, just calculate the standard
            # variance
            self.V_hat = (
                (1 / (self.n - self.k))
                * XXinv * (U_hat.T @ U_hat)
            )

        # HC1
        elif cov_est.lower() == 'hc1':
            # Calculate component of middle part of EHW sandwich,
            # S_i = X_i u_i, which makes it very easy to calculate
            # sum_i X_i X_i' u_i^2 = S'S
            S = (U_hat @ np.ones(shape=(1,self.k))) * X

            # Calculate EHW variance/covariance matrix
            self.V_hat = (
                (self.n / (self.n - self.k))
                * XXinv @ (S.T @ S) @ XXinv
            )

        # Clustered errors
        elif cov_est.lower() == 'cluster':
            # Calculate number of clusters
            J = len(np.unique(clustvar[:,0]))

            # Same thing as S above, but needs to be a DataFrame, because pandas
            # has the groupby method, which is needed in the next step
            S = pd.DataFrame((U_hat @ np.ones(shape=(1,self.k))) * X)

            # Sum all covariates within clusters
            S = S.groupby(clustvar[:,0], axis=0).sum()

            # Convert back to a NumPy array
            S = np.array(S).astype(self.fprec)

            # Calculate cluster-robust variance estimator
            self.V_hat = (
                (self.n / (self.n - self.k) ) * (J / (J - 1))
                * XXinv @ (S.T @ S) @ XXinv
            )

        # Some other unknown method
        else:
            # Print an error message
            raise ValueError('Error in ols.fit(): The specified covariance '
                             + 'estimator ({})'.format(cov_est)
                             + 'could not be recognized; please '
                             + 'specify a valid estimator')

        # Replace NaNs as zeros (happens if division by zero occurs)
        #self.V_hat[np.isnan(self.V_hat)] = 0
        #self.V_hat[np.isinf(self.V_hat)] = 0

        # Calculate the standard errors for all coefficients
        self.se = cvec(np.sqrt(np.diag(self.V_hat))).astype(self.fprec)

        sedf = pd.DataFrame(
            self.se, index=self.names_X, columns=['Standard error']
        )
        Vdf = pd.DataFrame(self.V_hat, index=self.names_X, columns=self.names_X)

        return sedf, Vdf


    # Define a function to calculate t-statistics
    def ols_t(self):
        """ Calculate t-statistics """
        # Calculate t-statistics
        self.t = self.coef / self.se

        # Return the results
        return pd.DataFrame(self.t, index=self.names_X, columns=['t-statistic'])


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
        cval = cvec(scs.t(df=self.n-self.k).ppf(quants)).astype(self.fprec).T

        # Calculate confidence intervals
        self.ci = (
            self.coef @ np.ones(shape=(1,2))
            + (self.se @ np.ones(shape=(1,2)))
            * (np.ones(shape=(self.k,1)) @ cval)
        )

        # Make a two element list of names for the upper and lower bound
        names_ci = ['{}%'.format(q*100) for q in quants]

        return pd.DataFrame(self.ci, index=self.names_X, columns=names_ci)


    # Define a function to calculate p-values
    def ols_p(self):
        """ Calculate p-values """

        # Calculate p-values
        self.p = 2 * (1 - scs.t(df=self.n-self.k).cdf(np.abs(self.t)))

        return pd.DataFrame(self.p, index=self.names_X, columns=['p-value'])


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


    # Define a function which calculates fitted values
    def predict(self, X, add_intercept=None):
        """ Calculate fitted values """

        # Check whether add_intercept was specified
        if add_intercept is not None:
            # If so, record the value
            add_icept = add_intercept
        else:
            # Otherwise, use the model's value
            add_icept = self.add_icept

        # Make sure X is two dimensional
        if np.ndim(X) == 1:
            X = cvec(X)
        else:
            X = np.array(X)

        # Check whether an intercept needs to be added
        if add_icept:
            # If so, set up an intercept
            cons = np.ones(shape=(X.shape[0], 1))

            # Add it to the data matrix
            X = np.concatenate([cons, X], axis=1)

        # Cast X to the set precision
        X = X.astype(self.fprec)

        # Get fitted values
        y_hat = X @ self.coef

        # Return them
        return y_hat


    # Define a function to calculate the R-squared
    def score(self, X, y):
        """ Calculate R-squared """

        # Make sure X is two dimensional
        if np.ndim(X) == 1:
            X = cvec(X)
        else:
            X = np.array(X)

        # Check whether an intercept needs to be added
        if self.add_icept:
            # If so, set up an intercept
            cons = np.ones(shape=(X.shape[0], 1))

            # Add it to the data matrix
            X = np.concatenate([cons, X], axis=1)

        # Make sure the data have the right precision
        X = X.astype(self.fprec)

        score_y = cvec(y).astype(self.fprec)

        # Get fitted values at X
        y_hat = self.predict(X, add_intercept=False)

        # Calculate residual sum of squares
        SSR = ((score_y - y_hat) ** 2).sum()

        # Calculate total sum of squares
        SST = ((score_y - score_y.mean()) ** 2).sum()

        # Calculate R-squared
        self.R2 = 1 - (SSR / SST)

        # Return it
        return self.R2


    # Define a function to calculate a Wald test (e.g. of joint significance)
    def wald(self, jointsig=None, R=None, b=None, V_hat=None):
        """ Calculate a Wald test

        The Wald test can be used to test linear restrictions of the form

        R @ beta = b

        where R is a matrix of restrictions, beta are the true coefficients of
        the underlying model, and b is a hypothesis about the value of their
        linear combination

        Inputs
        jointsig: list-like, either Boolean, 0-1 integer or strings, or None;
                  denotes variables to be used for a joint significance test.
                  If this is s a string-type, it has to contain names of
                  variables which appear in self.names_X. Those variables will
                  be tested. If it is a Boolean or 0-1 integer, it has to be of
                  length k, and all True or 1 elements denote variables which
                  will be tested. If this is not None, the R matrix will pick
                  out the indicated elements of beta.
        R: q by k matrix-like or None; first part of linear restrictions being
           imposed on the model. If R is not None, this matrix will be used. (If
           R is not None and jointsig is not None, jointsig will be ignored.)
        b: q by 1 vector-like or None; second part of linear restrictions being
           tested. If this is None, b will be a vector of zeroes.
        V_hat: k by k matrix-like or None; covariance matrix to use for
               calculating the Wald Statistic. If this is None, the model's
               estimated covariance matrix self.V_hat will be used.

        Outputs
        self.waldtable: 2 by 1 DataFrame, containing Wald statistic and
                        associated p-value
        """

        # Instantiate LHS restrictions matrix
        #
        # Check whether a joint significance test was specified
        if jointsig is not None:
            # If so, make sure the joint significance variables are a one
            # dimensional array (so I can iterate over them if necessary)
            jointsig = np.array(jointsig).flatten()

            # Check whether jointsign is a list of strings
            if isinstance(jointsig[0], str):
                # If so, get the associated variables
                Rbase = (
                    np.array([1 if v in jointsig else 0 for v in self.names_X])
                )
            else:
                # Otherwise, check whether an intercept needs to be added
                if self.add_icept or (self.k == len(jointsig) + 1):
                    # If so, add a zero at the beginning of jointsig
                    jointsig = np.array([0] + [j for j in jointsig])

                # Make sure jointsig is recorded as an integer array
                Rbase = jointsig.astype(int)

        # Alternatively, check whether R is None
        elif R is None:
            # If so, set up an array indicating that all variables need to be
            # restricted (that is, an array of all ones)
            Rbase = np.ones(shape=(self.k))

        # Check whether the restriction matrix R was left at its default None
        if R is None:
            # If so, set up a matrix of restrictions as a diagonal matrix
            R = np.diag(Rbase)

            # Keep only those rows corresponding to variables which are being
            # restricted
            R = R[Rbase != 0, :]

        # Alternatively, check whether an intercept needs to be added
        elif self.add_icept:
            # If so, add a column of zeros to the beginning of the restriction
            # matrix
            R = np.concatenate([np.zeros(shape=(R.shape[0], 1)), R], axis=1)

        # Either way, ensure restriction matrix has the correct precision
        R = R.astype(self.fprec)

        # Instantiate RHS null vector
        #
        # Check whether null values were provided
        if b is None:
            # If not, use the default null of everything being zero
            b = np.zeros(shape=(R.shape[0], 1)).astype(self.fprec)
        else:
            # Otherwise, just ensure b is a proper vector
            b = cvec(b)

        # Either way, make sure b has the correct precision
        b = b.astype(self.fprec)

        # Instantiate covariance matrix
        #
        # Check whether a covariance matrix was provided
        if V_hat is not None:
            # If so, use that
            V = np.array(V_hat).astype(self.fprec)
        else:
            # Otherwise, use the estimated covariance matrix
            V = self.V_hat

        # Calculate Wald statistic
        #
        # Calculate outer parts of the Wald 'sandwich', R beta - b
        Rbdiff = R @ self.coef - b

        # Calculate inner part of the Wald 'sandwich', (RVR')^(-1)
        RVRinv = scl.pinv(R @ V @ R.T)

        # Calculate the Wald statistic
        self.W = Rbdiff.T @ RVRinv @ Rbdiff

        # Calculate p-value
        #
        # Get rank of restriction matrix
        r = np.linalg.matrix_rank(R)

        # Calculate p-value using the appropriate chi-squared distribution
        self.pW = 1 - scs.chi2(df=r).cdf(self.W)

        # Make a table containing the result
        self.waldtable = (
            pd.DataFrame(np.concatenate([self.W, self.pW], axis=0),
                         index=['Wald statistic', 'p-value'],
                         columns=['Estimate'])
        )

        # Return the Wald statistic and associated p-value
        return self.waldtable


    # Define a function which generates a new draw of data (for bootstrapping)
    def simulate(self, X, residuals, coef=None):
        """ Simulate data for bootstrap draws """

        # Check whether coef was left at the default None
        if coef is None:
            # If so, set the simulation coefficients to the fitted coefficients
            sim_coef = self.coef
        else:
            # Otherwise, set the simulation coefficients to a proper column
            # vector based on coef
            sim_coef = cvec(coef).astype(self.fprec)

        if np.ndim(X) == 1:
            X = cvec(X)
        else:
            X = np.array(X)

        # Check whether an intercept needs to be added
        if self.add_icept:
            # If so, set up an intercept
            cons = np.ones(shape=(X.shape[0], 1))

            # Add it to the data matrix
            X = np.concatenate([cons, X], axis=1)

        # Make sure the data have the right precision
        X = X.astype(self.fprec)

        # Adjust the simulation residuals
        sim_residuals = cvec(residuals).astype(self.fprec)

        # Generate simulated data
        y = X @ sim_coef + sim_residuals

        # Return the result
        return y


    # Define a dummy method set_params(), so this can be used with that has such
    # functionality (e.g. scikit-learn). I might later expand this to be an
    # actual method, although it should not be necessary.
    def set_params(self, *args, **kwargs):
        """ Dummy method which does nothing """

        # Do nothing
        pass
