################################################################################
### 1: Setup
################################################################################

# Import necessary packages
import copy as cp
import joblib as jbl
import numpy as np
import pandas as pd
import scipy.stats
import sklearn.preprocessing as skp
from hdmpy import cvec
from lmpy.ols import ols
from multiprocessing import cpu_count

################################################################################
### 2: Auxiliary functions
################################################################################

# Currently, there are none

################################################################################
### 3: Bootstrap algorithms
################################################################################

################################################################################
### 3.1: Single iterations for each algorithm
################################################################################


# Define one iteration of the pairs bootstrap algorithm
def _b_iter_pairs(model, bootstrap_stat='coefficients', seed=0, fix_seed=True,
                  **kwargs_fit):
    """ Run one iteration of the pairs bootstrap

    Inputs
    model: Model object, must have a fit() method and return a dictionary of
           results model.est. Underlying model used for the boostrap estimation.
    bootstrap_stat: Statistic to bootstrap, must be a key in model.est
    seed: Scalar, seed to use if fix_seed is True
    fix_seed: Boolean, if True, fixes the seed at the provided value

    Output
    res: k by 1 vector, estimated statistics for the bootstrap sample
    """
    # Make sure the underlying model object remains unchanged (that can be a
    # problem when not running this in parallel, for example)
    model = cp.copy(model)

    # Get data from the model
    X = model.X
    y = model.y

    # Get number of iterations y and variables k
    n, k = X.shape

    # Draw an index for sample of bootstrap observations with replacement
    # (randint draws numbers between low and high-1, which combined with
    # Python's zero indexing means this can draw every observation in the
    # sample)
    #
    # Check whether to fix the seed
    if fix_seed:
        # If so, draw with fixed seed
        idx = scipy.stats.randint(low=0, high=n).rvs(size=n, random_state=seed)
    else:
        # Otherwise, draw wfully randomly
        idx = scipy.stats.randint(low=0, high=n).rvs(size=n)

    # Draw samples of observations according to idx
    Xstar = X[idx,:]
    ystar = y[idx,:]

    # Fit the model
    model.fit(y=ystar, X=Xstar, add_intercept=False, names_X=model.names_X,
              name_y=model.name_y, **kwargs_fit)

    # Get the estimated coefficients
    res = cvec(model.est[bootstrap_stat])

    # Return the result
    return res


# Define one iteration of a wild bootstrap
def _b_iter_wild(model_res, model, bootstrap_stat='coefficients', eta=1, seed=0,
                 fix_seed=True, **kwargs_fit):
    """ Run one iteration of the wild bootstrap

    Inputs
    model_res: Model object, must have a fit() and a simulate() method and
               return a dictionary of results model_res.est. Underlying
               restricted model used for the bootstrap estimation.
    model: Model object, must have a fit() method and return a dictionary of
           results model.est. Underlying unrestricted model used for the
           bootstrap estimation.
    bootstrap_stat: Statistic to bootstrap, must be a key in model.est
    eta: Scalar, absolute value of the two possible values of the two point
         distribution used to generate bootstrapped residuals
    seed: Scalar, seed to use if fix_seed is True
    fix_seed: Boolean, if True, fixes the seed at the provided value

    Output
    res: k by 1 vector, estimated statistics for the bootstrap sample
    """
    # Make sure the underlying model objects remain unchanged (that can be a
    # problem when not running this in parallel, for example)
    model_res = cp.copy(model_res)
    model = cp.copy(model)

    # Get residuals from the restricted model
    U_hat_res = model_res.est['residuals']

    # Get X from the unrestricted model
    X = model.X

    # Get the number of observations n
    n = X.shape[0]

    # Get modified residuals, starting with disturbances drawn from a two point
    # distribution
    #
    # Check whether to fix the seed
    if fix_seed:
        # If so, draw with a fixed seed
        E = scipy.stats.bernoulli(p=.5).rvs(size=n, random_state=seed)
    else:
        # Otherwise, draw fully randomly
        E = scipy.stats.bernoulli(p=.5).rvs(size=n)

    # Replace any zeros as -1
    E[E==0] = -1

    # Multiply by eta and convert to a proper column vector
    E = cvec(E * eta)

    # Get residuals times disturbance term
    U_hatstar = U_hat_res * E

    # Simulate outcomes
    ystar = model_res.simulate(residuals=U_hatstar)

    # Fit the unrestricted model to the bootstrap sample
    model.fit(y=ystar, X=X, add_intercept=False, names_X=model.names_X,
              name_y=model.name_y, **kwargs_fit)

    # Get the estimated statistics
    res = cvec(model.est[bootstrap_stat])

    # Return the result
    return res

# Define one iteration of the Cameron, Gelbach, and Miller (2008) cluster robust
# wild bootstrap
def _b_iter_cgm(model_res, model, bootstrap_stat='coefficients', eta=1, seed=0,
                fix_seed=True, labencode=skp.LabelEncoder(), **kwargs_fit):
    """ Run one iteration of the Cameron, Gelbach, and Miller (2008) bootstrap

    Inputs
    model_res: Model object, must have a fit() and a simulate() method and
               return a dictionary of results model_res.est. Underlying
               restricted model used for the bootstrap estimation.
    model: Model object, must have a fit() method and return a dictionary of
           results model.est. Underlying unrestricted model used for the
           bootstrap estimation.
    bootstrap_stat: Statistic to bootstrap, must be a key in model.est
    eta: Scalar, absolute value of the two possible values of the two point
         distribution used to generate bootstrapped residuals
    seed: Scalar, seed to use if fix_seed is True
    fix_seed: Boolean, if True, fixes the seed at the provided value

    Output
    res: k by 1 vector, estimated statistics for the bootstrap sample
    """
    # Make sure the underlying model objects remain unchanged (that can be a
    # problem when not running this in parallel, for example)
    model_res = cp.copy(model_res)
    model = cp.copy(model)

    # Get residuals from the restricted model
    U_hat_res = model_res.est['residuals']

    # Get X from the unrestricted model
    X = pd.DataFrame(model.X, columns=model.names_X)

    # Get cluster variable (doesn't matter from which model)
    clustvar = cvec(labencode.fit_transform(model.clustvar.flatten()))

    # Get the number of clusters in the data
    J = len(np.unique(clustvar))

    # Get modified residuals, starting with disturbances drawn from a two point
    # distribution
    #
    # Check whether to fix the seed
    if fix_seed:
        # If so, draw with a fixed seed
        E = scipy.stats.bernoulli(p=.5).rvs(size=J, random_state=seed)
    else:
        # Otherwise, draw fully randomly
        E = scipy.stats.bernoulli(p=.5).rvs(size=J)

    # Replace any zeros as -1
    E[E==0] = -1

    # Multiply by eta and convert to a proper column vector
    E = cvec(E * eta)

    # Use cluster indices to assign each unit it's cluster's disturbance
    E = E[clustvar[:,0],:]

    # Get residuals times disturbance term
    U_hatstar = U_hat_res * E

    # Simulate outcomes using the restricted model
    ystar = model_res.simulate(residuals=U_hatstar)

    # Fit the unrestricted model to the bootstrap sample
    model.fit(y=ystar, X=X, add_intercept=False,  names_X=model.names_X,
              name_y=model.name_y, **kwargs_fit)

    # Get the estimated statistics
    res = cvec(model.est[bootstrap_stat])

    # Return the result
    return res

################################################################################
### 4.2: Class to implement bootstrapping algorithms
################################################################################

class boot():
    """ Runs bootstrap algorithms """


    # Define initialization function
    def __init__(self, model=ols, y=None, X=None, algorithm='pairs',
                 impose_null_idx=None, eta=1, bootstrap_stat='t', level=None,
                 one_sided=None, B=4999, par=True, corecap=np.inf,
                 fix_seed=True, batch_size='auto', verbose=True,
                 fprec=np.float32, nround=4, **kwargs_fit):
        """ Initialize boot class

        Inputs
        model: Model, must have certain methods and contain specific named
               objects. All models in lmpy follow the conventions needed to be
               used as an input for boot(). Underlying model used to estimate
               statistics of interest on bootstrap samples.
        y: n by 1 vector-like or None, outcome variable. If model has not been
           fit, boot() can fit the model itself, given y and X were provided.
        X: n by k matrix-like or None, RHS variables
        algorithm: String, bootstrap algorithm to use. Possible choices are:

                   'pairs': Pairs bootstrap
                   'wild': Wild bootstrap
                   'cgm': Cameron, Gelbach, and Miller (2008) cluster robust
                          wild bootstrap
        impose_null_idx: Boolean k by 1 vector-like or None, if not None, has to
                         specify the position of variables in X for which a null
                         has to be imposed. Currently, the only possible null is
                         that the coefficients are zero. The pairs bootstrap
                         does not support imposing a null, and will ignore this
                         if provided.
        eta: Scalar, absolute value of the two values for the two point
             distributions used in wild-type bootstraps
        bootstrap_stat: String, has to be a key in the dictionary of model
                        estimates model.est. Statistics to bootstrap.
        level: Scalar or None, if None, uses model.level as default. Level for
               confidence intervals to construct.
        one_sided: Boolean or None, whether to construct one-sided confidence
                   intervals; if None, uses reasonable defaults, see
                   self.get_ci()
        B: Positive integer, number of bootstrap iterations to use
        par: Boolean, if True, runs bootstrap iterations in parallel
        corecap: Integer of np.inf, maximum number of cores to use. Setting this
                 to np.inf uses all available cores.
        fix_seed: Boolean, if True, seeds are fixed throughout the bootstrapping
                  routine, to ensure replicability
        batch_size: Scalar or 'max' or 'auto', batch size in joblib.Parallel.
                    See the joblib documentation for details. If set to 'auto',
                    uses the joblib default heuristic. If set to 'max', uses
                    batch_size = B / (cores used).
        verbose: Boolean, if True, boot() prints notes and warnings
        fprec: Float data type, all data will be cast to this type. The default
               is np.float32, which can speed up NumPy's linear algebra in
               certain settings.
        nround: Integer, results will be rounded to this number of decimals
        """
        # Instantiate parameters
        #
        # Instantiate model to use for bootstrapping
        self.model = cp.deepcopy(model)

        # Instantiate verbosity parameter
        self.verbose = verbose

        # Check whether the model is an OLS model
        if isinstance(self.model, ols):
            # Check whether clusters were provided, but the covariance estimator
            # was not specified, and clusters were not provided in kwargs_fit
            if (
                    self.model.clustvar is not None
                    and self.model.cov_est is None
                    and 'clusters' not in kwargs_fit.keys()
            ):
                # If so, specify it. (The reason I do this is because the model
                # will be fit many times during the bootstrap loop. If no
                # covariance estimator is specified, OLS sets self.clustvar to
                # None if clusters are not provided in ols.fit(). This behavior
                # helps handling certain use cases, but for bootstrapping, the
                # covariance estimator has to be specified to be prevent cluster
                # information being lost. An alternative is to specify clusters
                # in **kwargs_fit.)
                self.model.cov_est = 'cluster'

                # Check whether to be talkative
                if self.verbose:
                    # If so, display a note
                    print('\nNote in boot(): The provided model contains a',
                          'cluster ID variable (self.clustvar), but',
                          'covariance_estimator is None, and **kwargs_fit',
                          'does not provide cluster IDs. Everything should be',
                          'fine, but it is recommended to either specify the',
                          'covariance estimator, or to provide cluster IDs',
                          'via **kwargs_fit.')

        # Check whether the model came pre fitted
        if self.model.est is None:
            # Check whether data were provided
            if (y is None) or (X is None):
                # If not, raise an error
                raise ValueError('Error in boot(): Model has not been pre-fit, '
                                 + 'and data to fit on were not provided; '
                                 + 'either provide a pre-fit model, or '
                                 + 'complete data')
            else:
                # If they were provided, fit the model
                self.model.fit(X, y, **kwargs_fit)

        # Instantiate float precision data type
        self.fprec = fprec

        # Instantiate parameters for running bootstrapping algorithms
        self.algorithm = algorithm
        self.stat = bootstrap_stat
        self.B = B
        self.impose_null_idx = impose_null_idx
        self.eta = eta

        # Instantiate parameters related to parallel processing
        #
        # Basic parameters
        self.par = par
        self.corecap = corecap
        self.fix_seed = fix_seed

        # Check whether to use parallel processing
        if self.par:
            # If so, get the number of cores to use
            self.n_cores = np.amin([cpu_count(), self.corecap]).astype(int)
        else:
            # Otherwise, use just one core
            self.n_cores = 1

        # Batch size for joblib.Parallel()
        if batch_size == 'max':
            # If it is set to 'max', each workers has to do as many tasks as
            # there are bootstrap iterations divided by the number of cores
            self.batch_size = np.int(np.ceil(self.B/self.n_cores))
        else:
            # Otherwise, use the provided value
            self.batch_size = batch_size

        # Instantiate parameters for creating confidence intervals
        if level is not None:
            self.level = level
        else:
            self.level = self.model.level
        self.one_sided = one_sided

        # Instantiate parameters for self.summarize() results
        self.nround = nround

        # Instantiate data matrices
        self.X = None
        self.y = None

        # Instantiate variables created by other functions
        #
        # Variables created by self.bootstrap_distribution()
        self.bsamps = None

        # Variables created by self.get_ci()
        self.ci = None
        self.names_ci = None

        # Variables created by self.summarize()
        self.regtable = None

        # Variables created by self.boot_cov()
        self.V_hat_boot = None

        # Run functions to get bootstrapped confidence interval
        # Get bootstrap distribution
        self.bootstrap_distribution(**kwargs_fit)

        # Get confidence intervals
        self.get_ci()

        # Combine results into a dictionary
        self.est = {
            'original estimates': self.model.est[self.stat],
            'ci': pd.DataFrame(self.ci, index=self.model.est[self.stat].index,
                               columns=self.names_ci),
            'statistic': self.stat,
            'B': self.B,
            'algorithm': self.algorithm
        }

        # Check whether a null was imposed
        if self.impose_null_idx is not None:
            # If so, add the null index to the results
            self.est['null imposed'] = (
                pd.DataFrame(self.impose_null_idx, index=self.model.names_X,
                             columns=['Null imposed'])
            )


    # Define function to run different bootstrapping algorithms
    def bootstrap_distribution(self, **kwargs_fit):
        """ Get bootstrap distribution  """
        # Get data from the model
        self.X = self.model.X
        self.y = self.model.y

        # Check whether a vector indicating parameters which need to have a
        # null enforced was provided
        if self.impose_null_idx is not None:
            # Convert null indices to a proper column vector
            self.impose_null_idx = cvec(self.impose_null_idx)

            # Check whether the null index is one element too short, and the
            # model used added an intercept to the data
            if (
                    (self.impose_null_idx.shape[0] == self.X.shape[1] - 1)
                    and self.model.add_icept
            ):
                # If so, add a False at the beginning, assuming that this
                # happened because the intercept was left out of the model
                self.impose_null_idx = (
                    np.concatenate([cvec(False), self.impose_null_idx], axis=0)
                )

            # Get part of X corresponding to unrestricted coefficients in the
            # restricted model
            self.X_res = self.X[:,~self.impose_null_idx[:,0]]

            # Set up restricted model
            self.model_res = cp.deepcopy(self.model)

            # Fit the restricted model
            self.model_res.fit(self.X_res, self.y, add_intercept=False,
                               coef_only=True, **kwargs_fit)
        else:
            # Otherwise, set self.model_res to self.model, so this can easily be
            # passed to algorithms which could have a null imposed
            self.model_res = cp.deepcopy(self.model)

        # Check which algorithm to use
        #
        # Pairs bootstrap
        if self.algorithm.lower() == 'pairs':
            # Check whether a null was impose
            if self.impose_null_idx is not None:
                # Check whether to be talkative
                if self.verbose:
                    # If so, print a note
                    print('\nNote in boot(): impose_null_idx was provided, but '
                          + 'the pairs bootstrap does not support a null being '
                          + 'imposed. The null will be ignored.')

            # Get bootstrapped distribution for statistic
            self.bsamps = (
                jbl.Parallel(n_jobs=self.n_cores, batch_size=self.batch_size)(
                    jbl.delayed(_b_iter_pairs)(
                        model=self.model, bootstrap_stat=self.stat, seed=b,
                        fix_seed=self.fix_seed, **kwargs_fit)
                    for b in np.arange(self.B)
                )
            )

        # Wild bootstrap
        elif self.algorithm.lower() == 'wild':
            # Get bootstrapped distribution for statistic
            self.bsamps = (
                jbl.Parallel(n_jobs=self.n_cores, batch_size=self.batch_size)(
                    jbl.delayed(_b_iter_wild)(
                        model_res=self.model_res, model=self.model,
                        bootstrap_stat=self.stat, eta=self.eta, seed=b,
                        fix_seed=self.fix_seed, **kwargs_fit)
                    for b in np.arange(self.B)
                )
            )

        # Cameron, Gelbach, and Miller (2008) cluster robust bootstrap
        elif self.algorithm.lower() == 'cgm':
            # Get bootstrapped distribution for statistic
            self.bsamps = (
                jbl.Parallel(n_jobs=self.n_cores, batch_size=self.batch_size)(
                    jbl.delayed(_b_iter_cgm)(
                        model_res=self.model_res, model=self.model,
                        bootstrap_stat=self.stat, eta=self.eta,
                        seed=b, fix_seed=self.fix_seed, **kwargs_fit)
                    for b in np.arange(self.B)
                )
            )

        # Anything else
        else:
            # Print an error message
            raise ValueError('Error in boot.bootstrap_distribution(): The '
                             + 'specified bootstrap algorithm could not be '
                             + 'recognized; please specify a valid algorithm.')

        # Concatenate bootstrap samples into a single k by B DataFrame
        self.bsamps = np.concatenate(self.bsamps, axis=1)


    # Define a function to calculate bootstrapped confidence intervals
    def get_ci(self, one_sided=None):
        """ Get bootstrapped confidence interval """
        # Set one sided parameter
        #
        # If one was provided ...
        if one_sided is not None:
            # ... continue as is
            pass

        # Otherwise, if one was set in __init__() ...
        elif self.one_sided is not None:
            # ... use that
            one_sided = self.one_sided

        # Otherwise, if this is bootstrapping a Wald statistic ...
        elif self.stat in ['wald']:
            # ... use an upper one sided test
            one_sided = 'upper'

        # Otherwise ...
        else:
            # ... use a two sided test
            one_sided = False

        # Get quantiles to compute
        if one_sided == 'upper':
            quants = [0, 1 - self.level]
        elif one_sided == 'lower':
            quants = [self.level, 1]
        else:
            quants = [self.level/2, 1 - self.level/2]

        # Get the confidence intervals
        self.ci = np.quantile(self.bsamps, quants, axis=1).T.astype(self.fprec)

        # Replace upper or lower bounds if applicable
        if (one_sided == 'upper') and self.stat in ['wald']:
            self.ci[:,0] = 0
        elif one_sided == 'upper':
            self.ci[:,0] = -np.inf
        elif one_sided == 'lower':
            self.ci[:,1] = np.inf

        # Make a two element list of names for the upper and lower bound
        self.names_ci = ['{}%'.format(q*100) for q in quants]


    # Define a function to summarize the results
    def summarize(self):
        """ Produce a pandas DataFrame summarizing the results """
        # Make a regression table DataFrame
        self.regtable = (
            pd.concat(
                [self.est['original estimates'],
                 self.est['ci']
                ], axis=1
            )
        )

        # Add null imposed information if applicable
        if (
                (self.impose_null_idx is not None)
                and (self.impose_null_idx.shape[0] == self.regtable.shape[0])
        ):
            self.regtable = (
                pd.concat([self.regtable, self.est['null imposed']], axis=1)
            )

        # Round the results
        self.regtable = self.regtable.round(self.nround)

        # Add a name to the DataFrame
        self.regtable.name = 'Bootstrapped ' + self.stat

        # Return the result, to make it easily printable
        return self.regtable


    # Define a function to calculate the covariance matrix across bootstrap
    # samples
    def boot_cov(self):
        """ Calculate covariance matrix of bootstrapped statistics """
        # Calculate covariance
        self.V_hat_boot = np.cov(self.bsamps, rowvar=True)


    # Define a function to run Wald test
    def wald(self, jointsig=None, R=None, b=None):
        """ Run a Wald test

        First run boot_cov(), then use this to calculate a Wald statistic. Note
        that this is usually not a good idea, since using the bootstrap to
        estimate statistics, rather than the distribution of a pivot, does not
        provide any asymptotic refinements over analytical methods. The only
        real use case is if the covariance matrix of the underlying model is too
        complicated to be estimated analytically.

        Inputs
        jointsig, R, b: See self.model's documentation

        Outputs
        self.model.waldtable, see self.model's documentation
        """
        # Check whether self.boot_cov() has been run
        if self.V_hat_boot is not None:
            # If so, get a Wald statistic using the bootstrapped covariance
            # matrix
            self.model.wald(jointsig=jointsig, R=R, b=b, V_hat=self.V_hat_boot)
        else:
            # Otherwise, potentially display a warning
            if self.verbose:
                print('\nNote in boot.wald(): Please use boot.get_cov() before',
                      'running boot.wald(). Returning Wald statistic based on',
                      'model covariance instead.')

            # Just get the Wald statistic from the underlying model
            self.model.wald(jointsig=jointsig, R=R, b=b)

        # Save the results
        self.W_boot = self.model.W
        self.pW_boot = self.model.pW

        # Return the summary table
        return self.model.waldtable
