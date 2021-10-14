################################################################################
### 1: Setup
################################################################################

# Import necessary packages
import copy as cp
import joblib as jbl
import numpy as np
import pandas as pd
import scipy.stats as scs
import sklearn.preprocessing as skp
from lmpy.ols import cvec, ols
from multiprocessing import cpu_count

################################################################################
### 2: Auxiliary functions
################################################################################


# Define a function to calculate boostrap p value for a single statistic
def _get_p_i(bsamp, orig_estimate, one_sided=False, imp_null=False):
    """ Calculate bootstrap p-value for a single statistic

    Inputs
    bsamp: vector-like; bootstrapped statistics
    orig_estimate: float; statistic for the original sample
    one_sided: 'upper', 'lower', or False; if 'upper', the test is against the
               one sided alternative that the statistic is positive, if 'lower',
               the test is against the one sided alternative that it is
               negative, if False, the test is two sided
    imp_null: boolean; if True, a null was imposed while calculating the
              bootstrapped statistics. Currently, only the null that the
              underlying population value is zero is supported.

    Output
    p: float; bootstrap p-value

    Note
    Currently, the only supported null hypothesis is that

    orig_parameter == 0

    where orig_parameter is the parameter orig_estimate is an estimate of
    """

    # Check whether the original estimate was below zero
    if orig_estimate < 0:
        if one_sided == 'upper':
            # If so, never reject for an upper tailed alternative. (The
            # strongest permissible test under that alternative uses 0 as its
            # critical value, but given the original estimate is negative, even
            # that test fails to reject the null. So there is no strongest
            # permissible test which just rejects the null.)
            p = 1
        elif imp_null:
            # If the null was imposed, record the percentile ranking of the
            # original estimate in the bootstrap samples. (The area under the
            # bootstrap CDF below the original estimate.) A test which uses a
            # critical value just at the original estimate will just reject the
            # null. The 'weak' option makes this conservative - it counts values
            # weakly smaller than the original estimate.
            p = scs.percentileofscore(bsamp, orig_estimate, kind='weak')/100
        else:
            # If no null was imposed, record one minus the percentile ranking of
            # zero. (The are under the bootstrap CDF above zero.) A test which
            # uses a critical value just at zero will just reject the null. The
            # 'strict' option makes this conservative - it counts only values
            # strictly below zero.
            p = 1 - scs.percentileofscore(bsamp, 0, kind='strict')/100
    else:
        if one_sided == 'lower':
            # If the original estimate was positive, never reject for a lower
            # tailed alternative
            p = 1
        elif imp_null:
            # If the null was imposed, record one minus the percentile ranking
            # of the original estimate
            p = 1 - scs.percentileofscore(bsamp, orig_estimate, kind='weak')/100
        else:
            # If no null was imposed, record the percentile ranking of zero
            p = scs.percentileofscore(bsamp, 0, kind='strict')/100

    # If the test is two sided, double the p-value. (The two cases above ensure
    # that the test under consideration uses a critical value which allows it to
    # just reject the null. This just puts an equal amount of mass outside of
    # -orig_estimate, which will not affect rejection of the null.
    if not one_sided:
        p = np.amin([2 * p, 1])

    # Return the p-value
    return p

################################################################################
### 3: Bootstrap algorithms
################################################################################

################################################################################
### 3.1: Single iterations for each algorithm
################################################################################


# Define one iteration of the pairs bootstrap algorithm
def _b_iter_pairs(model, X, y, weights=None, bootstrap_stat='coefficients',
                  seed=0, fix_seed=True, copy_data=True, **kwargs_fit):
    """ Run one iteration of the pairs bootstrap

    Inputs
    model: Model object; must have a fit() method and return a dictionary of
           results model.est. Underlying model used for the boostrap estimation.
    X: n by k matrix-like; RHS variables
    y: n by 1 vector-like; outcome variable
    weights: n by 1 vector-like; weights to use in the estimation
    bootstrap_stat: Statistic to bootstrap; must be a key in model.est
    seed: Scalar; seed to use if fix_seed is True
    fix_seed: Boolean; if True, fixes the seed at the provided value
    copy_data: Boolean; if True, copies input data
    kwargs_fit: Other keyword arguments, which will be passed on to model.fit()

    Output
    res: k by 1 vector, estimated statistics for the bootstrap sample
    """

    # Make sure the underlying model object remains unchanged (that can be a
    # problem when not running this in parallel, for example)
    model = cp.copy(model)

    # Copy data if necessary
    if copy_data:
        X = cp.copy(X)
        y = cp.copy(y)
        W = cp.copy(weights)

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
        idx = scs.randint(low=0, high=n).rvs(size=n, random_state=seed)
    else:
        # Otherwise, draw wfully randomly
        idx = scs.randint(low=0, high=n).rvs(size=n)

    # Draw samples of observations according to idx
    Xstar = X[idx,:]
    ystar = y[idx,:]
    Wstar = W[idx,:]

    # Fit the model
    model.fit(X=Xstar, y=ystar, weights=Wstar, **kwargs_fit)

    # Get the estimated coefficients
    res = cvec(model.est[bootstrap_stat])

    # Return the result
    return res


# Define one iteration of a wild bootstrap
def _b_iter_wild(model_res, model, X, U_hat_res, impose_null, weights=None,
                 bootstrap_stat='coefficients', eta=1, seed=0, fix_seed=True,
                 copy_data=True, **kwargs_fit):
    """ Run one iteration of the wild bootstrap

    Inputs
    model_res: Model object, must have a fit() and a simulate() method and
               return a dictionary of results model_res.est. Underlying
               restricted model used for the bootstrap estimation.
    model: Model object, must have a fit() method and return a dictionary of
           results model.est. Underlying unrestricted model used for the
           bootstrap estimation.
    X: n by k matrix-like; RHS variables
    U_hat_res: n by 1 vector-like; residuals from the restricted model fit on
               the original data
    impose_null: k by 1 Boolean vector-like, the null will be imposed whereever
                 impose_null is True
    weights: n by 1 vector-like, weights to use for estimations
    bootstrap_stat: Statistic to bootstrap, must be a key in model.est
    eta: Scalar, absolute value of the two possible values of the two point
         distribution used to generate bootstrapped residuals
    seed: Scalar, seed to use if fix_seed is True
    fix_seed: Boolean, if True, fixes the seed at the provided value
    copy_data: Boolean, if True, copies the data before using them
    kwargs_fit: Other keyword arguments, which will be passed on to model.fit()

    Output
    res: k by 1 vector, estimated statistics for the bootstrap sample
    """

    # Make sure the underlying model objects remain unchanged (that can be a
    # problem when not running this in parallel, for example)
    model_res = cp.copy(model_res)
    model = cp.copy(model)

    # Copy data if needed
    if copy_data:
        X = cp.copy(X)
        U_hat_res = cp.copy(U_hat_res)

    # Set up X for the restricted model, ensuring that it is two dimensional
    if X.shape[1] == 1:
        X_res = cvec(X)
    else:
        X_res = np.array(X)

    # Subset to X variables which no null was imposed on
    X_res = X_res[:, ~impose_null]

    # Get the number of observations n
    n = X.shape[0]

    # Get modified residuals, starting with disturbances drawn from a two point
    # distribution
    #
    # Check whether to fix the seed
    if fix_seed:
        # If so, draw with a fixed seed
        E = scs.bernoulli(p=.5).rvs(size=n, random_state=seed)
    else:
        # Otherwise, draw fully randomly
        E = scs.bernoulli(p=.5).rvs(size=n)

    # Replace any zeros as -1
    E[E==0] = -1

    # Multiply by eta and convert to a proper column vector
    E = cvec(E * eta)

    # Get residuals times disturbance term
    U_hatstar = U_hat_res * E

    # Simulate outcomes
    ystar = model_res.predict(X_res) + U_hatstar

    # Fit the unrestricted model to the bootstrap sample
    model.fit(X=X, y=ystar, weights=weights, **kwargs_fit)

    # Get the estimated statistics
    res = cvec(model.est[bootstrap_stat])

    # Return the result
    return res

# Define one iteration of the Cameron, Gelbach, and Miller (2008) cluster robust
# wild bootstrap
def _b_iter_cgm(model_res, model, X, U_hat_res, impose_null, clusters, weights,
                bootstrap_stat='coefficients', eta=1, seed=0, fix_seed=True,
                copy_data=True, **kwargs_fit):
    """ Run one iteration of the Cameron, Gelbach, and Miller (2008) bootstrap

    Inputs
    model_res: Model object, must have a fit() and a simulate() method and
               return a dictionary of results model_res.est. Underlying
               restricted model used for the bootstrap estimation.
    model: Model object, must have a fit() method and return a dictionary of
           results model.est. Underlying unrestricted model used for the
           bootstrap estimation.
    X: n by k matrix-like, RHS variables
    U_hat_res: n by 1 vector-like, residuals from the restricted model fit on
               the original data
    impose_null: k by 1 Boolean vector-like, the null will be imposed whereever
                 impose_null is True
    clusters: n by 1 vector-like, cluster indices, have to be able to be used as
              an index for a numpy array
    weights: n by 1 vector-like, weights to use for estimations
    bootstrap_stat: Statistic to bootstrap, must be a key in model.est
    eta: Scalar, absolute value of the two possible values of the two point
         distribution used to generate bootstrapped residuals
    seed: Scalar, seed to use if fix_seed is True
    fix_seed: Boolean, if True, fixes the seed at the provided value
    copy_data: Boolean, if True, copies the data before using them
    kwargs_fit: Other keyword arguments, which will be passed on to model.fit()

    Output
    res: k by 1 vector, estimated statistics for the bootstrap sample
    """

    # Make sure the underlying model objects remain unchanged (that can be a
    # problem when not running this in parallel, for example)
    model_res = cp.copy(model_res)
    model = cp.copy(model)

    # Copy data if needed
    if copy_data:
        X = cp.copy(X)
        U_hat_res = cp.copy(U_hat_res)

    # Set up X for the restricted model, ensuring that it is two dimensional
    if X.shape[1] == 1:
        X_res = cvec(X)
    else:
        X_res = np.array(X)

    # Subset to X variables which no null was imposed on
    X_res = X_res[:, ~impose_null]

    # Get the number of clusters in the data
    J = len(np.unique(clusters))

    # Get modified residuals, starting with disturbances drawn from a two point
    # distribution
    #
    # Check whether to fix the seed
    if fix_seed:
        # If so, draw with a fixed seed
        E = scs.bernoulli(p=.5).rvs(size=J, random_state=seed)
    else:
        # Otherwise, draw fully randomly
        E = scs.bernoulli(p=.5).rvs(size=J)

    # Replace any zeros as -1
    E[E==0] = -1

    # Multiply by eta and convert to a proper column vector
    E = cvec(E * eta)

    # Use cluster indices to assign each unit its cluster's disturbance
    E = E[clusters[:,0],:]

    # Get residuals times disturbance term
    U_hatstar = U_hat_res * E

    # Simulate outcomes using the restricted model
    ystar = model_res.predict(X_res) + U_hatstar

    # Fit the unrestricted model to the bootstrap sample
    model.fit(X=X, y=ystar, clusters=clusters, weights=weights, **kwargs_fit)

    # Get the estimated statistics
    res = cvec(model.est[bootstrap_stat])

    # Return the result
    return res

################################################################################
### 3.2: Class to implement bootstrapping algorithms
################################################################################


class boot():
    """ Runs bootstrap algorithms """


    # Define initialization function
    def __init__(self, X, y=None, model=ols, algorithm='pairs',
                 impose_null_idx=None, eta=1, bootstrap_stat='t', level=None,
                 one_sided=None, clusters=None, weights=None,
                 get_boot_cov=False, residuals=None, B=4999, store_bsamps=False,
                 par=True, corecap=np.inf, fix_seed=True, batch_size='auto',
                 verbose=True, fprec=np.float64, nround=4,
                 labencode=skp.LabelEncoder(), **kwargs_fit):
        """ Initialize boot class

        Inputs
        X: n by k matrix-like; RHS variables
        y: n by 1 vector-like or None; outcome variable. If model has not been
           fit, boot() can fit the model itself, given y was provided. Needed
           for the pairs bootstrap.
        model: Model, must have certain methods and contain specific named
               objects. All models in lmpy follow the conventions needed to be
               used as an input for boot(). Underlying model used to estimate
               statistics of interest on bootstrap samples.
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
        clusters: n by 1 vector-like or None, clusters to use for CGM algorithm
        weights: n by 1 vector-like or None, weights to use in estimations
        get_boot_cov: Boolean, whether to calculate bootstrap covariance matrix;
                      if True, calculates covariance matrix based on bootstrap
                      samples, and returns it as part of self.est
        residuals: n by 1 vector-like or None, residuals from fitting the model
                   (restricted model if a null is being imposed); not needed if
                   y was provided
        B: Positive integer, number of bootstrap iterations to use
        store_bsamps: Boolean, if True, stores bootstrap samples as self.bsamps
        par: Boolean, if True, runs bootstrap iterations in parallel
        corecap: Integer or np.inf, maximum number of cores to use. Setting this
                 to np.inf uses all available cores.
        fix_seed: Boolean, if True, seeds are fixed throughout the bootstrapping
                  routine, to ensure replicability
        batch_size: Scalar or 'max' or 'auto', batch size in joblib.Parallel.
                    See the joblib documentation for details. If set to 'auto',
                    uses the joblib default heuristic. If set to 'max', uses
                    batch_size = B / (cores used).
        verbose: Boolean, if True, boot() prints notes and warnings
        fprec: Float data type, all floats will be cast to this type. The
               default value is np.float64, to ensure high precision. Using
               np.float32 instead can speed up NumPy's linear algebra in some
               settings.
        nround: Integer, results will be rounded to this number of decimals
        labencode: Label encoder, has to be able to convert an array of labels
                   (numbers or strings) into a numerical numpy array
        kwargs_fit: Other keyword arguments, which will be passed on to the
                    .fit() methods of any models
        """

        # Instantiate parameters
        #
        # Instantiate model to use for bootstrapping
        self.model = cp.deepcopy(model)

        # Instantiate verbosity parameter
        self.verbose = verbose

        # Check whether the model came pre fitted
        if self.model.est is None:
            if y is not None:
                # If not, fit the model
                self.model.fit(X, y, **kwargs_fit)
            else:
                raise ValueError('Error in boot(): Model has not been pre-fit, '
                                 + 'and data to fit on were not provided; '
                                 + 'either provide a pre-fit model, or '
                                 + 'complete data')

        # Instantiate float precision data type
        self.fprec = fprec

        # Instantiate parameters for running bootstrapping algorithms
        self.algorithm = algorithm
        self.stat = bootstrap_stat
        self.B = B
        self.impose_null_idx = impose_null_idx
        self.eta = eta
        self.get_boot_cov = get_boot_cov
        self.store_bsamps = store_bsamps

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

        if one_sided is not None:
            self.one_sided = one_sided
        else:
            self.one_sided = False

        # Instantiate parameters for self.summarize() results
        self.nround = nround

        # Variables created by self.summarize()
        self.regtable = None

        # Convert cluster variable to numerically encoded, if it was provided (I
        # do this so these can be used to index a numpy array when used in
        # algorithms, which doesn't work with string data)
        if clusters is not None:
            if isinstance(clusters, pd.Series):
                clusters = cvec(labencode.fit_transform(clusters))
            else:
                clusters = cvec(labencode.fit_transform(clusters.flatten()))

        # Run functions to get bootstrapped confidence interval
        # Get bootstrap distribution
        bsamps = self.bootstrap_distribution(
            X=X, y=y, U_hat_res=residuals, clusters=clusters, weights=weights,
            **kwargs_fit
        )

        # Get confidence intervals
        cidf = self.get_ci(bsamps)

        # Get p values
        pdf = self.get_p(bsamps)

        # Combine results into a dictionary
        self.est = {
            'original estimates': self.model.est[self.stat],
            'ci': cidf,
            'p': pdf,
            'statistic': self.stat,
            'B': self.B,
            'algorithm': self.algorithm
        }

        # Check whether a null was imposed
        if self.impose_null_idx is not None:
            if len(self.model.names_X) == self.impose_null_idx.shape[0] + 1:
                self.est['null imposed'] = (
                    pd.DataFrame(self.impose_null_idx,
                                 index=self.model.names_X[1:],
                                 columns=['Null imposed'])
                )
            else:
                self.est['null imposed'] = (
                    pd.DataFrame(self.impose_null_idx, index=self.model.names_X,
                                 columns=['Null imposed'])
                )

        # Check whether the bootstrapped covariance matrix has to be calculated
        if self.get_boot_cov:
            # If so, get the covariance matrix
            V_hat_boot = self.boot_cov(bsamps)

            # Record the matrix
            self.est['bootstrap covariance matrix'] = (
                V_hat_boot
            )

            # Record a Wald test based on the matrix
            self.est['wald test'] = (
                self.wald(V_hat_boot)
            )

        # Store bootstrap samples if desired
        if self.store_bsamps:
            self.bsamps = bsamps


    # Define function to run different bootstrapping algorithms
    def bootstrap_distribution(self, X, y, U_hat_res=None, clusters=None,
                               weights=None, **kwargs_fit):
        """ Get bootstrap distribution  """

        # Get number of columns of X (manually setting to one if it's a
        # Series, since that only has one dimension)
        if isinstance(X, pd.Series):
            #k = 1
            X = pd.DataFrame(X)
        #else:
        #    k = X.shape[1]

        # Check whether a vector indicating parameters which need to have a
        # null enforced was provided
        if self.impose_null_idx is not None:
            # Convert null indices to a proper column vector
            self.impose_null_idx = cvec(self.impose_null_idx)

            # Check whether the null index is one element too short, and the
            # model used added an intercept to the data
            if (
                    (self.impose_null_idx.shape[0] == X.shape[1] - 1)
                    and self.model.add_icept
                    and (self.stat != 'wald')
            ):
                # If so, add a False at the beginning, assuming that this
                # happened because the intercept was left out of the model
                self.impose_null_idx = (
                    np.concatenate([cvec(False), self.impose_null_idx], axis=0)
                )

            # Get part of X corresponding to unrestricted coefficients in the
            # restricted model
            X_res = np.array(X)
            X_res = X_res[:, ~self.impose_null_idx[:,0]]

            # Set up restricted model
            self.model_res = cp.deepcopy(self.model)

            # Fit the restricted model
            self.model_res.fit(
                X_res, y, clusters=clusters, weights=weights, coef_only=True,
                **kwargs_fit
            )

            # Set up a null impose Series to pass on to bootstrap algorithms
            impose_null_passon = self.impose_null_idx[:,0]
        else:
            # Otherwise, set self.model_res to self.model, so this can easily be
            # passed to algorithms which could have a null imposed
            self.model_res = cp.deepcopy(self.model)

            # Set up a null impose Series to pass on to bootstrap algorithms
            impose_null_passon = np.zeros(shape=X.shape[1]).astype(bool)

        # Get residuals if needed
        if (U_hat_res is None) and (self.algorithm.lower() in ['wild', 'cgm']):
            # Make sure LHS data were provided
            if y is not None:
                # Use correct model to get residuals (restricted if null is
                # imposed)
                if self.impose_null_idx is not None:
                    U_hat_res = cvec(y) - self.model_res.predict(X_res)
                else:
                    U_hat_res = cvec(y) - self.model.predict(X)
            else:
                # Otherwise, raise an error
                raise ValueError('Error in boot(): The chosen bootstrap '
                                 + 'algorithm ({}) '.format(self.algorithm)
                                 + 'requires residuals, but residuals were not '
                                 + 'provided, and data to fit on were not '
                                 + 'provided; either provide residuals, or '
                                 + 'complete data')

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
            bsamps = (
                jbl.Parallel(n_jobs=self.n_cores, batch_size=self.batch_size)(
                    jbl.delayed(_b_iter_pairs)(
                        model=self.model, X=X, y=y, clusters=clusters,
                        weights=weights, bootstrap_stat=self.stat, seed=b,
                        fix_seed=self.fix_seed, **kwargs_fit)
                    for b in np.arange(self.B)
                )
            )

        # Wild bootstrap
        elif self.algorithm.lower() == 'wild':
            # Get bootstrapped distribution for statistic
            bsamps = (
                jbl.Parallel(n_jobs=self.n_cores, batch_size=self.batch_size)(
                    jbl.delayed(_b_iter_wild)(
                        model_res=self.model_res, model=self.model, X=X,
                        U_hat_res=U_hat_res, clusters=clusters, weights=weights,
                        impose_null=impose_null_passon,
                        bootstrap_stat=self.stat, eta=self.eta, seed=b,
                        fix_seed=self.fix_seed, **kwargs_fit)
                    for b in np.arange(self.B)
                )
            )

        # Cameron, Gelbach, and Miller (2008) cluster robust bootstrap
        elif self.algorithm.lower() == 'cgm':
            # Get bootstrapped distribution for statistic
            bsamps = (
                jbl.Parallel(n_jobs=self.n_cores, batch_size=self.batch_size)(
                    jbl.delayed(_b_iter_cgm)(
                        model_res=self.model_res, model=self.model, X=X,
                        U_hat_res=U_hat_res, clusters=clusters, weights=weights,
                        impose_null=impose_null_passon,
                        bootstrap_stat=self.stat, eta=self.eta, seed=b,
                        fix_seed=self.fix_seed, **kwargs_fit)
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
        return np.concatenate(bsamps, axis=1)


    # Define a function to calculate bootstrapped confidence intervals
    def get_ci(self, bsamps):
        """ Get bootstrapped confidence interval """

        # Get quantiles to compute
        if self.one_sided == 'upper':
            quants = [0, 1 - self.level]
        elif self.one_sided == 'lower':
            quants = [self.level, 1]
        else:
            quants = [self.level/2, 1 - self.level/2]

        # Get the confidence intervals. Doing these individually allows me to
        # ensure that intervals are conservative, in the sense that if a
        # quantile lies between two bootstrap observations, the test picks the
        # one which will make the confidence interval largest.
        ci = np.zeros(shape=(bsamps.shape[0], 2))
        ci[:,0] = (
            np.quantile(
                bsamps, quants[0], axis=1, interpolation='lower'
            ).T.astype(self.fprec)
        )
        ci[:,1] = (
            np.quantile(
                bsamps, quants[1], axis=1, interpolation='higher'
            ).T.astype(self.fprec)
        )

        # Replace upper or lower bounds if applicable
        if (self.one_sided == 'upper') and self.stat in ['wald']:
            ci[:,0] = 0
        elif self.one_sided == 'upper':
            ci[:,0] = -np.inf
        elif self.one_sided == 'lower':
            ci[:,1] = np.inf

        # Make a two element list of names for the upper and lower bound
        names_ci = ['{}%'.format(q*100) for q in quants]

        # Combine the confidence interval and names into a DataFrame
        ci = pd.DataFrame(
            ci, index=self.model.est[self.stat].index, columns=names_ci
        )

        # Return the DataFrame
        return ci


    # Define a function to calculate bootstrapped p-values
    def get_p(self, bsamps):
        """ Get bootstrapped p-values """

        # Get the original estimate
        orig = self.model.est[self.stat]

        # Set up an empty DataFrame for the bootstrapped p-values
        p = pd.DataFrame(index=orig.index, columns=['p-value'], dtype=np.float)

        # Check whether no null needs to be imposed
        if self.impose_null_idx is None:
            # If so, set up a vector of False, indicating that no nulls should
            # ever be imposed
            imp0 = np.zeros(shape=p.shape).astype(bool)
        else:
            # Otherwise, get the imposed nulls
            imp0 = self.impose_null_idx

            # If one element is missing from imp0, add a first element
            # indicating that the intercept is unrestricted
            if imp0.shape[0] == p.shape[0] - 1:
                imp0 = (
                    np.concatenate([cvec(False), self.impose_null_idx], axis=0)
                )

        # Get bootstrapped p-values (this could be parallelized, altough it is
        # generally fast relative to getting the bootstrap samples)
        for i in np.arange(p.shape[0]):
            p.iloc[i, 0] = (
                _get_p_i(
                    bsamp=bsamps[i,:],
                    orig_estimate=self.model.est[self.stat].iloc[i,:].values,
                    one_sided=self.one_sided,
                    imp_null=imp0[i,0]
                )
            )

        # Return vector of bootstrapped p-values
        return p


    # Define a function to summarize the results
    def summarize(self):
        """ Produce a pandas DataFrame summarizing the results """

        # Make a regression table DataFrame
        self.regtable = (
            pd.concat(
                [self.est['original estimates'],
                 self.est['ci'],
                 self.est['p']
                ], axis=1
            )
        )

        # Add null imposed information if applicable
        if (
                self.impose_null_idx is not None
        ):
            self.regtable = (
                pd.concat([self.regtable, self.est['null imposed']], axis=1)
            )

            # The first row will be missing if there was an intercept in X, and
            # it looks nicer if the resulting NAN is replaced with False
            if self.impose_null_idx.shape[0] != self.regtable.shape[0]:
                self.regtable.iloc[0,-1] = False

        # Round the results
        self.regtable = self.regtable.round(self.nround)

        # Add a name to the DataFrame
        self.regtable.name = 'Bootstrapped ' + self.stat

        # Return the result, to make it easily printable
        return self.regtable


    # Define a function to calculate the covariance matrix across bootstrap
    # samples
    def boot_cov(self, bsamps, weights=None):
        """ Calculate covariance matrix of bootstrapped statistics """

        # Calculate covariance
        return np.cov(bsamps, rowvar=True, aweights=weights)


    # Define a function to run Wald test
    def wald(self, V_hat_boot=None, jointsig=None, R=None, b=None,
             weights=None):
        """ Run a Wald test

        First run boot_cov(), then use this to calculate a Wald statistic. Note
        that this is usually not a good idea, since using the bootstrap to
        estimate statistics, rather than the distribution of a pivot, does not
        provide any asymptotic refinements over analytical methods. The only
        real use case is if the covariance matrix of the underlying model is too
        complicated to be estimated analytically.

        Inputs
        V_hat_boot, jointsig, R, b: See self.model.wald's documentation

        Outputs
        self.model.waldtable: See self.model.wald's documentation
        """

        # Check whether self.boot_cov() has been runa bootstrapped covariance
        # matrix was provided
        if V_hat_boot is not None:
            # If so, get a Wald statistic using the bootstrapped covariance
            # matrix
            self.model.wald(jointsig=jointsig, R=R, b=b, V_hat=V_hat_boot)
        else:
            # Otherwise, potentially display a warning
            if self.verbose:
                print('\nNote in boot.wald(): Please provide a covariance',
                      'matrix when running boot.wald(). Returning Wald',
                      'statistic based on model covariance instead.')

            # Just get the Wald statistic from the underlying model
            self.model.wald(jointsig=jointsig, R=R, b=b, weights=weights)

        # Save the results
        self.W_boot = self.model.W
        self.pW_boot = self.model.pW

        # Return the summary table
        return self.model.waldtable
