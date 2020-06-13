# Truncated Normal KDE
#
# Licensed under the BSD 3-Clause License
# Copyright (c) 2019, Yuriy Sverchkov

from scipy.stats import truncnorm
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from scipy.special import logsumexp
import numpy as np


class TruncatedNormalKernelDensity(BaseEstimator):
    """Kernel Density Estimation on Bounded Support Using Truncated Normal Kernels

    Parameters
    ----------
    bandwidth : float
        The bandwidth of the kernel.

    lowerbound : float
        The lower-bound of the support.

    upperbound : float
        The upper-bound of the support.
    """
    def __init__(self,
                 bandwidth: float = 1.0,
                 lowerbound: float = -np.inf,
                 upperbound: float = np.inf):

        self.bandwidth = bandwidth
        self.lowerbound = lowerbound
        self.upperbound = upperbound

        if not bandwidth > 0:
            raise ValueError("Bandwidth must be positive.")

        if upperbound < lowerbound:
            raise ValueError(f"Upperbound (got {upperbound}) can't be lower than lowerbound (got {lowerbound})!")

    def fit(self, X, y=None, sample_weight=None):
        """Fit the Kernel Density model on the data.

        Parameters
        ----------
        X : array_like, shape (n_samples,)
            list of data points.
        y : not used.
        sample_weight : array_like, shape (n_samples,), optional
            list of sample weights attached to data X.
        """
        # X = check_array(X, order='C', dtype=DTYPE)
        self.points_ = X

        if sample_weight is None:
            weights = np.ones(shape=X.shape)
        else:
            # TODO Checks
            weights = sample_weight

        self.log_weights_ = np.log(weights)
        self.log_normalizer_ = np.log(sum(weights))
        self.lowers_ = (self.lowerbound - self.points_) / self.bandwidth
        self.uppers_ = (self.upperbound - self.points_) / self.bandwidth

        return self

    def score_samples(self, X):
        # TODO Array shape check
        samples = X.reshape(-1)
        return(
            logsumexp([truncnorm.logpdf(samples, a=lower, b=upper, loc=mean, scale=self.bandwidth) + log_weight for
                       (mean, lower, upper, log_weight) in
                       zip(self.points_, self.lowers_, self.uppers_, self.log_weights_)],
                      axis=0,
                      return_sign=False) - self.log_normalizer_)

    def score(self, X):
        return(sum(self.score_samples(X)))
    
    def sample(self, n_samples=1, random_state=None):

        rng = check_random_state(random_state)

        using_points = rng.choice(
            a=len(self.points_),
            p=np.exp(self.log_weights_ - self.log_normalizer_).reshape(-1),
            size=n_samples,
            replace=True)

        samples = truncnorm.rvs(
            a=self.lowers_[using_points],
            b=self.uppers_[using_points],
            loc=self.points_[using_points],
            scale=self.bandwidth,
            #size=,
            random_state=rng
        )

        return(samples)


# Test script
if __name__ == "__main__":

    from sklearn.neighbors import KernelDensity

    print("Training set:")
    x = np.random.normal(0, 10, 10).reshape(-1, 1)
    print(x)

    print("Test set:")
    y = np.random.normal(0, 100, 8).reshape(-1, 1)
    print(y)

    print("Kernel bandwidth:")
    bw = np.random.uniform(1, 5)
    print(bw)

    print("Our KDE:")
    my_kde = TruncatedNormalKernelDensity(bandwidth=bw)
    my_kde.fit(x)
    print(my_kde.score_samples(y))
    print(my_kde.score(y))

    print("SciKitLearn KDE:")
    skl_kde = KernelDensity(kernel='gaussian', bandwidth=bw)
    skl_kde.fit(x)
    print(skl_kde.score_samples(y))
    print(skl_kde.score(y))

    print("Test that truncation works:")
    y_vals = sorted(y)
    up = y_vals[5]
    low = y_vals[2]

    print(f"With upperbound {up}:")
    up_kde = TruncatedNormalKernelDensity(bandwidth=bw, upperbound=up)
    up_kde.fit(x)
    print(up_kde.score_samples(y))

    print(f"With lowerbound {low}:")
    low_kde = TruncatedNormalKernelDensity(bandwidth=bw, lowerbound=low)
    low_kde.fit(x)
    print(low_kde.score_samples(y))

    print(f"With upperbound {up} and lowerbound {low}:")
    both_kde = TruncatedNormalKernelDensity(bandwidth=bw, upperbound=up, lowerbound=low)
    both_kde.fit(x)
    print(both_kde.score_samples(y))
