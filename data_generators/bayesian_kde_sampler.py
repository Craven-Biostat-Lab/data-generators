# BSD 3-Clause License
#
# Copyright (c) 2020, Mark Craven's Research Lab

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from data_generators.tnkde import TruncatedNormalKernelDensity
from pomegranate import *

class BayesianSampleGenerator():
    def fit(self, data, discrete_features=None, bandwidth=1.0,
            num_discretization_bins=4, pseudocount=1.0):
        """Fit the Chow Liu Bayesian Sampler on the data.

        Parameters
        ----------
        data : array_like, shape (n_samples, n_features)
            List of data points to train from.
        discrete_features : array_like, shape (n_features)
            Array with true for discrete features and false for continuous
            features. If None all features are treated as continuous.
        bandwidth : double
            Bandwidth of the truncated kde to use for the continuous features.
        num_discretization_bins : int
            Number of bins used to discretize the continuous features. Uses less
            bins for features where the bin width is too narrow.
        pseudocount : double
            Pseudocount to use in the bayesian network.
        """
        if bandwidth <= 0:
            raise ValueError("Bandwidth must be positive.")
            
        if discrete_features != None and \
           len(discrete_features) != data.shape[1]:
            raise ValueError("Discrete features array and data arrays"
                             "shape don't match.")
            
        if num_discretization_bins < 0:
            raise ValueError("Number of descretization bins can't be negetive.")
        
        if num_discretization_bins == 0:
            for bool in discrete_features:
                if bool:
                    raise ValueError("Number of descretization bins can't be"
                                      "zero if there is a continuous feature.")
            
        if pseudocount < 0:
            raise ValueError("Pseudocount can't be negative.")
        
        if discrete_features == None:
            discrete_features = [False] * data.shape[1]

        self.num_features_ = data.shape[1]
        self.discrete_features_ = discrete_features
        self.num_discretization_bins_ = num_discretization_bins

        discretized_data = np.array(data, copy=True)
        continuous_data = data[:, np.invert(discrete_features)]

        discretizer = KBinsDiscretizer(n_bins=num_discretization_bins,
                                    encode='ordinal', strategy='quantile')
        discretizer.fit(continuous_data)

        discretized_data[:, np.invert(discrete_features)] = \
                        discretizer.transform(continuous_data)
        self.discretizer_ = discretizer

        self.model_ = BayesianNetwork.from_samples(discretized_data,
                      algorithm='chow-liu', n_jobs=-1, pseudocount=pseudocount)
        self.model_.bake()
        
        # Table for bin edges
        bins = discretizer.bin_edges_

        # Kdes for continuous data.
        self.tnkdes_ = []

        i = 0
        for k in range(self.num_features_):
            if discrete_features[k]:
                continue
            
            bins[i][0] = -np.inf
            bins[i][len(bins[i]) - 1] = np.inf
            bin_kdes = []
            
            # loop of boundary
            for j in range(len(bins[i]) - 1):
                # Bound for this bin.
                lower_bound = bins[i][j]
                upper_bound = bins[i][j+1]
                
                # Create a kde using the data in the current bin.
                current_feature_data = data[:, k]
                cur_bin_data = current_feature_data[discretized_data[:, k] == j]
                kde = TruncatedNormalKernelDensity(bandwidth=bandwidth,
                        lowerbound=lower_bound, upperbound=upper_bound)
                kde.fit(cur_bin_data)
                bin_kdes.append(kde)
                
            i = i + 1
            self.tnkdes_.append(bin_kdes)

    def sample(self, num_samples=1, burnin_period=100, constraints={}):
        """Get new samples similar to the training data.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples to return. Defaults to 1.
        burnin_period : int, optional
            Burn-in period before drawing each instance from the gibbs sampler.
            Defaults to 100.
        constraints : dict, optional
            Evidence to set constant while samples are generated. The format of
            the disctionry enteries are 'string representation of feature
            number' : val. For example: {'0' : 6.0, '5' : -4.5}
        """
        if num_samples <= 0:
            raise ValueError("Number of samples requested must be positive.")
            
        if burnin_period < 0:
            raise ValueError("Burn-in period can't be negative.")
            
        if constraints == None:
            constraints = {}
        
        original_constraints = constraints.copy()
        constraints_array = np.array([0] * self.num_features_)
        
        i = 0
        for key in constraints:
            constraints_array[int(key)] = constraints[key]
        
        constraints_array[np.invert(self.discrete_features_)] = \
        self.discretizer_.transform(
            constraints_array[np.invert(self.discrete_features_)].reshape(1, -1)
            )[0]

        for key in constraints:
            constraints[key] = constraints_array[int(key)]

        # Get samples from the bayesian net. We still need to sample the kdes
        # for the continuos data.
        sample_table = np.array(self.model_.sample(n=num_samples,
                                burnin=burnin_period, evidence=constraints))
        
        # Loop over all continuos features.
        i = 0
        for k in range(self.num_features_):
            if self.discrete_features_[k]:
                continue

            # Loop over all bins for the feature.
            for j in range(len(self.tnkdes_[i])):
                current_feature_samples = sample_table[:,k]
                num_samples_in_bin = \
                current_feature_samples[current_feature_samples == j].shape[0]
                
                # Add number of bins to avoid collisions in future iterations.
                # This is subtracted below.
                current_feature_samples[current_feature_samples == j] = \
                self.tnkdes_[i][j].sample(n_samples = num_samples_in_bin) + \
                self.num_discretization_bins_
                
                sample_table[:,k] = current_feature_samples
            
            i = i + 1
            # Subtract number of bins added above to avoid collisions.
            sample_table[:, k] -= self.num_discretization_bins_
            
        for key in original_constraints:
            if not self.discrete_features_[int(key)]:
                sample_table[:, int(key)] = original_constraints[key]

        return sample_table
