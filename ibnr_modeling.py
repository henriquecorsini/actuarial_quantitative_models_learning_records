# This is a quite simple Monte Carlo-flavoured calculator to estimate IBNRs.
# I chose to write this project due to the apparent scarcity of real data,
# whose origin I can ascertain. Unfortunately, this means that the modelling
# numbers might be somewhat unrealistic. I claim, nonetheless, that
# the numbers were 'cooked' with the INBR data found in the first chapter of
# 'LOSS RESERVING An Actuarial Perspective' written by Greg Taylor in mind.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# I first begin by describing the method by which model IBNRs are obtained.
# This is done by randomly selecting a parameter lambda, which is then used
# to select a number of IBNRs from a Poisson distribution with parameter
# lambda (the same lambda randomly selected).

def claim_number_model(rng, lambda_sampler, *, size=(), **sampler_kwargs):
    lam = np.asarray(lambda_sampler(rng=rng, size=size, **sampler_kwargs), dtype = float)
    if np.any(lam < 0):
        raise ValueError("Poisson intensities must be nonnegative.")
    claims = rng.poisson(lam=lam)
    return claims, lam

-- Examples of lambda_sampler.

def lognormal_lambda_sampler(rng, mean, sigma, size=()):
    return rng.lognormal(mean=mean, sigma=sigma, size=size)

def normal_lambda_sampler(rng, loc, scale, size=()):
    lam = rng.normal(loc=loc, scale=scale, size=size)
    return np.clip(lam, 0.0, None)

-- ...

def ibnr_claim_count_model(rng, n_occurrence_periods, reporting_pattern, lambda_sampler, *, sampler_kwargs = None):
    
    sampler_kwargs = sampler_kwargs or {}
    reporting_pattern = np.asarray(reporting_pattern, dtype=float)
    
    if np.any(reporting_pattern < 0) or np.any(reporting_pattern > 1):
        raise ValueError("reporting probabilities must lie in [0,1].")
    
    ultimate_counts, lambdas = claim_number_model(rng, lambda_sampler, size=n_occurrence_periods, **sampler_kwargs)
    reported_incremental = np.zeros((n_occurrence_periods, n_development_periods), dtype=int)
    remaining = ultimate_counts.copy()

    for i in range(0, n_occurrence_periods):
        for j in range(0, n_occurrence_periods - i):
            if j == n_development_periods - 1:
                reported_incremental[i,j] = remaining[i]
            else:
                reported_incremental[i,j] = rng.binomial(remaining[i], reporting_pattern[j])
                remaining[i] -= reported_incremental[i,j]
            
    reported_cumulative = reported_incremental.cumsum(axis=1)
    ibnr_by_occurrence = ultimate_counts - reported_cumulative[:,-1]

    triangle = pd.DataFrame(
        reported_cumulative,
        index=[f"AY_{i+1}" for i ]
    )
