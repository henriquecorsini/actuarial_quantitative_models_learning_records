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

# Examples of lambda_sampler.

def lognormal_lambda_sampler(rng, mean, sigma, size=()):
    return rng.lognormal(mean=mean, sigma=sigma, size=size)

def normal_lambda_sampler(rng, loc, scale, size=()):
    lam = rng.normal(loc=loc, scale=scale, size=size)
    return np.clip(lam, 0.0, None)

# It is now time to introduce an model generating claim count data. This 
# will be used in the future to test and compare methods which are used to
# estimate IBNR numbers in real scenarios. Let me describe the hypotheses
# of the model. I assume that the matter is dealt in the present and that 
# the insurance portfolio is static in time, which is often reasonable in 
# small time horizons. Hence, there is no need to introduce any counting 
# variable besides n_occurrence_periods: staticness means that the whole
# history of the portfolio is known. The estimation of the amount of claims
# in development period k is done in a roundabout way by introducing a 
# reporting pattern vector, which estimates the probability that an 
# unreported claim be reported after k periods. It is reasonable to assume
# that such reporting pattern vector be increasing: the longer a claim has 
# remained unreported, the more likely it is that it should be reported in
# the near future.

def ibnr_claim_count_model(rng, n_occurrence_periods, reporting_pattern, lambda_sampler, *, sampler_kwargs = None):
    
    sampler_kwargs = sampler_kwargs or {}
    reporting_pattern = np.asarray(reporting_pattern, dtype=float)
    
    if np.any(reporting_pattern < 0) or np.any(reporting_pattern > 1):
        raise ValueError("reporting probabilities must lie in [0,1].")
    
    ultimate_counts, lambdas = claim_number_model(rng, lambda_sampler, size=n_occurrence_periods, **sampler_kwargs)
    
    # This models the TOTAL amount of claims that occured in each period.
    # It is now necessary to model when they were reported. This shall be
    # done using the parameters from reporting_pattern.
    
    reported_incremental = np.zeros((n_occurrence_periods, n_occurence_periods), dtype=int)
    remaining = ultimate_counts.copy()

    for i in range(0, n_occurrence_periods):
        for j in range(0, n_occurrence_periods - i):
            if j>=len(report_pattern):
                reported_incremental[i,j] = rng.binomial(remaining[i], reporting_pattern[-1])
            else:
                reported_incremental[i,j] = rng.binomial(remaining[i], reporting_pattern[j])
            remaining[i] -= reported_incremental[i,j]
            
    reported_cumulative = reported_incremental.cumsum(axis=1)
    ibnr_by_occurrence = ultimate_counts - reported_cumulative[:,-1]

    triangle = pd.DataFrame(
        reported_cumulative,
        index=[f"AY_{i}" for i in range(0, n_occurrence_periods)],
        columns = [f"Dev_{j}" for j in range(0, n_occurrence_periods)]
    )

    summary = pd.DataFrame({
        "occurrence_period": [f"AY_{i}" for i in range(0, n_occurrence_periods)],
        "lambdas": lambdas,
        "ultimate_claim_count": ultimate_counts,
        "reported_to_date": reported_cumulative[:,-1],
        "ibnr_count": ibnr_by_occurrence
    })

    total_ibnr = int(ibnr_by_occurrence.sum())

    return {
        "triangle_cumulative": triangle,
        "reported_incremental": reported_incremental,
        "summary": summary,
        "total_ibnr": total_ibnr
    }
