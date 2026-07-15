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
    
    reported_incremental = np.zeros((n_occurrence_periods, n_occurrence_periods), dtype=int)
    remaining = ultimate_counts.copy()

    for i in range(0, n_occurrence_periods):
        for j in range(0, n_occurrence_periods - i):
            if j>=len(reporting_pattern):
                binom_parameter = np.clip(rng.normal(loc=reporting_pattern[-1], scale=0.1*reporting_pattern[-1]), 0.0, 1.0)
            else:
                binom_parameter = np.clip(rng.normal(loc=reporting_pattern[j], scale=0.1*reporting_pattern[j]), 0.0, 1.0)
            reported_incremental[i,j] = rng.binomial(remaining[i], binom_parameter)
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
    
def fit_exponential_tail_allowing_zeros(temporal_averages, threshold_ratio=0.05):
    temporal_averages = np.asarray(temporal_averages, dtype=float)

    if temporal_averages.ndim != 1:
        raise ValueError("temporal_averages must be a 1D array.")
    if np.any(temporal_averages < 0):
        raise ValueError("temporal_averages must be non-negative.")

    max_val = temporal_averages.max()
    threshold = threshold_ratio * max_val

    candidates = np.flatnonzero(temporal_averages < threshold)
    if len(candidates) == 0:
        raise ValueError("No index j satisfies the threshold condition.")

    j_tail = candidates[0]

    tail = temporal_averages[j_tail:]
    x = np.arange(len(tail))

    positive_mask = tail > 0
    if positive_mask.sum() < 2:
        raise ValueError("Not enough positive points to fit an exponential.")

    x_fit = x[positive_mask]
    y_fit = tail[positive_mask]

    slope, intercept = np.polyfit(x_fit, np.log(y_fit), 1)

    if slope>=0:
        raise ValueError("The fitted exponential tail is not decreasing.")

    a = np.exp(intercept)
    b = slope
    fitted_tail = a * np.exp(b * x)

    fitted_temporal_averages = temporal_averages.copy()
    fitted_temporal_averages[j_tail:] = fitted_tail

    return j_tail, a, b, fitted_temporal_averages

def estimating_ibnr_count_by_smoothed_means(reported_incremental_df, threshold_ratio=0.05):
    estimated_claim_count = reported_incremental_df.copy()
    row_amount, col_amount = estimated_claim_count.shape

    temporal_averages = []

    for j in range(col_amount):
        observed_rows = row_amount - j
        if observed_rows > 0:
            temporal_averages.append(
                estimated_claim_count.iloc[:observed_rows, j].mean()
            )
        else:
            temporal_averages.append(np.nan)

    j_tail, a, b, temporal_averages = fit_exponential_tail_allowing_zeros(
        temporal_averages, threshold_ratio
    )

    for j in range(col_amount):
        observed_rows = row_amount - j
        if observed_rows < row_amount:
            estimated_claim_count.iloc[observed_rows:row_amount, j] = temporal_averages[j]

    remaining_estimated_ibnr = a * np.exp(b * (col_amount - j_tail)) / (1 - np.exp(b))

    return estimated_claim_count, remaining_estimated_ibnr
