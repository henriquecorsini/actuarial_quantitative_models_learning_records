# This is a quite simple Monte Carlo-flavoured calculator to estimate IBNRs.
# I chose to write this project due to the apparent scarcity of real data,
# whose origin I can ascertain. Unfortunately, this means that the modelling
# numbers might be somewhat unrealistic. I claim, nonetheless, that
# the numbers were 'cooked' with the INBR data found in the first chapter of
# 'LOSS RESERVING An Actuarial Perspective' written by Greg Taylor in mind.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def claim_number_model(rng, lambda_sampler, *, size=(), **sampler_kwargs):
    lam = np.asarray(lambda_sampler(rng=rng, size=size, **sampler_kwargs), dtype = float)
    if np.any(lam < 0):
        raise ValueError("Poisson intensities must be nonnegative.")
    claims = rng.poisson(lam=lam)
    return claims, lam

def lognormal_lambda_sampler(rng, mean, sigma, size=()):
    return rng.lognormal(mean=mean, sigma=sigma, size=size)

def normal_lambda_sampler(rng, loc, scale, size=()):
    lam = rng.normal(loc=loc, scale=scale, size=size)
    return np.clip(lam, 0.0, None)

def ibnr_claim_count_model(rng, n_occurrence_periods, reporting_pattern, lambda_sampler, *, sampler_kwargs = None):

    sampler_kwargs = sampler_kwargs or {}
    reporting_pattern = np.asarray(reporting_pattern, dtype=float)

    if np.any(reporting_pattern < 0) or np.any(reporting_pattern > 1):
        raise ValueError("reporting probabilities must lie in [0,1].")

    ultimate_counts, lambdas = claim_number_model(rng, lambda_sampler, size=n_occurrence_periods, **sampler_kwargs)
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

    return ultimate_counts, reported_incremental

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
    estimated_claim_count = reported_incremental_df.copy().astype(float)
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
    estimated_claim_count["Remaining"] = remaining_estimated_ibnr

    return estimated_claim_count

def estimating_ibnr_count_by_a_smoothed_normalized_method(reported_incremental_df, threshold_ratio=0.05):
    estimated_claim_count = reported_incremental_df.copy().astype(float)
    row_amount, col_amount = estimated_claim_count.shape

    normalized_reports = reported_incremental_df.copy()
    normalized_reports = normalized_reports.div(normalized_reports.iloc[:,0], axis=0)

    normalized_parameters = [
        normalized_reports.iloc[:row_amount-j, j].mean() for j in range(col_amount)
        ]
    j_tail, a, b, smoothed_parameters = fit_exponential_tail_allowing_zeros(normalized_parameters, threshold_ratio)

    for j in range(col_amount):
        observed_rows = row_amount - j
        if observed_rows < row_amount:
            estimated_claim_count.iloc[observed_rows:row_amount, j] = smoothed_parameters[j] * estimated_claim_count.iloc[observed_rows:row_amount, 0]

    remaining_estimated_ibnr_parameter = a * np.exp(b * (col_amount - j_tail)) / (1 - np.exp(b))
    estimated_claim_count["Remaining"] = estimated_claim_count.iloc[:,0] * remaining_estimated_ibnr_parameter

    return estimated_claim_count

def estimating_ibnr_count_by_smoothed_simple_chain_ladder(reported_incremental_df, threshold_ratio=0.05):
    estimated_claim_count = reported_incremental_df.copy().astype(float)
    row_amount, col_amount = estimated_claim_count.shape

    cumsum_estimated_claim_count = estimated_claim_count.cumsum(axis=1)
    chain_parameters = pd.DataFrame(0, index=range(0,row_amount), columns=range(0,col_amount))
    chain_parameters.iloc[:, 0] = 1.0

    for j in range(1,col_amount):
        chain_parameters.iloc[:, j] = cumsum_estimated_claim_count.iloc[:, j] / cumsum_estimated_claim_count.iloc[:, j-1]

    average_parameters = [chain_parameters.iloc[:j,j].mean() for j in range(1,col_amount)] - 1
    j_tail, a, b, smoothed_parameters = fit_exponential_tail_allowing_zeros(average_parameters, threshold_ratio)
    remaining_estimated_ibnr_parameter = a * np.exp(b * (col_amount - j_tail)) / (1 - np.exp(b))

    for j in range(1, col_amount):
        observed_rows = row_amount - j
        if observed_rows < row_amount:
            estimated_claim_count.iloc[observed_rows:row_amount, j] = smoothed_parameters[j-1]

