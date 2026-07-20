# This is a quite simple Monte Carlo-flavoured calculator to estimate IBNRs.
# I chose to write this project due to the apparent scarcity of real data,
# whose origin I can ascertain. Unfortunately, this means that the modelling
# numbers might be somewhat unrealistic. I claim, nonetheless, that
# the numbers were 'cooked' with the INBR data found in the first chapter of
# 'LOSS RESERVING An Actuarial Perspective' written by Greg Taylor in mind.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Abstract functions introducing randomness in the model

def lognormal_lambda_sampler(
    rng, 
    mean, 
    sigma,
    *, 
    size=()
):
    return rng.lognormal(mean=mean, sigma=sigma, size=size)

def normal_lambda_sampler(
    rng, 
    loc, 
    scale,
    *, 
    size=()
):
    lam = rng.normal(loc=loc, scale=scale, size=size)
    return np.clip(lam, 0.0, None)

def noisy_hazard(
    rng, 
    base_p,
    *, 
    rel_scale=0.1, 
    min_scale=0.01
):
    scale = max(rel_scale * base_p, min_scale)
    return np.clip(rng.normal(loc=base_p, scale=scale), 0.0, 1.0)

def total_claim_number_model(
    rng,
    *,
    lambda_sampler,
    size=(), 
    **sampler_kwargs
):
    lam = np.asarray(
        lambda_sampler(rng=rng, size=size, **sampler_kwargs), 
        dtype = float
    )
    if np.any(lam < 0):
        raise ValueError("Poisson intensities must be nonnegative.")
    claims = rng.poisson(lam=lam)
    return claims

def reported_claims_model(
    rng, 
    occurrence_periods,
    *, 
    reporting_pattern, 
    lambda_sampler,
    sampler_kwargs = None
):
    sampler_kwargs = sampler_kwargs or {}
    reporting_pattern = np.asarray(reporting_pattern, dtype=float)

    if np.any(reporting_pattern < 0) or np.any(reporting_pattern > 1):
        raise ValueError("reporting probabilities must lie in [0,1].")

    ultimate_counts = total_claim_number_model(rng, 
                                               lambda_sampler=lambda_sampler,
                                               size=occurrence_periods,
                                               **sampler_kwargs
    )

    row_amount, col_amount = occurrence_periods, occurrence_periods
    reported_incremental = np.full((row_amount, col_amount), np.nan) 
    remaining = ultimate_counts.copy()

    for i in range(row_amount):
        obs_columns = col_amount - i
        for j in range(obs_columns):
            if j>=len(reporting_pattern):
                prob = noisy_hazard(rng, reporting_pattern[-1])
            else:
                prob = noisy_hazard(rng, reporting_pattern[j])
            reported_incremental[i,j] = rng.binomial(remaining[i], prob)
            remaining[i] -= reported_incremental[i,j]
    
    return ultimate_counts, reported_incremental

def fit_exp_tail(
    num_array,
    *, 
    threshold_ratio=0.05,
    return_array = False
):
    num_array = np.asarray(num_array, dtype=float)
    if num_array.ndim != 1:
        raise ValueError("num_array must be a 1D array.")
    if np.any(num_array < 0):
        raise ValueError("num_array must be non-negative.")

    max_val = num_array.max()
    threshold = threshold_ratio * max_val
    remaining = 0.0

    candidates = np.flatnonzero(num_array < threshold)
    if len(candidates) == 0:
        print("No index j satisfies the threshold condition.")
        return num_array, remaining

    j_tail = candidates[0]

    tail = num_array[j_tail:]
    x = np.arange(len(tail))

    positive_mask = tail > 0
    if positive_mask.sum() < 2:
        print("Not enough positive points to fit an exponential.")
        return num_array, remaining

    x_fit = x[positive_mask]
    y_fit = tail[positive_mask]

    slope, intercept = np.polyfit(x_fit, np.log(y_fit), 1)

    a = np.exp(intercept)
    b = slope
    fitted_tail = a * np.exp(b * x)

    fitted_num_array = num_array.copy()
    fitted_num_array[j_tail:] = fitted_tail

    if b < 0:
        remaining = a * np.exp(b * (len(fitted_num_array) - j_tail)) / (1 - np.exp(b))

    if return_array:
        return fitted_num_array, remaining

    return remaining

def fix_num_array(
    num_array,
    *,
    size=(),
    fill = True,
    means = False
):
    num_array = np.asarray(num_array, dtype=float)
    if num_array.ndim != 1:
        raise ValueError("num_array must be a 1D array.")
    if len(size) == 0:
        return num_array
    
    out_length = size[0]
    if out_length <= len(num_array):
        return num_array[:out_length]
      
    if fill:
        if means:
            raise ValueError("Either choose fill or means.")
        modified = np.full(out_length, num_array[-1], dtype=float) 
        modified[:len(num_array)] = num_array
        return modified

    if means:
        avg = np.average(num_array)
        modified = np.full(out_length, avg, dtype=float)
        modified[:len(num_array)] = num_array
        return modified
    
    out = np.zeros(out_length, dtype=float)
    out[:len(num_array)] = num_array
    return out

def apply_wgts_triang(
    triangle,
    *,
    weights = None,
    fill = True,
    means = False
):
    triangle = np.asarray(triangle, dtype=float)
    if triangle.ndim != 2:
        raise ValueError("triangle must be a 2D array.")
    row_amount, col_amount = triangle.shape

    if weights is None:
        weights = np.ones(row_amount, dtype=float)
    else:
        weights =  fix_num_array(
            weights,
            size=(row_amount,),
            fill=fill,
            means=means
        )

    if any(weight < 0 for weight in weights):
        raise ValueError("Weights must be non-negative.")

    for j in range(col_amount):
        obs_rows = row_amount - j
        if obs_rows <= 0:
            continue
        temp_weights = np.flip(weights[:obs_rows])
        triangle[obs_rows:row_amount, j] = np.average(
            triangle[:obs_rows, j], 
            weights=temp_weights
        ) 
  
    return triangle

def est_ibnr_count_by_means(
    reported_incremental, 
    *,
    details = False,
    weights=None,
    fill=True,
    means=False, 
    extrapolate = True,
    threshold_ratio=0.05
):
    reported_incremental = np.asarray(reported_incremental, dtype=float)
    if reported_incremental.ndim != 2:
        raise ValueError("reported_incremental must be a 2D array.")
    row_amount, col_amount = reported_incremental.shape
    remaining_estimated_ibnr = np.zeros(row_amount, dtype=float)

    cumulative = reported_incremental.nancumsum(axis=1)

    projected_incremental = apply_wgts_triang(
        triangle=reported_incremental,
        weights=weights,
        fill=fill,
        means=means
    )

    projected_cumulative = projected_incremental.cumsum(axis=1)
    
    if extrapolate:
        for i in range(row_amount):
            remaining_estimated_ibnr[i] = fit_exp_tail(
                num_array=projected_incremental[i, :],
                threshold_ratio=threshold_ratio,
                return_array=False
            )

    latest_observed_cumulative = np.array([
        cumulative[i, col_amount - i - 1] for i in range(row_amount)
    ])
    ultimate_counts = projected_cumulative[:, -1] + remaining_estimated_ibnr
    remaining_estimated_ibnr = ultimate_counts - latest_observed_cumulative

    if details:
        projected_incremental = pd.Dataframe(projected_incremental)
        projected_incremental["Total IBNR"] = remaining_estimated_ibnr
        projected_incremental["Total"] = projected_incremental.sum(axis=1)

        return projected_incremental

    return ultimate_counts, remaining_estimated_ibnr

def cumulative_to_incremental(
    cumulative  
):
    cumulative = np.asarray(cumulative, dtype=float)
    if cumulative.ndim != 2:
        raise ValueError("cumulative must be a 2D array.")
    row_amount, col_amount = cumulative.shape
    
    incremental = np.full_like(cumulative, np.nan)
    incremental[0,:] = cumulative[0, :]
    for j in range(1, col_amount):
        incremental[:, j] = cumulative[:, j] - cumulative[:, j - 1]
    
    return incremental

def dev_factors(
    reported_incremental,
    *,
    weights = None,
    fill = True,
    means = False
):
    reported_incremental = np.asarray(reported_incremental, dtype=float)
    if reported_incremental.ndim != 2:
        raise ValueError("reported_incremental must be a 2D array.")
    row_amount, col_amount = reported_incremental.shape
    cumulative = reported_incremental.cumsum(axis=1)

    if weights is None:
        weights = np.ones(row_amount, dtype=float)
    else:
        weights =  fix_num_array(
            weights,
            size=(row_amount,),
            fill=fill,
            means=means
        )

    development_factors = np.ones(col_amount - 1, dtype=float)
    for j in range(col_amount - 1):
        obs_rows = row_amount - (j+1)

        temp_weights = np.flip(weights[:obs_rows])
        numerator = np.average(cumulative[:obs_rows, j + 1], weights=temp_weights)
        denominator = np.average(cumulative[:obs_rows, j], weights=temp_weights)
        if denominator <= 0:
            raise ValueError(f"Development factor at column {j} cannot be estimated because the denominator is not positive.")
        development_factors[j] = numerator / denominator

    return development_factors

def apply_devf_triang(
    triangle,
    *,
    development_factors
):
    triangle = np.asarray(triangle, dtype=float)
    if triangle.ndim != 2:
        raise ValueError("triangle must be a 2D array.")
    row_amount, col_amount = triangle.shape

    development_factors = np.asarray(development_factors, dtype=float)
    if development_factors.ndim != 1:
        raise ValueError("development_factors must be a 1D array.")
    if len(development_factors) != col_amount - 1:
        raise ValueError(f"development_factors must have length {col_amount - 1}.")
    if any(factor<=0 for factor in development_factors):
        raise ValueError("Development factors must be positive.")

    i_0 = np.maximum(0, row_amount-col_amount)
    for i in range(i_0, row_amount):
        last_obs_dev = col_amount - i - 1
        if last_obs_dev < 0:
            continue
        for j in range(last_obs_dev + 1, col_amount):
            triangle[i, j] = triangle[i, j - 1] * development_factors[j - 1]
  
    return triangle        

def est_ibnr_count_by_ladder(
    reported_incremental,
    *,
    details = False,
    weights = None,
    fill = True,
    means = False,
    extrapolate = False,
    threshold_ratio=0.05
):
    reported_incremental = np.asarray(reported_incremental, dtype=float)
    if reported_incremental.ndim != 2:
        raise ValueError("reported_incremental must be a 2D array.")
    row_amount, col_amount = reported_incremental.shape
    remaining_estimated_ibnr = np.zeros(row_amount, dtype=float)

    cumulative = reported_incremental.cumsum(axis=1)

    development_factors = dev_factors(
        reported_incremental=reported_incremental,
        weights=weights,
        fill=fill,
        means=means
    )

    projected_cumulative = apply_devf_triang(
        triangle=cumulative,
        development_factors=development_factors
    )

    projected_incremental = cumulative_to_incremental(projected_cumulative)
    if extrapolate:
        for i in range(row_amount):
            remaining_estimated_ibnr[i] = fit_exp_tail(
                num_array=projected_incremental[i, :],
                threshold_ratio=threshold_ratio,
                return_array=False
            )

    latest_observed_cumulative = np.array([
        cumulative[i,np.maximum(0, col_amount - i - 1)] for i in range(row_amount)
    ])
    ultimate_counts = projected_cumulative[:, -1] + remaining_estimated_ibnr
    remaining_estimated_ibnr = ultimate_counts - latest_observed_cumulative

    if details:
        projected_incremental = pd.DataFrame(projected_incremental)
        projected_incremental["Total IBNR"] = remaining_estimated_ibnr
        projected_incremental["Total"] = projected_incremental.sum(axis=1)
        
        return projected_incremental

    return ultimate_counts, remaining_estimated_ibnr
