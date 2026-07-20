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
        if return_array:
            return num_array, remaining
        return remaining

    j_tail = candidates[0]

    tail = num_array[j_tail:]
    x = np.arange(len(tail))

    positive_mask = tail > 0
    if positive_mask.sum() < 2:
        if return_array:
            return num_array, remaining
        return remaining

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
    fill_with_last = False,
    fill_with_mean = False
):
    num_array = np.asarray(num_array, dtype=float)
    if num_array.ndim != 1:
        raise ValueError("num_array must be a 1D array.")
    if len(size) == 0:
        return num_array
    
    out_length = size[0]
    if out_length <= len(num_array):
        return num_array[:out_length]
      
    if fill_with_last:
        if fill_with_mean:
            raise ValueError("You cannot choose to fill missing entries with the last element and the mean at the same time.")
        modified = np.full(out_length, num_array[-1], dtype=float) 
        modified[:len(num_array)] = num_array
        return modified

    if fill_with_mean:
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
    fill_with_last = False,
    fill_with_mean = False
):
    triangle = np.asarray(triangle, dtype=float, copy=True)
    if triangle.ndim != 2:
        raise ValueError("triangle must be a 2D array.")
    row_amount, col_amount = triangle.shape

    if weights is None:
        weights = np.ones(row_amount, dtype=float)
    else:
        weights =  fix_num_array(
            weights,
            size=(row_amount,),
            fill_with_last=fill_with_last,
            fill_with_mean=fill_with_mean
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
    fill_with_last=False,
    fill_with_mean=False, 
    extrapolate = False,
    threshold_ratio=0.05
):
    reported_incremental = np.asarray(reported_incremental, dtype=float)
    if reported_incremental.ndim != 2:
        raise ValueError("reported_incremental must be a 2D array.")
    row_amount, col_amount = reported_incremental.shape
    
    remaining_estimated_ibnr = np.zeros(row_amount, dtype=float)
    cumulative = reported_incremental.cumsum(axis=1)

    projected_incremental = apply_wgts_triang(
        triangle=reported_incremental,
        weights=weights,
        fill_with_last=fill_with_last,
        fill_with_mean=fill_with_mean
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

    if details:
        remaining_estimated_ibnr = ultimate_counts - latest_observed_cumulative
        projected_incremental = pd.DataFrame(projected_incremental)
        projected_incremental["Total IBNR"] = remaining_estimated_ibnr
        projected_incremental["Total"] = projected_incremental.sum(axis=1)

        return projected_incremental

    return ultimate_counts

def cumulative_to_incremental(
    cumulative  
):
    cumulative = np.asarray(cumulative, dtype=float)
    if cumulative.ndim != 2:
        raise ValueError("cumulative must be a 2D array.")
    row_amount, col_amount = cumulative.shape
    
    incremental = np.full_like(cumulative, np.nan)
    incremental[:,0] = cumulative[:, 0]
    for j in range(1, col_amount):
        incremental[:, j] = cumulative[:, j] - cumulative[:, j - 1]
    
    return incremental

def dev_factors(
    triangle,
    *,
    weights = None,
    fill_with_last = False,
    fill_with_mean = False
):
    triangle = np.asarray(triangle, dtype=float, copy=True)
    if triangle.ndim != 2:
        raise ValueError("reported_incremental must be a 2D array.")
    row_amount, col_amount = triangle.shape
    
    if weights is None:
        weights = np.ones(row_amount, dtype=float)
    else:
        weights =  fix_num_array(
            weights,
            size=(row_amount,),
            fill_with_last=fill_with_last,
            fill_with_mean=fill_with_mean
        )

    development_factors = np.ones(col_amount - 1, dtype=float)
    for j in range(col_amount - 1):
        obs_rows = row_amount - (j+1)

        temp_weights = np.flip(weights[:obs_rows])
        numerator = np.average(triangle[:obs_rows, j + 1], weights=temp_weights)
        denominator = np.average(triangle[:obs_rows, j], weights=temp_weights)
        if denominator <= 0:
            raise ValueError(f"Development factor at column {j} cannot be estimated because the denominator is not positive.")
        development_factors[j] = numerator / denominator

    return development_factors

def apply_devf_triang(
    triangle,
    *,
    development_factors
):
    triangle = np.asarray(triangle, dtype=float, copy=True)
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
    fill_with_last = False,
    fill_with_mean = False,
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
        triangle=cumulative,
        weights=weights,
        fill_with_last=fill_with_last,
        fill_with_mean=fill_with_mean
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

    if details:
        remaining_estimated_ibnr = ultimate_counts - latest_observed_cumulative
        projected_incremental = pd.DataFrame(projected_incremental)
        projected_incremental["Total IBNR"] = remaining_estimated_ibnr
        projected_incremental["Total"] = projected_incremental.sum(axis=1)
        
        return projected_incremental

    return ultimate_counts

def monte_carlo_ibnr_count_exp(
    num_scenarios=10_000,
    *,
    ladder_method=True,
    mean_method=True,
    weight_kwargs=None,
    occurrence_periods=10,
    reporting_pattern=(0.6,),
    lambda_sampler=lognormal_lambda_sampler,
    sampler_kwargs=None,
    seed=123
):
    weight_kwargs = weight_kwargs or {None: None}
    sampler_kwargs = sampler_kwargs or {"mean": 6.3, "sigma": 0.1}
    rng = np.random.default_rng(seed)

    estimators = {}
    if ladder_method:
        for key in weight_kwargs.keys():
            if key is None:
                estimators[f"ladder_equal"] = lambda tri: est_ibnr_count_by_ladder(tri)
            else:
                estimators[f"ladder_{key}"] = lambda tri: est_ibnr_count_by_ladder(tri, weights=weight_kwargs[key])
    if mean_method:
        for key in weight_kwargs.keys():
            if key is None:
                estimators[f"mean_equal"] = lambda tri: est_ibnr_count_by_means(tri)
            else:
                estimators[f"mean_{key}"] = lambda tri: est_ibnr_count_by_means(tri, weights=weight_kwargs[key])

    rows = []

    for scenario in range(num_scenarios):
        ultimate_counts, reported_incremental = reported_claims_model(
            rng=rng,
            occurrence_periods=occurrence_periods,
            reporting_pattern=reporting_pattern,
            lambda_sampler=lambda_sampler,
            sampler_kwargs=sampler_kwargs
        )

        observed_cumulative = np.nancumsum(reported_incremental, axis=1)
        latest_observed = np.array([
            observed_cumulative[i, occurrence_periods - i - 1]
            for i in range(occurrence_periods)
        ])
        true_ibnr = ultimate_counts - latest_observed

        for method, estimator in estimators.items():
            try:
                est_ultimate = estimator(reported_incremental)
                ibnr_error = (est_ultimate - latest_observed) - true_ibnr

                rows.append({
                    "scenario": scenario,
                    "method": method,
                    "true_claim_total": ultimate_counts.sum(),
                    "est_claim_total": est_ultimate.sum(),
                    "true_ibnr_total": true_ibnr.sum(),
                    "est_ibnr_total": (est_ultimate - latest_observed).sum(),
                    "%_ibnr_error": (ibnr_error.sum())/(true_ibnr.sum())*100,
                    "ibnr_error": ibnr_error.sum(),
                    "sum_abs_ibnr_error": abs(ibnr_error).sum(),
                    "std_ibnr_error": (ibnr_error**2).sum()
                })
            except Exception:
                rows.append({
                    "scenario": scenario,
                    "method": method,
                    "true_claim_total": ultimate_counts.sum(),
                    "est_claim_total": np.nan,
                    "true_ibnr_total": true_ibnr.sum(),
                    "est_ibnr_total": np.nan,
                    "%_ibnr_error": np.nan,
                    "ibnr_error": np.nan,
                    "sum_abs_ibnr_error": np.nan,
                    "sum_sq_ibnr_error": np.nan
                })

    results = pd.DataFrame(rows)

    #summary = results.groupby("method", dropna=False).agg(
    #    mean_true_ibnr=("true_ibnr_total", "mean"),
    #    mean_est_ibnr=("est_ibnr_total", "mean"),
    #    bias=("ibnr_error", "mean"),
    #    mae=("abs_ibnr_error", "mean"),
    #    rmse=("sq_ibnr_error", lambda x: np.sqrt(np.nanmean(x))),
    #    std_error=("ibnr_error", "std"),
    #    failure_rate=("est_ibnr_total", lambda x: x.isna().mean())
    #).reset_index()

    return results#, summary
