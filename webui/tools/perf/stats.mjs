// Statistics of the browser measurements (browser.mjs; tests in stats.test.mjs).

export function quantile(values, q) {
  if (!values.length) return null
  const sorted = [...values].sort((a, b) => a - b)
  const pos = (sorted.length - 1) * q
  const lo = Math.floor(pos)
  const hi = Math.ceil(pos)
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo)
}
export const mean = (values) => (values.length ? values.reduce((a, b) => a + b, 0) / values.length : null)
export const round = (value, digits = 1) => (value == null ? null : Number(value.toFixed(digits)))
export const summary = (values) => ({ n: values.length, mean: round(mean(values)), p50: round(quantile(values, 0.5)), p95: round(quantile(values, 0.95)), p99: round(quantile(values, 0.99)), max: round(values.length ? Math.max(...values) : null) })

// two-sided 95 % Student t quantiles by degrees of freedom
export const T95 = [12.71, 4.3, 3.18, 2.78, 2.57, 2.45, 2.36, 2.31, 2.26, 2.23, 2.2, 2.18, 2.16, 2.14, 2.13, 2.12, 2.11, 2.1, 2.09, 2.09, 2.08, 2.07, 2.07, 2.06, 2.06, 2.06, 2.05, 2.05, 2.05, 2.04]
// one-sided 95 % Student t quantiles by degrees of freedom (the upper bound of the overhead)
export const T95_ONE_SIDED = [6.31, 2.92, 2.35, 2.13, 2.02, 1.94, 1.89, 1.86, 1.83, 1.81, 1.8, 1.78, 1.77, 1.76, 1.75, 1.75, 1.74, 1.73, 1.73, 1.72, 1.72, 1.72, 1.71, 1.71, 1.71, 1.71, 1.7, 1.7, 1.7, 1.7]

/** Student t quantile for `df` degrees of freedom: the table, else the Cornish-Fisher expansion around `z`. */
export function tQuantile(table, z, df) {
  if (df >= 1 && df <= table.length) return table[df - 1]
  return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)
}

/**
 * Mean of per-pair differences with its two-sided 95 % interval and the one-sided 95 % bounds
 * (`upper95`: the overhead is below it; `lower95`: above it).
 */
export function pairStats(values) {
  const n = values.length
  if (!n) return { n, mean: null, ci95: null, upper95: null, lower95: null, median: null }
  const m = mean(values)
  const se = n > 1 ? Math.sqrt(values.reduce((total, v) => total + (v - m) ** 2, 0) / (n - 1)) / Math.sqrt(n) : null
  const oneSided = se == null ? null : tQuantile(T95_ONE_SIDED, 1.645, n - 1) * se
  return {
    n,
    mean: round(m, 2),
    ci95: se == null ? null : round(tQuantile(T95, 1.96, n - 1) * se, 2),
    upper95: oneSided == null ? null : round(m + oneSided, 2),
    lower95: oneSided == null ? null : round(m - oneSided, 2),
    median: round(quantile(values, 0.5), 2),
  }
}

/**
 * Whether an adaptive `interleaved` run (`--repeat` jobs at least, `--max-repeat` at most) has
 * settled the overhead budget after `jobs` jobs, from the pooled window pairs' `stats`:
 * "within" once the one-sided upper 95 % bound is at or under `target` %, "over" once the
 * one-sided lower bound is above it; otherwise another job, until the cap ("inconclusive").
 */
export function repeatDecision({ jobs, minJobs, maxJobs, target, stats }) {
  const settled = jobs >= minJobs && stats.n >= 2
  if (settled && stats.upper95 <= target) return { stop: true, verdict: "within", reason: `upper 95 % bound ${stats.upper95} % <= ${target} %` }
  if (settled && stats.lower95 > target) return { stop: true, verdict: "over", reason: `lower 95 % bound ${stats.lower95} % > ${target} %` }
  if (jobs >= Math.max(minJobs, maxJobs)) return { stop: true, verdict: "inconclusive", reason: `${jobs} jobs (the cap), upper 95 % bound ${stats.upper95} %` }
  return { stop: false, verdict: null, reason: null }
}

/**
 * Window pairs needed for a one-sided upper 95 % bound at `target` % when the pairs scatter by
 * `sd` % around `expected` % (normal approximation; null when `expected` is not below `target`).
 */
export function pairsNeeded({ sd, expected = 0, target = 1 }) {
  if (expected >= target) return null
  return Math.ceil(((1.645 * sd) / (target - expected)) ** 2)
}
