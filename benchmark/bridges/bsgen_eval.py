"""bsgen_eval — Python port of the bsgen bootstrap-sampling-with-breeding sampler.

Line-faithful port of the reference implementation by Polyvyanyy et al.
(org.jbpt.pm.gen.bootstrap.EventLogSampling + org.jbpt.pm.log.Trace.crossover,
public jbpt codebase), which implements Algorithms 3 and 5 of:

  Polyvyanyy, Moffat, Garcia-Banuelos: Bootstrapping Generalization of
  Process Models Discovered from Event Data. CAiSE 2022: 36-54.

Provenance note: the original src/bsgen/bsgen_eval.py used for the D1 run was
itself a from-paper reconstruction (AI-assisted) and was never committed; this
port replaces it, derived directly from the authors' published Java. Validated
against the published D1 M6adapted results (bootstrap-noise agreement) before
use at scale.

Semantics preserved exactly:
  - getBreedingSites (Alg. 3): all (p1, p2) with t1[p1:p1+k] == t2[p2:p2+k];
    None when either trace is shorter than k.
  - Trace.crossover(t1, t2, site, k) [BSite overload used by logBreeding]:
    prefix = t1[0 : p1+k-1], suffix = t2[p2+k-1 :] (the site's last symbol is
    contributed by t2 only).
  - logBreeding: n = (|log1|+1)//2 pair draws; the breeding-probability draw
    happens on EVERY pair (before the sites check), as in the Java.
  - logSamplingWithBreeding (Alg. 5): generation 0 = the log; generation i =
    logBreeding(log, generation i-1); pool = union of all generations;
    result = n traces sampled with replacement from the pool.

Randomness uses Python's global `random` module so that the calling bridge's
per-miner random.seed(SEED) makes every cell reproducible (the Java reference
is unseeded; seeding is our benchmark convention).

Interface (as expected by bridges/run_m6_adapted.py and the archived bridge):
  log_sample_with_breeding(log, generations, sample_size, k, p) -> EventLog
  dedup(log) -> EventLog   (unique activity sequences, first occurrence kept)

Performance: breeding-site search is implemented with a k-gram position index
(dict lookup) instead of the Java's nested position scan. The produced SITE SET
is identical and deterministically ordered by (p1, p2); only the running time
differs (matters on D4/D5 scale).
"""
import random

from pm4py.objects.log.obj import EventLog, Trace, Event


# ── internal: work on tuples of activity names ─────────────────────────────

def _names(trace):
    return tuple(e["concept:name"] for e in trace)


def _to_trace(names):
    t = Trace()
    for a in names:
        t.append(Event({"concept:name": a}))
    return t


def _breeding_sites(t1, t2, k):
    """Algorithm 3. t1/t2 are name tuples. Returns list of (p1, p2) sorted by
    (p1, p2), or None if either trace is shorter than k (Java early exit)."""
    if len(t1) < k or len(t2) < k:
        return None
    index2 = {}
    for p2 in range(len(t2) - k + 1):
        index2.setdefault(t2[p2:p2 + k], []).append(p2)
    sites = []
    for p1 in range(len(t1) - k + 1):
        for p2 in index2.get(t1[p1:p1 + k], ()):
            sites.append((p1, p2))
    sites.sort()
    return sites


def _crossover(t1, t2, p1, p2, k):
    """Trace.crossover BSite overload: t1[0:p1+k-1] + t2[p2+k-1:]."""
    return t1[:p1 + k - 1] + t2[p2 + k - 1:]


def _log_breeding(log1, log2, k, p):
    """logBreeding. log1/log2 are lists of name tuples; returns list or None."""
    if log1 is None or log2 is None:
        return None
    if not log1 or not log2:
        return None
    result = []
    n = (len(log1) + 1) // 2
    for _ in range(n):
        t1 = random.choice(log1)
        t2 = random.choice(log2)
        sites = _breeding_sites(t1, t2, k)
        draw = random.random()  # drawn on every pair, as in the Java
        if draw < p and sites:
            p1, p2 = random.choice(sites)
            result.append(_crossover(t1, t2, p1, p2, k))
            result.append(_crossover(t2, t1, p2, p1, k))
        else:
            result.append(t1)
            result.append(t2)
    return result


# ── public interface ────────────────────────────────────────────────────────

def log_sample_with_breeding(log, generations, sample_size, k, p):
    """Algorithm 5 (logSamplingWithBreeding). `log` is a pm4py EventLog;
    returns a pm4py EventLog of `sample_size` traces sampled with replacement
    from the union of the original log and `generations` bred generations."""
    if log is None or (len(log) == 0 and sample_size > 0):
        return None
    if sample_size < 0 or k < 1 or not (0.0 <= p <= 1.0):
        return None

    base = [_names(t) for t in log]
    gens = [base]
    for _ in range(generations):
        bred = _log_breeding(base, gens[-1], k, p)
        gens.append(bred)

    pool = []
    for g in gens:
        if g is not None:
            pool.extend(g)
    if not pool or sample_size <= 0:
        return None

    out = EventLog()
    for _ in range(sample_size):
        out.append(_to_trace(random.choice(pool)))
    return out


def dedup(log):
    """Deduplicate an EventLog by activity sequence (first occurrence kept)."""
    seen = set()
    out = EventLog()
    for t in log:
        key = _names(t)
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out
