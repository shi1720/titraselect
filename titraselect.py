"""Retrospective, finite-library risk-controlled affinity screening.

No model-generated or simulated experimental observations enter the benchmark.
Run from this directory: python titraselect.py --data ../../work/data/bnab
"""
from pathlib import Path
import argparse
import itertools
import json
import time
import csv
import gzip
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression

TASKS = {
    'CR6261_H1': ('6261', 'abc', 9.0, -7.05),
    'CR6261_H9': ('6261', 'xyz', 9.0, -7.05),
    'CR9114_H1': ('9114', 'abc', 9.5, -7.05),
    'CR9114_H3': ('9114', 'def', 7.0, -6.05),
    'CR9114_FluB': ('9114', 'ghi', 7.0, -6.05),
    'CH65_MA90': ('CH65', 'ab', 9.0, 0),
    'CH65_SI06': ('CH65', 'de', 9.0, 0),
    'CH65_G189E': ('CH65', 'gh', 9.0, 0),
}


@lru_cache(maxsize=100000)
def remaining_error_bound(M, n, k, delta):
    """Exact one-sided hypergeometric inversion for a fixed accepted population.

    M accepted items in calibration + test; n randomly sampled for calibration;
    k errors among those n. Returns an upper bound on test error fraction.
    """
    if M <= n:
        return 0.0
    if n == 0:
        return 1.0
    lo, hi = k, M - n + k
    # The lower-tail probability decreases monotonically with total errors E.
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if hypergeom.logcdf(k, M, mid, n) > np.log(delta):
            lo = mid
        else:
            hi = mid - 1
    return (lo - k) / (M - n)


def contrast_scores(Y, E, c, tau, pairs):
    """Sign of the three-point Langmuir contrast; log-space spread is a scale.

    The supplied errors are NOT treated as confidence intervals. Calibration
    operates on observed cross-replicate decision errors, independently of them.
    """
    F = 10.0 ** Y
    # First-order propagation is used for ranking confidence, not coverage.
    spread = np.log(10.0) * F * E
    out = []
    h = c / (tau + c)
    for i, j in pairs:
        if c[i] > c[j]:
            i, j = j, i
        q = (h[i] - h[-1]) / (h[j] - h[-1])
        D = F[:, i] - q * F[:, j] - (1 - q) * F[:, -1]
        scale = spread[:, i] + q * spread[:, j] + (1 - q) * spread[:, -1]
        out.append(D / np.maximum(scale, 1e-12))
    return np.stack(out, axis=1)


def balanced_error(y, pred):
    values = [np.mean(pred[y == k] != k) for k in [0, 1] if np.any(y == k)]
    return float(np.mean(values))


def invariant_ratios(Y, columns):
    """Ratios of blank-subtracted geometric-mean fluorescence.

    columns[0] is the highest concentration. Negative/near-zero dynamic ranges
    are handled by a fixed value, not by reading reference labels.
    """
    F=10.**Y
    den=F[:,columns[0]]-F[:,-1]
    num=F[:,columns]-F[:,-1,None]
    ratios=np.full(num.shape,.5)
    np.divide(num,den[:,None],out=ratios,where=(den[:,None]>1e-12))
    return np.clip(ratios,0,1)


def calibrate(scores, labels, train, cal, test, alpha=0.1, delta=0.05, protect_hits=False):
    """Choose independent acceptance gates for positive and negative decisions.

    The score and grid must be fitted without calibration or test labels.
    Union bound covers both decision classes and every threshold in the grid.
    Source features of the full remaining library are deliberately transductive.
    """
    pred = (scores >= 0).astype(int)
    margin = np.abs(scores)
    accept = np.zeros(len(labels), dtype=bool)
    records = []
    quantiles = np.linspace(0, .95, 20)
    family_size = (4 if protect_hits else 2) * len(quantiles)
    for klass in [0, 1]:
        reference = margin[train][pred[train] == klass]
        grid = np.quantile(reference, quantiles) if len(reference) else np.array([np.inf])
        best = None
        for gate in grid:
            mask = (pred == klass) & (margin >= gate) & np.isfinite(scores)
            n = int(mask[cal].sum())
            k = int(((pred[cal] != labels[cal]) & mask[cal]).sum())
            test_n = int(mask[test].sum())
            M = n + test_n
            bound = remaining_error_bound(M, n, k, delta / family_size)
            hit_loss_bound=None
            if protect_hits and klass==0:
                # Errors on predicted negatives are true positive labels.
                upper_inside=int(round(bound*test_n))
                n_out=len(cal)-n
                d_out=len(test)-test_n
                out_cal=~mask[cal]
                out_failures=int((out_cal & (labels[cal]==0)).sum())
                upper_failures=remaining_error_bound(n_out+d_out,n_out,out_failures,
                                                     delta/family_size)
                lower_outside=d_out-int(round(upper_failures*d_out))
                hit_loss_bound=upper_inside/(upper_inside+lower_outside) if upper_inside else 0.
            qualifies=bound<=alpha and (hit_loss_bound is None or hit_loss_bound<=alpha)
            if test_n and qualifies:
                candidate = (test_n, -bound, float(gate), mask, n, k, bound,hit_loss_bound)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
        if best is not None:
            _, _, gate, mask, n, k, bound,hit_loss_bound = best
            accept |= mask
            records.append(dict(klass=klass, gate=gate, calibration_n=n,
                                calibration_errors=k, bound=bound,
                                hit_loss_bound=hit_loss_bound,
                                test_accepted=int(mask[test].sum())))
        else:
            records.append(dict(klass=klass, gate=None, calibration_n=0,
                                calibration_errors=0, bound=None, test_accepted=0))
    acc = accept[test]
    err = (pred[test] != labels[test])
    pos=acc & (pred[test]==1)
    neg=acc & (pred[test]==0)
    return dict(coverage=float(acc.mean()), accepted=int(acc.sum()),
                errors=int((acc & err).sum()),
                positive_accepted=int(pos.sum()),negative_accepted=int(neg.sum()),
                false_discoveries=int((pos&err).sum()),false_omissions=int((neg&err).sum()),
                test_positives=int((labels[test]==1).sum()),
                hit_retention=float(1-(neg&err).sum()/max(1,(labels[test]==1).sum())),
                hit_recall=float((pos & (labels[test]==1)).sum()/max(1,(labels[test]==1).sum())),
                selective_error=float(err[acc].mean()) if acc.any() else None,
                forced_error=float(err.mean()),
                balanced_forced_error=balanced_error(labels[test], pred[test]),
                gates=records), accept, pred


def load_task(root, task):
    ab, letters, cutoff, top = TASKS[task]
    raw, fits = [], []
    if ab == 'CH65':
        meta=pd.read_csv(root/'sample_info.tsv',sep='\t')
        grids=[]
        for letter in letters:
            if letter=='a':
                # The source authors remove spurious counts for a missing fragment.
                d=pd.read_csv(root/'Kds_CH65a_f3jrem.csv',dtype={'geno':str})
            else:
                d=pd.read_csv(root/f'Kds_CH65{letter}.tsv',sep='\t',dtype={'geno':str})
            d=d.set_index('geno')
            assert not d.index.duplicated().any()
            # Source export rotates curves by one column (cc-1): PE0 is blank.
            cols=list(range(1,10))+[0]
            vals=np.c_[d[[f'mean_log10PE{x}' for x in cols]].to_numpy(),
                       d[[f'std_log10PE{x}' for x in cols]].to_numpy()]
            raw.append(pd.DataFrame(vals,index=d.index,columns=range(1,21)))
            fits.append(pd.DataFrame({1:d.log10Kd,2:d.A,3:d.B,4:d.r2,5:d.sigma}))
            m=meta[(meta.replicate==letter)&meta.concentration_float.notna()]
            levels=m.concentration_float.drop_duplicates().to_numpy()
            assert levels[-1]==0 and len(levels)==10
            grids.append(np.r_[10.**(-levels[:-1]),0])
        np.testing.assert_allclose(grids[0],grids[1],rtol=1e-10,atol=0)
        return raw,fits,grids,cutoff
    grid_metadata=json.loads((root/'concentration_grids.json').read_text())
    grids=[]
    for letter in letters:
        name = ab + letter
        d = pd.read_csv(root / (name + '_meanbins_errors.csv'), header=None, dtype={0: str})
        f = pd.read_csv(root / (name + '_kd_meanbin_err.csv'), header=None, dtype={0: str})
        assert not d[0].duplicated().any() and not f[0].duplicated().any()
        raw.append(d.set_index(0))
        fits.append(f.set_index(0))
        grid=np.array(grid_metadata[name]['log10_concentrations'])
        grids.append(np.r_[10.**grid[:-1],0.])
    return raw, fits, grids, cutoff


def run(root, out, seeds, alpha, cal_cap=300, cutoff_offset=0.0, save_predictions=True, protect_hits=False):
    out.mkdir(parents=True, exist_ok=True)
    rows, cohort_rows, stress_rows = [], [], []
    sample_file=gzip.open(out/'heldout_predictions.csv.gz','wt',newline='') if save_predictions else None
    sample_writer=csv.DictWriter(sample_file,fieldnames=['task','source','method','genotype',
        'label','prediction','score','accepted']) if sample_file else None
    if sample_writer: sample_writer.writeheader()
    started = time.time()
    for task in TASKS:
        raw, fits, grids, cutoff = load_task(root, task)
        cutoff += cutoff_offset
        for source in range(len(raw)):
            c=grids[source]
            others = [r for r in range(len(raw)) if r != source]
            ids = raw[source].index
            # This evaluable-cohort restriction is held constant across methods.
            reference = []
            for r in others:
                good = fits[r].loc[(fits[r][4] >= .8) & (fits[r][5] <= 1), 1]
                ids = ids.intersection(good.index, sort=False)
                reference.append(good)
            d = raw[source].loc[ids]
            C=len(c)
            Y = d.loc[:, 1:C].to_numpy(float)
            E = d.loc[:, C+1:2*C].to_numpy(float)
            valid = np.isfinite(Y).all(1) & np.isfinite(E).all(1) & (E > 0).all(1)
            ids, Y, E = ids[valid], Y[valid], E[valid]
            pkd = np.stack([x.loc[ids].clip(round(-np.log10(c[0])), round(-np.log10(c[-2]))) for x in reference], axis=1)
            y = (pkd.mean(1) >= cutoff).astype(int)
            N = len(y)
            cohort_rows.append(dict(task=task, source=source, n=N,
                positive=int(y.sum()), negative=int((1-y).sum()),
                reference_disagreement=float(((pkd[:,0]>=cutoff)!=(pkd[:,1]>=cutoff)).mean()) if len(others)==2 else None,
                source_library=len(raw[source])))
            pairs = list(itertools.combinations(range(C-1), 2))
            S = contrast_scores(Y, E, c, 10.0**(-cutoff), pairs)
            fixed_pair = tuple(sorted([0, int(np.argmin(np.abs(np.log10(c[:-1]) + cutoff)))]))
            if fixed_pair[0] == fixed_pair[1]: fixed_pair = (0, 1)
            fixed_score = S[:, pairs.index(fixed_pair)]
            source_fit = fits[source].reindex(ids)
            # A failed full-curve fit yields zero confidence and is not silently dropped.
            full_score = (source_fit[1].to_numpy() - cutoff) / np.maximum(source_fit[5].to_numpy(), .05)
            full_score = np.nan_to_num(full_score, nan=0, posinf=0, neginf=0)
            for seed in seeds:
                rng = np.random.default_rng(seed)
                perm = rng.permutation(N)
                n_train = min(200, int(.2*N))
                n_cal = min(cal_cap, int(.3*N))
                train = perm[:n_train]
                cal = perm[n_train:n_train+n_cal]
                test = perm[n_train+n_cal:]
                berrors = [balanced_error(y[train], S[train, j] >= 0) for j in range(len(pairs))]
                best = int(np.argmin(berrors))
                pair = pairs[best]
                methods = {
                    'fixed_contrast_3': (fixed_score, 3, fixed_pair),
                    'selected_contrast_3': (S[:, best], 3, pair),
                    'published_fit_full': (full_score, C, ()),
                }
                classifiers = {}
                ratio=invariant_ratios(Y,list(pair))[:,1]
                iso=IsotonicRegression(out_of_bounds='clip',y_min=0,y_max=1)
                iso.fit(ratio[train],y[train])
                methods['learned_ratio_3']=(2*iso.predict(ratio)-1,3,pair)
                # An equally budgeted learned baseline and a full-panel learned reference.
                for name, cols in [('learned_panel_3', [pair[0], pair[1], C-1]),
                                   ('learned_panel_full', list(range(C))),
                                   ('learned_invariant_full',list(range(C))),
                                   ('single_signal_1', [int(np.argmin(np.abs(np.log10(c[:-1]) + cutoff)))])]:
                    clf = ExtraTreesClassifier(n_estimators=100, max_depth=8,
                        min_samples_leaf=5, random_state=seed, n_jobs=1,
                        class_weight='balanced')
                    model_X=invariant_ratios(Y,cols) if name=='learned_invariant_full' else Y[:,cols]
                    clf.fit(model_X[train], y[train])
                    if len(clf.classes_) == 2:
                        p = clf.predict_proba(model_X)[:, 1]
                    else:
                        p = np.full(N, float(clf.classes_[0]))
                    methods[name] = (2*p-1, len(cols), tuple(cols))
                    classifiers[name] = (clf, cols)
                # Constructed test-only affine changes: preserve the ideal Kd,
                # but do NOT represent a new measured biological replicate.
                srng = np.random.default_rng(seed + 9114)
                gain = 10.0**srng.normal(0,.5,N)
                offset = 10.0**Y[:,-1] * srng.uniform(0,2,N)
                F = 10.0**Y
                Fshift = gain[:,None]*F + offset[:,None]
                Yshift = np.log10(Fshift)
                Eshift = gain[:,None]*F/Fshift*E
                Sshift = contrast_scores(Yshift,Eshift,c,10.0**(-cutoff),pairs)
                for method, (scores, budget, panel) in methods.items():
                    result, accept, pred = calibrate(scores, y, train, cal, test, alpha=alpha,protect_hits=protect_hits)
                    rows.append(dict(task=task, source=source, seed=seed,
                        method=method, budget=budget, panel=list(panel),
                        cutoff=cutoff, n=N, train_n=len(train), cal_n=len(cal),
                        protect_hits=protect_hits,
                        test_n=len(test), alpha=alpha, delta=.05, **result))
                    if method != 'published_fit_full':
                        if method == 'fixed_contrast_3': shifted = Sshift[:,pairs.index(fixed_pair)]
                        elif method == 'selected_contrast_3': shifted = Sshift[:,best]
                        elif method == 'learned_ratio_3':
                            shifted=2*iso.predict(invariant_ratios(Yshift,list(pair))[:,1])-1
                        else:
                            clf, cols = classifiers[method]
                            sx=invariant_ratios(Yshift,cols) if method=='learned_invariant_full' else Yshift[:,cols]
                            if len(clf.classes_) == 2:
                                shifted = 2*clf.predict_proba(sx)[:,1]-1
                            else: shifted = np.full(N,2*float(clf.classes_[0])-1)
                        spred=(shifted>=0).astype(int)
                        saccept=np.zeros(N,dtype=bool)
                        for g in result['gates']:
                            if g['gate'] is not None:
                                saccept |= (spred==g['klass']) & (np.abs(shifted)>=g['gate'])
                        at=saccept[test];er=spred[test]!=y[test]
                        stress_rows.append(dict(task=task,source=source,seed=seed,method=method,
                            coverage=float(at.mean()),accepted=int(at.sum()),
                            errors=int((at&er).sum()),
                            selective_error=float(er[at].mean()) if at.any() else None,
                            decision_flip=float((spred[test]!=pred[test]).mean()),
                            acceptance_flip=float((saccept[test]!=accept[test]).mean())))
                    if save_predictions and seed == seeds[0]:
                        for k in test:
                            sample_writer.writerow(dict(task=task,source=source,method=method,
                                genotype=ids[k],label=int(y[k]),prediction=int(pred[k]),
                                score=float(scores[k]),accepted=bool(accept[k])))
                print(task, source, seed, 'n=', N, 'pair=', pair, flush=True)
    pd.DataFrame(cohort_rows).to_csv(out/'cohorts.csv',index=False)
    pd.DataFrame(stress_rows).to_csv(out/'affine_stress.csv',index=False)
    if sample_file: sample_file.close()
    (out/'runs.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    flat = [{k:v for k,v in x.items() if k not in ['gates','panel']} for x in rows]
    pd.DataFrame(flat).to_csv(out/'runs.csv',index=False)
    (out/'execution.json').write_text(json.dumps(dict(seconds=time.time()-started,seeds=seeds,
        alpha=alpha,cal_cap=cal_cap,cutoff_offset=cutoff_offset),indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('results'))
    p.add_argument('--seeds', type=int, nargs='+', default=[1720,1721,1722,1723,1724])
    p.add_argument('--alpha',type=float,default=.1)
    p.add_argument('--cal-cap',type=int,default=300)
    p.add_argument('--cutoff-offset',type=float,default=0.)
    p.add_argument('--no-predictions',action='store_true')
    p.add_argument('--protect-hits',action='store_true')
    args=p.parse_args()
    run(args.data,args.out,args.seeds,args.alpha,args.cal_cap,args.cutoff_offset,not args.no_predictions,args.protect_hits)
