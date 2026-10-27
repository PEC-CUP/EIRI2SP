"""Real diesel samples: paths, measured properties, RI calibration, TIC slicing.

Four samples (5#, 9#, 33#, 36#) from ``data/exp/data-wt``:

  data_list/<id>_gcms.xlsx   exported GC-MS TIC  (Time / Intensity, 0-68.3 min)
  data_list/<id>_gcfid.xlsx  exported GC-FID trace (retained as an independent
                             diagnostic only; not used in the main route)
  data_list/property.xlsx    density, MW, C wt%, H wt%, H/C  (measured)
  data_list/distillation curve.xlsx   one sheet per sample, yield% vs bp degC
  GC-MS/<id>_30m.raw         the raw run; converted to mzML/<id>_30m.mzML with
                             ``msconvert ... --ignoreUnknownInstrumentError
                             --filter "peakPicking true 1-"``

The RI anchors are the **GC-MS** retention times of the n-alkanes C9-C28 given by
the operator (33#: 8.67 ... 43.69 min); ``ri_from_rt`` is a van den Dool-Kratz
interpolation with a linear end extrapolation. Slice areas come from the GC-MS
TIC integrated to a linear baseline. Unit response is used for all compound
classes when these TIC areas are converted to mass weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
DW = BASE / "data" / "exp" / "data-wt"
DL = DW / "data_list"
MZML = DW / "mzML"
OUT = DW / "results"

# All four measured distillates are processed through the same pipeline.  The 3-ring
# naphthene share is high for 36#, but the reported reference groups DBE 1-3 into one
# naphthene class, so 36# remains comparable at the three-class level used here.
SAMPLES = ("5", "9", "33", "36")

# GC-MS retention times of the n-alkanes (min) -- 33# provided by the operator,
# identical run program for all four samples
ALKANE_MS_RT = {9: 8.67, 10: 11.96, 11: 14.84, 12: 17.43, 13: 19.81, 14: 22.02,
                15: 24.11, 16: 26.06, 17: 27.93, 18: 29.69, 19: 31.37, 20: 32.98,
                21: 34.52, 22: 35.99, 23: 37.40, 24: 38.75, 25: 40.06, 26: 41.31,
                27: 42.52, 28: 43.69}

# effective data range: the first 2 min are a contamination peak and are cut;
# the run is clean up to 50 min (post-run bake afterwards)
RT_LO, RT_HI = 2.0, 50.0
# Slicing grid: the virtual complex mixtures cut the chromatogram on a UNIFORM RI grid
# (5 RI step), and the fingerprint is a function of (spectrum, RI) -- so the real
# samples use the same grid.  A uniform RT grid (0.1 min) would give 3-4 RI slices at
# the light end and 8-9 RI at the heavy end, i.e. an asymmetric picture on the RI axis.
SLICE_WIDTH = 5.0          # in RI units
RT_SLICE_WIDTH = 0.1       # kept for reference / reports
BG_PRE = (2.1, 3.4)        # clean window right after the cut -- TIC baseline start
BG_POST = (50.0, 65.0)     # post-run: column bleed only -- TIC baseline end
BL_RT = (3.0, 50.0)        # the baseline ramp spans these two retention times
RI_TRAIN_LO, RI_TRAIN_HI = 860.0, 3100.0   # RI range the translator was trained on


def ri_from_rt(rt):
    """van den Dool-Kratz RI of the GC-MS retention time(s) in minutes."""
    cs = np.array(sorted(ALKANE_MS_RT))
    ts = np.array([ALKANE_MS_RT[c] for c in cs], float)
    rt = np.asarray(rt, float)
    i = np.clip(np.searchsorted(ts, rt), 1, len(cs) - 1)
    t0, t1 = ts[i - 1], ts[i]
    ri = 100.0 * (cs[i - 1] + (cs[i] - cs[i - 1]) * (rt - t0) / (t1 - t0))
    lo = rt < ts[0]                      # linear extension below C9 ...
    slope = 100.0 * (cs[1] - cs[0]) / (ts[1] - ts[0])
    ri = np.where(lo, 100.0 * cs[0] + slope * (rt - ts[0]), ri)
    hi = rt > ts[-1]                     # ... and above C28
    slope = 100.0 * (cs[-1] - cs[-2]) / (ts[-1] - ts[-2])
    ri = np.where(hi, 100.0 * cs[-1] + slope * (rt - ts[-1]), ri)
    return ri


def load_properties() -> pd.DataFrame:
    d = pd.read_excel(DL / "property.xlsx")
    d.columns = [c.strip() for c in d.columns]
    d["sample"] = d["sample"].astype(str).str.rstrip("#")
    return d.set_index("sample")


def load_distillation(sample: str) -> pd.DataFrame:
    key = "%s#" % sample
    d = pd.read_excel(DL / "distillation curve.xlsx", sheet_name=key, header=0)
    d.columns = ["yield_pct", "bp_C"]
    return d.apply(pd.to_numeric, errors="coerce").dropna()


def load_chromatogram(sample: str, kind: str = "gcms") -> pd.DataFrame:
    d = pd.read_excel(DL / ("%s_%s.xlsx" % (sample, kind)), header=3)
    d.columns = ["t", "i"]
    return d.apply(pd.to_numeric, errors="coerce").dropna()


def parse_and_bin(mzml: Path, mz_lo: int = 50, mz_hi: int = 500):
    """(rts, tics, absolute m/z matrix) of one mzML file."""
    sys.path.insert(0, str(BASE / "src" / "dataset_construction"))
    from parse_mzml import parse
    rts, tics, spectra = parse(str(mzml))
    n = mz_hi - mz_lo + 1
    m = np.zeros((len(spectra), n), dtype=np.float64)
    for i, s in enumerate(spectra):
        idx = np.round(s["mz"]).astype(int)
        keep = (idx >= mz_lo) & (idx <= mz_hi)
        np.add.at(m[i], idx[keep] - mz_lo, s["intensity"][keep].astype(np.float64))
    return rts, tics, m


def slice_sample(sample: str, rt_lo: float = RT_LO, rt_hi: float = RT_HI,
                 width: float = SLICE_WIDTH, mz_lo: int = 50, mz_hi: int = 500,
                 verbose: bool = True) -> pd.DataFrame:
    """Slice one sample to baseline; areas come from the GC-MS TIC.

    Returns a frame with columns ``rt, ri, area`` plus ``mz_50 ... mz_500``
    (the TIC-normalised slice spectrum, sum = 1).
    """
    mzml = MZML / ("%s_30m.mzML" % sample)
    rts, tics, absm = parse_and_bin(mzml, mz_lo, mz_hi)

    post = (rts >= BG_POST[0]) & (rts <= BG_POST[1])
    bg_spec = np.median(absm[post], axis=0)
    net_abs = np.clip(absm - bg_spec[None, :], 0.0, None)

    pre = (rts >= BG_PRE[0]) & (rts <= BG_PRE[1])
    bl_pre = float(np.percentile(tics[pre], 10))
    bl_post = float(np.percentile(tics[post], 10))

    def baseline(rt):
        f = np.clip((rt - BL_RT[0]) / (BL_RT[1] - BL_RT[0]), 0.0, 1.0)
        return bl_pre + (bl_post - bl_pre) * f

    ri_tr = ri_from_rt(rts)
    net_tic = np.clip(tics - baseline(rts), 0.0, None)
    grid_lo = float(np.ceil(ri_tr[(rts >= rt_lo)].min() / width) * width)
    grid_hi = float(np.floor(ri_tr[(rts <= rt_hi)].max() / width) * width)
    rows = []
    edges = np.arange(grid_lo, grid_hi + 1e-9, width)
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        m = (ri_tr >= lo) & (ri_tr < hi) & (rts >= rt_lo) & (rts <= rt_hi)
        if not m.any():
            continue
        area = float(net_tic[m].sum())
        spec = net_abs[m].sum(axis=0)
        tot = spec.sum()
        if area <= 0 or tot <= 0:
            continue
        rt_center = float(np.average(rts[m], weights=np.clip(net_tic[m], 1e-12, None)))
        rows.append({"rt": rt_center, "ri": 0.5 * (lo + hi), "area": area,
                     "spec": spec / tot})
    df = pd.DataFrame([{k: r[k] for k in ("rt", "area")} for r in rows])
    df["ri"] = [r["ri"] for r in rows]
    ei = np.vstack([r["spec"] for r in rows])
    spec_df = pd.DataFrame(ei, columns=[f"mz_{i}" for i in range(mz_lo, mz_hi + 1)])
    df = pd.concat([df, spec_df], axis=1)
    if verbose:
        net_all = np.clip(tics - baseline(rts), 0.0, None)
        tot_net = float(net_all.sum())
        inside = float(net_all[(rts >= rt_lo) & (rts <= rt_hi)].sum())
        before20 = float(net_all[rts < 20.0].sum())
        print("  %-3s slices %3d  RT %.2f-%.2f  RI %.0f-%.0f  window covers %.1f%% "
              "of the net TIC (%.1f%% of it elutes before 20 min)"
              % (sample, len(df), df.rt.min(), df.rt.max(), df.ri.min(),
                 df.ri.max(), 100 * inside / tot_net,
                 100 * before20 / inside if inside else float("nan")))
        tot_area = float(df.area.sum())
        below = float(df.area[df.ri < RI_TRAIN_LO].sum() / tot_area)
        above = float(df.area[df.ri > RI_TRAIN_HI].sum() / tot_area)
        print("      outside the training RI window %.0f-%.0f: %.1f %% below "
              "(RI from the van den Dool-Kratz extension), %.1f %% above"
              % (RI_TRAIN_LO, RI_TRAIN_HI, 100 * below, 100 * above))
    return df


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    props = load_properties()
    print("measured properties:\n", props.to_string())
    for s in SAMPLES:
        d = load_distillation(s)
        print("  %-3s distillation %d points, %.1f-%.1f degC"
              % (s, len(d), d.bp_C.min(), d.bp_C.max()))
    print("\nslicing the four samples (raw GC-MS TIC areas):")
    for s in SAMPLES:
        df = slice_sample(s)
        df.to_csv(OUT / ("slices_%s.csv" % s), index=False)
