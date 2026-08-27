"""CICIDS2017 loader for PerMFL.

Returns the four-tuple every loader in utils/model_utils.py returns:
    clients, groups, train_data, test_data

Reads TrafficLabelling/, NOT MachineLearningCVE/. The latter carries no
Timestamp and no Flow ID, so neither a temporal split nor conversation-level
grouping is possible on it.

Configuration is by environment variable so no code edit is needed per run:

  CICIDS_DIR                directory holding the eight CSVs
  CICIDS_PARTITION          domain (default) | dirichlet
  CICIDS_SPLIT              temporal (default) | random
  CICIDS_MIN_SAMPLES        drop classes below this count      (default 2000)
  CICIDS_MAX_PER_CLIENT     cap on rows per client             (default 20000)
  CICIDS_RARE_FLOOR         rows of a rare class protected from the cap (2000)
  CICIDS_RARE_BELOW         classes rarer than this are concentrated    (20000)
  CICIDS_CLIENTS_PER_RARE   how many clients receive a rare class       (3)
  CICIDS_DIRICHLET_ALPHA    concentration for the dirichlet partition   (0.5)
  CICIDS_TEAM_SEED          seed for group_division 1                   (0)

Measured facts that drive the defaults, all on the full capture:

  * 309,745 rows carry Inf or NaN in the flow-rate columns, or are exact
    duplicates. Replacing Inf with zero rather than dropping the row
    manufactures a large cluster of identical rows spanning several classes.
  * Macro F1 for a centralised model, cleaned data, random split:
        classes  linear   MLP(100)
        15       0.3106   0.6464
        12       0.3931   0.7598
         9       0.5186   0.9331
    Classes below roughly 2000 rows are unlearnable and each contributes 0.
    A single linear layer caps near 0.52 regardless, so --model_name dnn is
    required here; mclr exists to instantiate the paper's convex case.
"""
import os, glob
import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("CICIDS_DIR",
    "/Users/kaushik/Projects/Masters/Dissertation/datasets/OneDrive_1_03-08-2026/TrafficLabelling")
PARTITION       = os.environ.get("CICIDS_PARTITION", "domain")
SPLIT           = os.environ.get("CICIDS_SPLIT", "temporal")
MIN_SAMPLES     = int(os.environ.get("CICIDS_MIN_SAMPLES", "2000"))
MAX_PER_CLIENT  = int(os.environ.get("CICIDS_MAX_PER_CLIENT", "20000"))
RARE_FLOOR      = int(os.environ.get("CICIDS_RARE_FLOOR", "2000"))
RARE_BELOW      = int(os.environ.get("CICIDS_RARE_BELOW", "20000"))
CLIENTS_PER_RARE= int(os.environ.get("CICIDS_CLIENTS_PER_RARE", "3"))
DIR_ALPHA       = float(os.environ.get("CICIDS_DIRICHLET_ALPHA", "0.5"))
MIN_CLIENT_ROWS = int(os.environ.get("CICIDS_MIN_CLIENT_ROWS", "200"))
# Attack classes per client for CICIDS_PARTITION=labelskew. Follows the
# Label1/Label3/Label6 ladder of Stones From Other Hills (IEEE IoT J 2025),
# which sweeps how many classes each client holds and reports FedAvg, Local
# and PFL at every level. One class per client is maximal heterogeneity and
# trivially favours local training; the ladder shows where federation starts
# to pay.
CLASSES_PER_CLIENT = int(os.environ.get("CICIDS_CLASSES_PER_CLIENT", "3"))
# CICIDS_CROSS_TEST=1 distributes the held-out test rows across ALL clients
# regardless of which classes each client trained on, so every client is
# evaluated on all classes. Follows Stones From Other Hills (IEEE IoT J 2025)
# Table II, where gateway 0 trains on classes 3-8 and is tested on 0,1,2,3,4,8,
# explicitly to measure adaptation to unseen / zero-day traffic. Local training
# cannot succeed here by construction; federation can.
CROSS_TEST = int(os.environ.get("CICIDS_CROSS_TEST", "0"))
# Percentile clipping, standard in published CICIDS2017 preprocessing and the
# one leakage-free step our pipeline was missing. Measured on the cleaned
# matrix: 58 of 79 features have max/median > 1e4 and 43 exceed 1e6, three at
# int64 max. Standardising against outliers of that scale compresses the
# useful range of nearly every feature toward zero. Bounds are fitted on TRAIN
# ONLY, so no test statistic crosses the split.
CLIP_PCT = float(os.environ.get("CICIDS_CLIP_PCT", "0"))
# Graded domain structure, after SCMoE's pfl/data/domain_partitioner.py.
# Pure domain partitioning is the alpha->0 case: zero overlap between groups.
# Dirichlet gives graded skew but, as that module's docstring argues, produces
# a continuum rather than discrete groups, so no clustering can recover
# anything and a null measures the partition rather than the algorithm.
# Mixing keeps recoverable groups AND grades the skew:
#     dist = MIX * domain_dist + (1 - MIX) * uniform
# MIX=1 pure domains, MIX=0 IID.
DOMAIN_MIX = float(os.environ.get("CICIDS_DOMAIN_MIX", "1.0"))
# Keep this fraction of each client's BENIGN rows, all attack rows. Thinned
# per client, never globally, so no client is left benign-free -- a detector
# with no notion of normal has no decision boundary. At 82% BENIGN the
# majority class otherwise dominates gradient cosine similarity and
# degenerates clustering.
BENIGN_FRACTION = float(os.environ.get("CICIDS_BENIGN_FRACTION", "1.0"))

DAY_FILES = {
    "Monday":    ["Monday-WorkingHours"],
    "Tuesday":   ["Tuesday-WorkingHours"],
    "Wednesday": ["Wednesday-workingHours"],
    "Thursday":  ["Thursday-WorkingHours-Morning-WebAttacks",
                  "Thursday-WorkingHours-Afternoon-Infilteration"],
    "Friday":    ["Friday-WorkingHours-Morning",
                  "Friday-WorkingHours-Afternoon-PortScan",
                  "Friday-WorkingHours-Afternoon-DDos"],
}
DOMAIN_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
IDENTIFIERS = {"flow id", "source ip", "source port", "destination ip"}

CLASS_NAMES = []          # filled by the loader, after thresholding


def _fix_timestamps(s):
    """Seven of eight files use 12-hour clocks with no meridiem, so Wednesday
    appears to run 8:42 to 2:43. The capture window is 08:00-17:00, so hours
    1 to 7 are afternoon. Monday is already 24-hour with seconds."""
    dt = pd.to_datetime(s, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    todo = dt.isna()
    if todo.any():
        alt = pd.to_datetime(s[todo], format="%d/%m/%Y %H:%M", errors="coerce")
        bump = alt.dt.hour.between(1, 7).fillna(False)
        alt = alt + pd.to_timedelta(np.where(bump, 12, 0), unit="h")
        dt.loc[todo] = alt
    return dt


CACHE = os.environ.get("CICIDS_CACHE", "data/cicids_clean.npz")


def _load_all(verbose):
    """Parse, clean and cache. The cleaning is config-independent, so the
    cache is shared across every partition and split setting."""
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        if verbose:
            print(f"  cache hit: {CACHE} ({len(z['lab']):,} rows)")
        return z["lab"], z["X"], z["day"], z["t"], list(z["feat"])
    frames = []
    for day in DOMAIN_ORDER:
        for frag in DAY_FILES[day]:
            hits = glob.glob(os.path.join(DATA_DIR, frag + "*.csv"))
            if not hits:
                raise FileNotFoundError(f"{frag}*.csv not under {DATA_DIR}")
            d = pd.read_csv(hits[0], encoding="latin-1", low_memory=False,
                            skipinitialspace=True)
            d.columns = [str(c).strip() for c in d.columns]
            d = d.dropna(how="all")            # 288,602 blank rows in one file
            d["_day"] = DOMAIN_ORDER.index(day)
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    df["Label"] = (df["Label"].astype(str)
                   .str.encode("ascii", "ignore").str.decode("ascii")
                   .str.replace(r"\s+", " ", regex=True).str.strip())
    df = df[df["Label"] != ""]
    df["_t"] = _fix_timestamps(df["Timestamp"].astype(str).str.strip())
    df = df[df["_t"].notna()]

    feat = [c for c in df.columns
            if c.strip().lower() not in IDENTIFIERS
            and c not in ("Timestamp", "Label", "_day", "_t")]
    X = df[feat].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    keep = X.notna().all(axis=1)
    before = len(df)
    df, X = df[keep], X[keep]
    dup = pd.concat([X, df["Label"]], axis=1).duplicated()
    df, X = df[~dup.values], X[~dup.values]
    if verbose:
        print(f"  cleaned: {len(df):,} rows "
              f"(dropped {before - len(df):,} Inf/NaN and duplicates)")
    lab = df["Label"].to_numpy()
    Xa = X.to_numpy(np.float32)
    day = df["_day"].to_numpy(np.int64)
    t = df["_t"].to_numpy("datetime64[s]").astype(np.int64)
    os.makedirs(os.path.dirname(CACHE) or ".", exist_ok=True)
    np.savez_compressed(CACHE, lab=lab, X=Xa, day=day, t=t, feat=np.array(feat, object))
    if verbose:
        print(f"  cached -> {CACHE}")
    return lab, Xa, day, t, feat


def read_cicids_data(NUM_USERS, NUM_LABELS, NUM_GROUPS, group_division, verbose=True):
    global CLASS_NAMES
    rng = np.random.RandomState(int(os.environ.get("PERMFL_SEED", "0")))
    lab, X, day, t, feat = _load_all(verbose)

    vc = pd.Series(lab).value_counts()
    kept = sorted(vc[vc >= MIN_SAMPLES].index)
    dropped = [(k, int(v)) for k, v in vc.items() if k not in set(kept)]
    if verbose and dropped:
        print(f"  dropped {len(dropped)} class(es) under {MIN_SAMPLES}: "
              + ", ".join(f"{n}({c})" for n, c in dropped))
    keepset = set(kept)
    mask = np.array([l in keepset for l in lab])
    lab, X, day, t = lab[mask], X[mask], day[mask], t[mask]
    CLASS_NAMES = list(kept)
    C = len(CLASS_NAMES)
    lut = {n: i for i, n in enumerate(CLASS_NAMES)}
    y = np.array([lut[l] for l in lab], dtype=np.int64)
    if verbose:
        print(f"  {C} classes: {CLASS_NAMES}")

    # ---------------- assign rows to clients ----------------
    owner = np.full(len(y), -1, dtype=np.int64)
    cls_counts = np.bincount(y, minlength=C)

    if PARTITION == "domain":
        per_dom = NUM_USERS // len(DOMAIN_ORDER)
        client_domain = [d for d in range(len(DOMAIN_ORDER)) for _ in range(per_dom)]
        for d in range(len(DOMAIN_ORDER)):
            members = [i for i, dd in enumerate(client_domain) if dd == d]
            rows = np.where(day == d)[0]
            for c in range(C):
                rc = rows[y[rows] == c]
                if len(rc) == 0:
                    continue
                # Rare classes go to a FEW clients, not spread thin. Splitting
                # a 2000-row class over 20 clients leaves 100 each, too few to
                # learn; concentrating keeps it learnable for those holding it.
                if cls_counts[c] < RARE_BELOW:
                    take = members[:min(CLIENTS_PER_RARE, len(members))]
                else:
                    take = members
                owner[rc] = np.array(take)[np.arange(len(rc)) % len(take)]
    elif PARTITION == "domainmix":
        # (C, N) distribution matrix, same shape the dirichlet path produces
        per_dom = NUM_USERS // len(DOMAIN_ORDER)
        cd = [d for d in range(len(DOMAIN_ORDER)) for _ in range(per_dom)]
        while len(cd) < NUM_USERS:
            cd.append(len(DOMAIN_ORDER) - 1)
        cd = np.array(cd); rng.shuffle(cd)   # client index must not encode domain
        client_domain = cd.tolist()
        dist = np.zeros((C, NUM_USERS))
        for c in range(C):
            rows = np.where(y == c)[0]
            if len(rows) == 0:
                continue
            dom = int(np.bincount(day[rows], minlength=len(DOMAIN_ORDER)).argmax())
            holders = np.flatnonzero(cd == dom)
            hard = np.zeros(NUM_USERS)
            if c == 0 or holders.size == 0:
                hard[:] = 1.0 / NUM_USERS          # BENIGN spread over everyone
            else:
                hard[holders] = 1.0 / holders.size
            dist[c] = DOMAIN_MIX * hard + (1 - DOMAIN_MIX) / NUM_USERS
            dist[c] /= dist[c].sum()
            cut = (np.cumsum(dist[c])[:-1] * len(rows)).astype(int)
            for u, part in enumerate(np.split(rows, cut)):
                owner[part] = u

    elif PARTITION == "labelskew":
        client_domain = [-1] * NUM_USERS
        attacks = [c for c in range(C) if c != 0]
        k = max(1, min(CLASSES_PER_CLIENT, len(attacks)))
        # deal attack classes round-robin so every class has roughly the same
        # number of holders and neighbouring clients overlap
        holders = {c: [] for c in attacks}
        for u in range(NUM_USERS):
            for j in range(k):
                holders[attacks[(u * k + j) % len(attacks)]].append(u)
        for c in attacks:
            rc = np.where(y == c)[0]
            hs = holders[c] or list(range(NUM_USERS))
            owner[rc] = np.array(hs)[np.arange(len(rc)) % len(hs)]
        ben = np.where(y == 0)[0]                 # BENIGN to everyone
        owner[ben] = np.arange(len(ben)) % NUM_USERS

    elif PARTITION == "dirichlet":
        client_domain = [-1] * NUM_USERS
        for c in range(C):
            rc = np.where(y == c)[0]
            rng.shuffle(rc)
            p = rng.dirichlet([DIR_ALPHA] * NUM_USERS)
            if cls_counts[c] < RARE_BELOW:      # concentrate rare classes
                pick = rng.choice(NUM_USERS, min(CLIENTS_PER_RARE, NUM_USERS), replace=False)
                q = np.zeros(NUM_USERS); q[pick] = p[pick] + 1e-9
                p = q / q.sum()
            cut = (np.cumsum(p)[:-1] * len(rc)).astype(int)
            for u, part in enumerate(np.split(rc, cut)):
                owner[part] = u
    else:
        raise ValueError(f"CICIDS_PARTITION must be domain, domainmix, labelskew or dirichlet, got {PARTITION!r}")
    # A low dirichlet alpha can leave a client with no rows at all, and
    # DataLoader rejects batch_size=0. Give any empty client a small stratified
    # slice taken from the largest holder of each class.
    for u in range(NUM_USERS):
        if (owner == u).sum() >= MIN_CLIENT_ROWS:
            continue
        for c in range(C):
            pool = np.where((y == c) & (owner != u))[0]
            if len(pool) > MIN_CLIENT_ROWS:
                owner[rng.choice(pool, MIN_CLIENT_ROWS // C + 1, replace=False)] = u

    # ---------------- per-client cap, rare classes protected ----------------
    sel = []
    for u in range(NUM_USERS):
        idx = np.where(owner == u)[0]
        if len(idx) <= MAX_PER_CLIENT:
            sel.append(idx); continue
        keep_parts, budget = [], MAX_PER_CLIENT
        rare_here = [c for c in range(C)
                     if cls_counts[c] < RARE_BELOW and (y[idx] == c).any()]
        # Never let the rare-class reservation consume the whole budget, or a
        # client ends up holding rare classes and no BENIGN at all.
        per_rare = min(RARE_FLOOR, MAX_PER_CLIENT // (2 * max(len(rare_here), 1)))
        for c in rare_here:
            ic = idx[y[idx] == c]
            # take the LAST rows in time order, so the per-class temporal split
            # still cuts at the end of the class timeline rather than mid-way
            k = min(len(ic), per_rare)
            keep_parts.append(ic[-k:]); budget -= k
        common = np.concatenate([idx[y[idx] == c] for c in range(C)
                                 if cls_counts[c] >= RARE_BELOW and (y[idx] == c).any()] or
                                [np.array([], dtype=np.int64)])
        if budget > 0 and len(common):
            keep_parts.append(rng.choice(common, min(budget, len(common)), replace=False))
        sel.append(np.sort(np.concatenate(keep_parts)) if keep_parts else idx[:MAX_PER_CLIENT])

    # ---------------- split ----------------
    train_data = {"users": [], "user_data": {}, "num_samples": []}
    test_data  = {"users": [], "user_data": {}, "num_samples": []}
    tr_all = []
    for u, idx in enumerate(sel):
        tr, te = [], []
        for c in np.unique(y[idx]):
            rc = idx[y[idx] == c]
            rc = rc[np.argsort(t[rc])] if SPLIT == "temporal" else rc[rng.permutation(len(rc))]
            cut = max(1, int(round(0.75 * len(rc))))
            tr.append(rc[:cut]); te.append(rc[cut:])
        tr_all.append(np.concatenate(tr) if tr else np.array([], np.int64))
        sel[u] = (np.concatenate(tr) if tr else np.array([], np.int64),
                  np.concatenate(te) if te else np.array([], np.int64))

    if CROSS_TEST:
        # pool every client's held-out rows, then deal them back out to ALL
        # clients stratified by class, so each client is tested on all classes
        pooled_te = np.concatenate([te for _, te in sel]) if sel else np.array([], np.int64)
        by_cls = {c: pooled_te[y[pooled_te] == c] for c in np.unique(y[pooled_te])}
        for c in by_cls:
            rng.shuffle(by_cls[c])
        newte = [[] for _ in range(NUM_USERS)]
        for c, rows in by_cls.items():
            for u in range(NUM_USERS):
                newte[u].append(rows[u::NUM_USERS])
        sel = [(tr, np.sort(np.concatenate(newte[u])) if newte[u] else np.array([], np.int64))
               for u, (tr, _) in enumerate(sel)]

    if BENIGN_FRACTION < 1.0:
        # TRAIN ONLY. SCMoE thins after the split (tabular_federated_data.py
        # 502-518) and leaves test untouched, which is correct: the test set
        # must reflect real traffic at its natural ~82% BENIGN because that is
        # the deployment condition. Thinning test would inflate macro F1 by
        # shrinking the majority class in evaluation.
        newsel = []
        for u, (tr, te) in enumerate(sel):
            ben = tr[y[tr] == 0]
            keep = max(int(round(len(ben) * BENIGN_FRACTION)), min(len(ben), 50))
            drop = rng.choice(ben, len(ben) - keep, replace=False) if len(ben) > keep else np.array([], np.int64)
            newsel.append((np.setdiff1d(tr, drop, assume_unique=False), te))
        sel = newsel
        tr_all = [tr for tr, _ in sel]

    pooled = X[np.concatenate(tr_all)]
    if CLIP_PCT > 0:
        lo = np.percentile(pooled, 100 - CLIP_PCT, axis=0)
        hi = np.percentile(pooled, CLIP_PCT, axis=0)
        X = np.clip(X, np.minimum(lo, hi), np.maximum(lo, hi))
        pooled = X[np.concatenate(tr_all)]
        if verbose:
            print(f"  clipped features to [{100-CLIP_PCT:.1f}, {CLIP_PCT:.1f}] train percentiles")
    mu, sd = pooled.mean(0), pooled.std(0); sd[sd == 0] = 1.0

    for u, (tr, te) in enumerate(sel):
        a = ((X[tr] - mu) / sd).astype(np.float32)
        b = ((X[te] - mu) / sd).astype(np.float32)
        train_data["users"].append(u)
        train_data["user_data"][u] = {"x": a.tolist(), "y": y[tr].astype(float).tolist()}
        train_data["num_samples"].append(len(a))
        test_data["users"].append(u)
        test_data["user_data"][u] = {"x": b.tolist(), "y": y[te].astype(float).tolist()}
        test_data["num_samples"].append(len(b))

    # ---------------- teams ----------------
    group = [[] for _ in range(NUM_GROUPS)]
    per_team = NUM_USERS // NUM_GROUPS
    if group_division == 0:
        for i in range(NUM_USERS):
            group[min(i // per_team, NUM_GROUPS - 1)].append(i)
    else:
        order = list(range(NUM_USERS))
        np.random.RandomState(int(os.environ.get("CICIDS_TEAM_SEED", "0"))).shuffle(order)
        for i, u in enumerate(order):
            group[min(i // per_team, NUM_GROUPS - 1)].append(u)

    if verbose:
        _report_heterogeneity(train_data, NUM_USERS, C)
        print(f"  features={len(feat)} partition={PARTITION} split={SPLIT} "
              f"train={sum(train_data['num_samples']):,} test={sum(test_data['num_samples']):,}")
    read_cicids_data.client_domain = client_domain
    read_cicids_data.num_classes = C
    return train_data["users"], group, train_data["user_data"], test_data["user_data"]


def _report_heterogeneity(train_data, n, C):
    """Mean pairwise Jensen-Shannon divergence between client label
    distributions. 0 means every client has the same mix, 1 means disjoint.
    Gives a single measurable number for how non-IID a partition actually is."""
    P = np.zeros((n, C))
    for u in range(n):
        for v in train_data["user_data"][u]["y"]:
            P[u, int(v)] += 1
    P = P / np.maximum(P.sum(1, keepdims=True), 1)
    def js(p, q):
        m = 0.5 * (p + q)
        def kl(a, b):
            m = a > 0
            return float(np.sum(a[m] * np.log2(a[m] / np.where(b[m] > 0, b[m], 1))))
        return 0.5 * kl(p, m) + 0.5 * kl(q, m)
    d = [js(P[i], P[j]) for i in range(n) for j in range(i + 1, n)]
    eff = float(np.mean([np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))) for p in P]))
    print(f"  heterogeneity: mean pairwise JSD {np.mean(d):.4f} "
          f"(0=identical, 1=disjoint), effective classes/client {eff:.2f} of {C}")
