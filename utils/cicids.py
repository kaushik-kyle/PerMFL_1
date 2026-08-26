"""CICIDS2017 loader for PerMFL.

Returns the same four-tuple as every other loader in utils/model_utils.py:
    clients, groups, train_data, test_data

Design decisions, all deliberate and all documented in RUNS.md:

  * Reads TrafficLabelling/, NOT MachineLearningCVE/. The latter has no
    Timestamp column, so a temporal split on it is impossible.
  * Clients are built from day-domains, the natural structure of the capture.
    5 domains x 4 clients = 20 clients. Sequential team assignment therefore
    reproduces the domains exactly, which gives ground truth for ARI.
  * Split is temporal WITHIN each client AND WITHIN each class. A global split
    would leave Thursday/Friday clients with no training data for their own
    classes; a per-client-only split gives the test set whichever attack ran
    last and puts the others entirely in train.
  * Train is the EARLIER 75% of each class, test the LATER 25%. Every other
    loader in this repo takes test from the front, which is correct for a
    shuffled split and wrong for a temporal one.
"""
import os, glob, numpy as np, pandas as pd

DATA_DIR = os.environ.get("CICIDS_DIR",
    "/Users/kaushik/Projects/Masters/Dissertation/datasets/OneDrive_1_03-08-2026/TrafficLabelling")

# day -> (domain index, filename fragments)
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

# 15 raw labels -> 9 classes. Collapses the DoS variants, the two Patators,
# and the three Web Attack variants, which otherwise leave classes with 11 to
# 21 rows that cannot survive a 75/25 split.
LABEL_MAP = {
    "BENIGN": 0, "Bot": 1, "DDoS": 2,
    "DoS Hulk": 3, "DoS GoldenEye": 3, "DoS slowloris": 3, "DoS Slowhttptest": 3,
    "FTP-Patator": 4, "SSH-Patator": 4,
    # Heartbleed has 11 rows in the entire capture. Split 75/25 across four
    # Wednesday clients it yields 2-3 test samples total, which cannot be
    # learned and makes macro F1 a coin flip on that class. Folded into DoS,
    # which it belongs to anyway as a Wednesday DoS-family attack.
    "Heartbleed": 3,
    "Infiltration": 5, "PortScan": 6,
}
WEB_ATTACK = 7   # any label starting "Web Attack"
NUM_CLASSES = 8
CLASS_NAMES = ["BENIGN", "Bot", "DDoS", "DoS", "BruteForce",
               "Infiltration", "PortScan", "WebAttack"]

BENIGN_RATIO = float(os.environ.get("CICIDS_BENIGN_RATIO", "3.0"))
MAX_PER_CLIENT = int(os.environ.get("CICIDS_MAX_PER_CLIENT", "8000"))


def _fix_timestamps(s):
    """CICIDS timestamps are 12-hour with no AM/PM in 7 of 8 files.

    Wednesday runs '8:42' to '2:43' because 2:43 is 14:43. The capture window
    is 08:00-17:00, so hours 1-7 are afternoon and get +12. Monday is the one
    file already in 24-hour form with seconds, and is parsed separately.
    """
    dt = pd.to_datetime(s, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    todo = dt.isna()
    if todo.any():
        alt = pd.to_datetime(s[todo], format="%d/%m/%Y %H:%M", errors="coerce")
        bump = alt.dt.hour.between(1, 7)
        alt = alt + pd.to_timedelta(np.where(bump.fillna(False), 12, 0), unit="h")
        dt.loc[todo] = alt
    return dt


def _load_day(day):
    frames = []
    for frag in DAY_FILES[day]:
        hits = glob.glob(os.path.join(DATA_DIR, frag + "*.csv"))
        if not hits:
            raise FileNotFoundError(f"{frag}*.csv not found under {DATA_DIR}")
        df = pd.read_csv(hits[0], encoding="latin-1", low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df = df.dropna(how="all")                     # the 288,602 blank rows
        df = df[df["Label"].notna()]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _to_class(lbl):
    lbl = str(lbl).strip()
    if lbl.startswith("Web Attack"):
        return WEB_ATTACK
    return LABEL_MAP.get(lbl, -1)


def read_cicids_data(NUM_USERS, NUM_LABELS, NUM_GROUPS, group_division, verbose=True):
    rng = np.random.RandomState(0)
    per_domain = NUM_USERS // len(DOMAIN_ORDER)
    assert per_domain >= 1, "need at least 5 clients, one per day-domain"

    client_frames, client_domain = [], []
    feat_cols = None

    for d_idx, day in enumerate(DOMAIN_ORDER):
        df = _load_day(day)
        df["_y"] = df["Label"].map(_to_class)
        df = df[df["_y"] >= 0]
        df["_t"] = _fix_timestamps(df["Timestamp"].astype(str).str.strip())
        df = df[df["_t"].notna()]

        if feat_cols is None:
            drop = {"Flow ID", "Source IP", "Source Port", "Destination IP",
                    "Protocol", "Timestamp", "Label", "_y", "_t"}
            feat_cols = [c for c in df.columns if c not in drop]

        # keep every attack row, subsample BENIGN to BENIGN_RATIO x attacks
        atk = df[df["_y"] != 0]
        ben = df[df["_y"] == 0]
        want = int(len(atk) * BENIGN_RATIO) if len(atk) else len(ben)
        if want and len(ben) > want:
            ben = ben.iloc[rng.choice(len(ben), want, replace=False)]
        day_df = pd.concat([atk, ben]).sort_values("_t", kind="mergesort")

        # deal out to this domain's clients, round-robin over time so every
        # client spans the whole day rather than one contiguous slice
        for k in range(per_domain):
            sub = day_df.iloc[k::per_domain]
            if len(sub) > MAX_PER_CLIENT:
                step = len(sub) / MAX_PER_CLIENT
                sub = sub.iloc[(np.arange(MAX_PER_CLIENT) * step).astype(int)]
            client_frames.append(sub)
            client_domain.append(d_idx)
        if verbose:
            print(f"  {day:<10} rows={len(day_df):>7}  attacks={len(atk):>7}  "
                  f"classes={sorted(day_df['_y'].unique())}")

    # ---- clean, temporal split, standardise on pooled TRAIN only ----
    Xtr, ytr, Xte, yte = [], [], [], []
    for sub in client_frames:
        X = sub[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
        X[~np.isfinite(X)] = np.nan
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = sub["_y"].to_numpy(np.int64)
        # Per-CLASS temporal split, not per-client. CICIDS attacks run in
        # contiguous time windows, so a flat cut by time gives the test set
        # whichever attack ran last and puts the rest entirely in train.
        # Measured on the flat version: Wednesday clients trained on 2913 DoS
        # and tested on zero, Friday trained on 1808 PortScan and tested on
        # zero. Splitting each class on its own timeline keeps every class in
        # both halves while still separating same-burst near-duplicates.
        tr_idx, te_idx = [], []
        for cls in np.unique(y):
            rows = np.where(y == cls)[0]          # already in timestamp order
            cut = int(round(0.75 * len(rows)))
            if len(rows) == 1:                    # singleton class -> train
                cut = 1
            tr_idx.append(rows[:cut]); te_idx.append(rows[cut:])
        tr_idx = np.sort(np.concatenate(tr_idx))
        te_idx = np.sort(np.concatenate(te_idx)) if te_idx else np.array([], int)
        Xtr.append(X[tr_idx]); ytr.append(y[tr_idx])
        Xte.append(X[te_idx]); yte.append(y[te_idx])

    pooled = np.concatenate(Xtr)
    mu, sd = pooled.mean(0), pooled.std(0)
    sd[sd == 0] = 1.0

    train_data = {"users": [], "user_data": {}, "num_samples": []}
    test_data = {"users": [], "user_data": {}, "num_samples": []}
    for i in range(NUM_USERS):
        a = ((Xtr[i] - mu) / sd).astype(np.float32)
        b = ((Xte[i] - mu) / sd).astype(np.float32)
        train_data["users"].append(i)
        train_data["user_data"][i] = {"x": a.tolist(), "y": ytr[i].astype(float).tolist()}
        train_data["num_samples"].append(len(a))
        test_data["users"].append(i)
        test_data["user_data"][i] = {"x": b.tolist(), "y": yte[i].astype(float).tolist()}
        test_data["num_samples"].append(len(b))

    # ---- teams ----
    group = [[] for _ in range(NUM_GROUPS)]
    per_team = NUM_USERS // NUM_GROUPS
    if group_division == 0:          # sequential == domain-aligned == oracle
        for i in range(NUM_USERS):
            group[min(i // per_team, NUM_GROUPS - 1)].append(i)
    elif group_division == 1:        # random control, seeded for reproducibility
        order = list(range(NUM_USERS))
        np.random.RandomState(int(os.environ.get("CICIDS_TEAM_SEED", "0"))).shuffle(order)
        for i, u in enumerate(order):
            group[min(i // per_team, NUM_GROUPS - 1)].append(u)
    else:
        raise ValueError("group_division must be 0 (domain-aligned) or 1 (random)")

    if verbose:
        print(f"  features={len(feat_cols)}  clients={NUM_USERS}  teams={NUM_GROUPS}")
    read_cicids_data.client_domain = client_domain   # ground truth for ARI
    return train_data["users"], group, train_data["user_data"], test_data["user_data"]
