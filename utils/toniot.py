"""TON-IoT Network loader for PerMFL.

Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv, the
pre-sampled subset UNSW provides for ML evaluation.

Chosen as a third dataset because it is near-balanced where CICIDS2017 is not:

    dataset      majority class   accuracy floor   macro F1 floor
    CICIDS2017   81.7% BENIGN     0.8171           0.0999
    NSL-KDD      53.5% normal     0.5346           ~0.15
    TON-IoT      23.7% normal     0.2369           ~0.038

Macro F1 averages equally over classes, so on CICIDS eight of nine classes are
learned from 18.3% of the rows and the metric has little room to move. If an
intervention also helps here, at a floor of 0.038 with nine attack classes at
9.48% each, the result is not an artefact of CICIDS's imbalance.

src_ip and dst_ip are dropped as FEATURES -- attacker addresses are constant
(192.168.159.30-39 per the ground-truth folder) so they leak the label -- but
dst_ip is retained for the victim-host partition.

  TONIOT_DIR         directory holding train_test_network.csv
  TONIOT_PARTITION   domain (default) | victimip | dirichlet
  TONIOT_MIN_ENTROPY host label-entropy floor for victimip     (0.2, after DP-FL)
  TONIOT_ALPHA       dirichlet concentration                   (0.5)
"""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("TONIOT_DIR",
    "data/TON_IoT/Train_Test_datasets/Train_Test_Network_dataset")
PARTITION = os.environ.get("TONIOT_PARTITION", "domain")
MIN_ENTROPY = float(os.environ.get("TONIOT_MIN_ENTROPY", "0.2"))
ALPHA = float(os.environ.get("TONIOT_ALPHA", "0.5"))
MIN_ROWS = int(os.environ.get("TONIOT_MIN_CLIENT_ROWS", "200"))
CROSS_TEST = int(os.environ.get("TONIOT_CROSS_TEST", "0"))

DROP = {"src_ip", "src_port", "dst_ip", "dst_port", "ts", "label", "type"}
CAT = ["proto", "service", "conn_state", "dns_query", "ssl_version", "ssl_cipher",
       "ssl_subject", "ssl_issuer", "http_method", "http_uri", "http_version",
       "http_user_agent", "http_orig_mime_types", "http_resp_mime_types",
       "weird_name", "weird_addl", "weird_notice", "dns_AA", "dns_RD", "dns_RA",
       "dns_rejected", "ssl_resumed", "ssl_established", "http_trans_depth"]
CLASS_NAMES = []


def _entropy(counts):
    p = counts[counts > 0] / counts.sum()
    k = len(counts)
    return float(-(p * np.log(p)).sum() / np.log(k)) if k > 1 else 0.0


def read_toniot_data(NUM_USERS, NUM_LABELS, NUM_GROUPS, group_division, verbose=True):
    global CLASS_NAMES
    seed = int(os.environ.get("PERMFL_SEED", "0"))
    rng = np.random.RandomState(seed)
    df = pd.read_csv(os.path.join(DATA_DIR, "train_test_network.csv"), low_memory=False)
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    df = df.dropna(how="all")

    CLASS_NAMES = sorted(df["type"].astype(str).str.strip().unique())
    y = df["type"].astype(str).str.strip().map({n: i for i, n in enumerate(CLASS_NAMES)}).to_numpy(np.int64)
    C = len(CLASS_NAMES)
    dst = df["dst_ip"].astype(str).to_numpy()

    feat_df = df.drop(columns=[c for c in df.columns if c in DROP], errors="ignore")
    cats = [c for c in feat_df.columns if c in CAT or feat_df[c].dtype == object]
    for c in cats:                          # high-cardinality strings -> frequency rank
        vc = feat_df[c].astype(str).value_counts()
        feat_df[c] = feat_df[c].astype(str).map({v: i for i, v in enumerate(vc.index)}).fillna(-1)
    X = feat_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    keep = X.notna().all(axis=1).to_numpy()
    X, y, dst = X[keep].to_numpy(np.float32), y[keep], dst[keep]
    if verbose:
        print(f"  TON-IoT Network: {len(X):,} rows, {X.shape[1]} features, {C} classes")

    owner = np.full(len(y), -1, np.int64)
    client_domain = [-1] * NUM_USERS
    if PARTITION == "victimip":
        # One client per destination host, selected by label entropy. Most
        # victim hosts see a single attack type, which would make every client
        # single-class; DP-FL filters on entropy > 0.2 for exactly this reason.
        hosts, ent = [], []
        for h in pd.unique(dst):
            rows = np.where(dst == h)[0]
            if len(rows) < MIN_ROWS:
                continue
            e = _entropy(np.bincount(y[rows], minlength=C))
            if e >= MIN_ENTROPY:
                hosts.append(h); ent.append((e, len(rows)))
        order = np.argsort([-r[1] for r in ent])[:NUM_USERS]
        chosen = [hosts[i] for i in order]
        if verbose:
            print(f"  victimip: {len(hosts)} hosts pass entropy>={MIN_ENTROPY}, "
                  f"taking top {len(chosen)} by volume")
        for u, h in enumerate(chosen):
            owner[dst == h] = u
        left = np.where(owner < 0)[0]       # unassigned rows spread round-robin
        owner[left] = np.arange(len(left)) % NUM_USERS
    elif PARTITION == "domain":
        # one attack family per client group, normal spread over everyone
        attacks = [c for c in range(C) if CLASS_NAMES[c] != "normal"]
        per = max(NUM_USERS // max(len(attacks), 1), 1)
        client_domain = [min(i // per, len(attacks) - 1) for i in range(NUM_USERS)]
        for d, c in enumerate(attacks):
            mem = [i for i in range(NUM_USERS) if client_domain[i] == d] or list(range(NUM_USERS))
            rows = np.where(y == c)[0]
            owner[rows] = np.array(mem)[np.arange(len(rows)) % len(mem)]
        nrm = np.where(np.array([CLASS_NAMES[v] == "normal" for v in y]))[0]
        owner[nrm] = np.arange(len(nrm)) % NUM_USERS
    else:
        for c in range(C):
            rows = np.where(y == c)[0]; rng.shuffle(rows)
            p = rng.dirichlet([ALPHA] * NUM_USERS)
            cut = (np.cumsum(p)[:-1] * len(rows)).astype(int)
            for u, part in enumerate(np.split(rows, cut)):
                owner[part] = u
    for u in range(NUM_USERS):
        if (owner == u).sum() < MIN_ROWS:
            pool = np.where(owner != u)[0]
            owner[rng.choice(pool, min(MIN_ROWS, len(pool)), replace=False)] = u

    sel = []
    for u in range(NUM_USERS):
        idx = np.where(owner == u)[0]
        tr, te = [], []
        for c in np.unique(y[idx]):
            rc = idx[y[idx] == c]; rc = rc[rng.permutation(len(rc))]
            cut = max(1, int(round(0.75 * len(rc))))
            tr.append(rc[:cut]); te.append(rc[cut:])
        sel.append((np.concatenate(tr), np.concatenate(te) if te else np.array([], np.int64)))
    if CROSS_TEST:
        pool = np.concatenate([t for _, t in sel])
        newte = [[] for _ in range(NUM_USERS)]
        for c in np.unique(y[pool]):
            rows = pool[y[pool] == c]; rng.shuffle(rows)
            for u in range(NUM_USERS):
                newte[u].append(rows[u::NUM_USERS])
        sel = [(tr, np.concatenate(newte[u])) for u, (tr, _) in enumerate(sel)]

    allc = np.concatenate([tr for tr, _ in sel])
    mu, sd = X[allc].mean(0), X[allc].std(0); sd[sd == 0] = 1.0
    train_data = {"users": [], "user_data": {}, "num_samples": []}
    test_data = {"users": [], "user_data": {}, "num_samples": []}
    for u, (tr, te) in enumerate(sel):
        a, b = ((X[tr] - mu) / sd).astype(np.float32), ((X[te] - mu) / sd).astype(np.float32)
        train_data["users"].append(u); test_data["users"].append(u)
        train_data["user_data"][u] = {"x": a.tolist(), "y": y[tr].astype(float).tolist()}
        test_data["user_data"][u] = {"x": b.tolist(), "y": y[te].astype(float).tolist()}
        train_data["num_samples"].append(len(a)); test_data["num_samples"].append(len(b))

    group = [[] for _ in range(NUM_GROUPS)]
    per_team = NUM_USERS // NUM_GROUPS
    order = list(range(NUM_USERS))
    if group_division != 0:
        np.random.RandomState(seed).shuffle(order)
    for i, u in enumerate(order):
        group[min(i // per_team, NUM_GROUPS - 1)].append(u)

    if verbose:
        P = np.zeros((NUM_USERS, C))
        for u in range(NUM_USERS):
            for v in train_data["user_data"][u]["y"]: P[u, int(v)] += 1
        P = P / np.maximum(P.sum(1, keepdims=True), 1)
        kl = lambda a, b: float(np.sum(a[a > 0] * np.log2(a[a > 0] / np.where(b[a > 0] > 0, b[a > 0], 1))))
        js = [0.5*kl(P[i], .5*(P[i]+P[j])) + 0.5*kl(P[j], .5*(P[i]+P[j]))
              for i in range(NUM_USERS) for j in range(i+1, NUM_USERS)]
        print(f"  heterogeneity: mean pairwise JSD {np.mean(js):.4f}  partition={PARTITION}")
    read_toniot_data.client_domain = client_domain
    read_toniot_data.num_classes = C
    read_toniot_data.num_features = X.shape[1]
    return train_data["users"], group, train_data["user_data"], test_data["user_data"]
