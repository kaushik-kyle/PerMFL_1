"""NSL-KDD loader for PerMFL.

Chosen as a second IDS dataset because it is the strongest available test of
PerMFL's premise, which is that the method helps "when there are known team
structures across devices":

  * four documented attack domains (DoS, Probe, R2L, U2R) plus normal, the
    same partition CFMD-i uses on this dataset, so team structure is real
    rather than imposed
  * a canonical train/test split ships with the data, so no split has to be
    designed and none of the CICIDS leakage questions apply
  * accuracy floor 0.4308 on the test set against 0.8148 for CICIDS, so
    accuracy and macro F1 both carry information

The canonical test set deliberately over-represents R2L (12.8% against 0.8%
in train). That is the benchmark's intended difficulty and is not corrected.

  NSLKDD_DIR             directory holding KDDTrain+.txt and KDDTest+.txt
  NSLKDD_PARTITION       domain (default) | dirichlet
  NSLKDD_DIRICHLET_ALPHA concentration for the dirichlet partition (0.5)
  NSLKDD_MIN_CLIENT_ROWS floor on rows per client                  (200)
"""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("NSLKDD_DIR", "data/nsl_kdd")
PARTITION = os.environ.get("NSLKDD_PARTITION", "domain")
DIR_ALPHA = float(os.environ.get("NSLKDD_DIRICHLET_ALPHA", "0.5"))
MIN_ROWS = int(os.environ.get("NSLKDD_MIN_CLIENT_ROWS", "200"))

COLS = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
 'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
 'root_shell','su_attempted','num_root','num_file_creations','num_shells',
 'num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count',
 'srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
 'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
 'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
 'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
 'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate',
 'label','difficulty']
CAT = ['protocol_type', 'service', 'flag']

DOS = {'neptune','back','land','pod','smurf','teardrop','mailbomb','apache2',
       'processtable','udpstorm','worm'}
PROBE = {'ipsweep','nmap','portsweep','satan','mscan','saint'}
R2L = {'ftp_write','guess_passwd','imap','multihop','phf','spy','warezclient',
       'warezmaster','sendmail','named','snmpgetattack','snmpguess','xlock',
       'xsnoop','httptunnel'}
U2R = {'buffer_overflow','loadmodule','perl','rootkit','ps','sqlattack','xterm'}
CLASS_NAMES = ["normal", "DoS", "Probe", "R2L", "U2R"]
NUM_CLASSES = 5
DOMAIN_ORDER = ["DoS", "Probe", "R2L", "U2R"]     # normal is spread across all


def _cls(l):
    l = str(l).strip()
    if l == "normal": return 0
    if l in DOS: return 1
    if l in PROBE: return 2
    if l in R2L: return 3
    if l in U2R: return 4
    return -1


def read_nslkdd_data(NUM_USERS, NUM_LABELS, NUM_GROUPS, group_division, verbose=True):
    rng = np.random.RandomState(0)
    tr = pd.read_csv(os.path.join(DATA_DIR, "KDDTrain+.txt"), names=COLS)
    te = pd.read_csv(os.path.join(DATA_DIR, "KDDTest+.txt"), names=COLS)
    for d in (tr, te):
        d.drop(columns=["difficulty"], inplace=True)
    ntr = len(tr)
    both = pd.concat([tr, te], ignore_index=True)
    y_all = both["label"].map(_cls).to_numpy(np.int64)
    both = both.drop(columns=["label"])

    # one-hot the three categoricals, fitted on train+test so the column set
    # matches; this encodes no label information
    X_all = pd.get_dummies(both, columns=CAT).to_numpy(np.float64)
    Xtr_r, ytr_r = X_all[:ntr], y_all[:ntr]
    Xte_r, yte_r = X_all[ntr:], y_all[ntr:]
    mu, sd = Xtr_r.mean(0), Xtr_r.std(0); sd[sd == 0] = 1.0
    Xtr_r = ((Xtr_r - mu) / sd).astype(np.float32)
    Xte_r = ((Xte_r - mu) / sd).astype(np.float32)
    if verbose:
        print(f"  NSL-KDD: train {len(Xtr_r):,} test {len(Xte_r):,} features {X_all.shape[1]} "
              f"classes {NUM_CLASSES}")

    def assign(y, n):
        owner = np.full(len(y), -1, np.int64)
        if PARTITION == "domain":
            per = max(n // len(DOMAIN_ORDER), 1)
            cd = [min(i // per, len(DOMAIN_ORDER) - 1) for i in range(n)]
            for d in range(len(DOMAIN_ORDER)):
                members = [i for i in range(n) if cd[i] == d]
                rows = np.where(y == d + 1)[0]          # this domain's attack class
                owner[rows] = np.array(members)[np.arange(len(rows)) % len(members)]
            normal = np.where(y == 0)[0]                # normal spread over everyone
            owner[normal] = np.arange(len(normal)) % n
            return owner, cd
        cd = [-1] * n
        for c in range(NUM_CLASSES):
            rc = np.where(y == c)[0]; rng.shuffle(rc)
            p = rng.dirichlet([DIR_ALPHA] * n)
            cut = (np.cumsum(p)[:-1] * len(rc)).astype(int)
            for u, part in enumerate(np.split(rc, cut)):
                owner[part] = u
        return owner, cd

    o_tr, client_domain = assign(ytr_r, NUM_USERS)
    o_te, _ = assign(yte_r, NUM_USERS)
    for o, y in ((o_tr, ytr_r), (o_te, yte_r)):      # no empty clients
        for u in range(NUM_USERS):
            if (o == u).sum() < MIN_ROWS:
                pool = np.where(o != u)[0]
                o[rng.choice(pool, min(MIN_ROWS, len(pool)), replace=False)] = u

    train_data = {"users": [], "user_data": {}, "num_samples": []}
    test_data = {"users": [], "user_data": {}, "num_samples": []}
    for u in range(NUM_USERS):
        a, b = np.where(o_tr == u)[0], np.where(o_te == u)[0]
        train_data["users"].append(u)
        train_data["user_data"][u] = {"x": Xtr_r[a].tolist(), "y": ytr_r[a].astype(float).tolist()}
        train_data["num_samples"].append(len(a))
        test_data["users"].append(u)
        test_data["user_data"][u] = {"x": Xte_r[b].tolist(), "y": yte_r[b].astype(float).tolist()}
        test_data["num_samples"].append(len(b))

    group = [[] for _ in range(NUM_GROUPS)]
    per_team = NUM_USERS // NUM_GROUPS
    order = list(range(NUM_USERS))
    if group_division != 0:
        np.random.RandomState(int(os.environ.get("NSLKDD_TEAM_SEED", "0"))).shuffle(order)
    for i, u in enumerate(order):
        group[min(i // per_team, NUM_GROUPS - 1)].append(u)

    if verbose:
        P = np.zeros((NUM_USERS, NUM_CLASSES))
        for u in range(NUM_USERS):
            for v in train_data["user_data"][u]["y"]:
                P[u, int(v)] += 1
        P = P / np.maximum(P.sum(1, keepdims=True), 1)
        def kl(a, b):
            m = a > 0
            return float(np.sum(a[m] * np.log2(a[m] / np.where(b[m] > 0, b[m], 1))))
        js = [0.5*kl(P[i], .5*(P[i]+P[j])) + 0.5*kl(P[j], .5*(P[i]+P[j]))
              for i in range(NUM_USERS) for j in range(i+1, NUM_USERS)]
        print(f"  heterogeneity: mean pairwise JSD {np.mean(js):.4f}  partition={PARTITION}")
    read_nslkdd_data.client_domain = client_domain
    read_nslkdd_data.num_classes = NUM_CLASSES
    read_nslkdd_data.num_features = X_all.shape[1]
    return train_data["users"], group, train_data["user_data"], test_data["user_data"]
