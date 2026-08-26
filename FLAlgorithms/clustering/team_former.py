"""Data-derived team formation for PerMFL.

PerMFL assigns teams in the data loader, before training starts, so no
gradient-based grouping is possible there. This module runs in the server's
global-round loop instead, where client updates exist.

Three pieces, each taken from a named source and adapted where PerMFL's
architecture forces a change:

  signal      which vector represents a client.
              'residual' is theta_ij - w_i, the proximal residual PerMFL
              already computes in pFedMeOptimizer.step. It measures how far a
              device wants to move GIVEN the pull its team is exerting, which
              is a better-posed question than raw gradient distance. Neither
              SCMoE nor CFMD-i can use this, since neither has a proximal term.
              'grad' is the raw update, which is what SCMoE-PFL uses.

  trigger     CFMD-i, Eq. 19-20. Cluster only once inter-client differences
              are large enough to be discriminative but not so large that the
              grouping is unstable. Replaces "cluster at a fixed round", which
              CFMD-i criticises as non-adaptive.

  form        SCMoE-PFL's MCTC, Algorithm 1: L2-normalise to discard
              magnitude, PCA to escape the distance-concentration effect that
              makes cosine meaningless in high dimensions, then seed centres
              from the most dissimilar pairs.

              DEVIATION: MCTC produces soft, overlapping membership. PerMFL
              sizes self.team, self.users and self.tau as fixed-width arrays
              and divides by team sample counts, so an empty or overlapping
              team breaks it. Assignment here is therefore hard and
              capacity-constrained to equal sizes. The size constraint plays
              the role of CFMD-i's cluster-balance term H(c1,c2) (Eq. 22).
"""
import numpy as np
import torch


def flatten(params):
    return torch.cat([p.data.reshape(-1) for p in params]).cpu().numpy()


def client_signal(user, team_model, mode="residual"):
    """One vector per client. See module docstring for the two modes."""
    theta = flatten(user.local_model)
    if mode == "grad":
        return theta
    if mode == "residual":
        return theta - flatten(team_model.parameters())
    raise ValueError(f"unknown signal mode {mode!r}")


def should_recluster(signals, eps_lo, eps_hi):
    """CFMD-i adaptive trigger.

    (i)  max pairwise difference > eps_hi: enough heterogeneity has emerged
    (ii) mean pairwise difference < eps_lo: shared structure still remains

    Returns (bool, max, mean) so the caller can log the thresholds it is
    actually operating against rather than guessing them.
    """
    S = np.stack(signals)
    d = np.linalg.norm(S[:, None, :] - S[None, :, :], axis=-1)
    iu = np.triu_indices(len(S), k=1)
    mx, mean = float(d[iu].max()), float(d[iu].mean())
    return (mx > eps_hi and mean < eps_lo), mx, mean


def form_teams(signals, num_teams, pca_dim=8, seed=0):
    """MCTC-style hard clustering with equal-size teams.

    Returns a list of num_teams lists of client indices.
    """
    S = np.stack(signals).astype(np.float64)
    n = len(S)

    # 1. L2-normalise: direction only, magnitude discarded (MCTC Eq. 1)
    norms = np.linalg.norm(S, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    S = S / norms

    # 2. PCA (MCTC Eq. 2). At most n-1 components exist for n clients.
    m = int(min(pca_dim, n - 1, S.shape[1]))
    if m >= 1:
        Sc = S - S.mean(0, keepdims=True)
        # economy SVD is the stable way to get the top components here
        _, _, Vt = np.linalg.svd(Sc, full_matrices=False)
        Z = Sc @ Vt[:m].T
    else:
        Z = S

    # 3. cosine similarity in the reduced space
    zn = np.linalg.norm(Z, axis=1, keepdims=True)
    zn[zn == 0] = 1.0
    Zn = Z / zn
    sim = Zn @ Zn.T

    # 4. seed centres from the most dissimilar pairs (MCTC Alg 1, lines 5-15)
    centres, work = [], sim.copy()
    np.fill_diagonal(work, np.inf)
    while len(centres) < num_teams:
        i, j = np.unravel_index(np.argmin(work), work.shape)
        for c in (i, j):
            if c not in centres and len(centres) < num_teams:
                centres.append(int(c))
        work[i, j] = work[j, i] = np.inf
        if not np.isfinite(work).any():
            break
    rng = np.random.RandomState(seed)
    for c in rng.permutation(n):                 # pad if ties exhausted early
        if len(centres) >= num_teams:
            break
        if c not in centres:
            centres.append(int(c))

    # 5. capacity-constrained assignment. Equal sizes stand in for CFMD-i's
    #    balance term and guarantee no empty team, which self.tau requires.
    cap = n // num_teams
    teams = [[] for _ in range(num_teams)]
    for t, c in enumerate(centres):
        teams[t].append(c)
    unassigned = [i for i in range(n) if i not in centres]
    # best-first over every (client, team) pair by similarity to that centre
    pairs = sorted(
        ((sim[i, centres[t]], i, t) for i in unassigned for t in range(num_teams)),
        key=lambda x: -x[0])
    placed = set()
    for _, i, t in pairs:
        if i in placed or len(teams[t]) >= cap:
            continue
        teams[t].append(i)
        placed.add(i)
    for i in unassigned:                          # remainder -> smallest team
        if i not in placed:
            teams[min(range(num_teams), key=lambda t: len(teams[t]))].append(i)
    return [sorted(t) for t in teams]


def agreement(teams, truth):
    """Adjusted Rand Index against known client labels (day-domains on CICIDS).

    Implemented here rather than pulled from sklearn, which is not a
    dependency of this repo.
    """
    lab = np.empty(sum(len(t) for t in teams), dtype=int)
    for t, members in enumerate(teams):
        for m in members:
            lab[m] = t
    truth = np.asarray(truth)
    k1, k2 = lab.max() + 1, truth.max() + 1
    cont = np.zeros((k1, k2), dtype=np.int64)
    for a, b in zip(lab, truth):
        cont[a, b] += 1
    comb = lambda x: x * (x - 1) / 2.0
    sum_ij = comb(cont).sum()
    sum_a = comb(cont.sum(1)).sum()
    sum_b = comb(cont.sum(0)).sum()
    total = comb(np.array(len(lab), dtype=np.float64))
    exp = sum_a * sum_b / total
    mx = (sum_a + sum_b) / 2.0
    return 0.0 if mx == exp else float((sum_ij - exp) / (mx - exp))
