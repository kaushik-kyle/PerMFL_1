# Concepts

What this project does, in machine learning terms. Every term used here is
defined in the glossary at the end. Terms appearing inside a definition are
themselves defined.

## 1. The problem

Intrusion detection data cannot be pooled. Each organisation's network traffic
is private, and the attacks each one sees differ. Two consequences:

- A single model trained on pooled data is not available, because the data
  cannot leave its owner.
- A single model trained by federated learning fits no one well, because the
  clients' data distributions differ. This is **statistical heterogeneity**.

Personalised federated learning answers the second point: each client keeps its
own model, close to a shared one but free to deviate.

## 2. What PerMFL adds

PerMFL places a third tier between the client and the server.

```
            global model  x          one, shared by everyone
                  |
        +---------+---------+
        |                   |
    team model w1       team model w2       one per group of clients
        |                   |
    +---+---+           +---+---+
    θ   θ   θ           θ   θ   θ          one per client, private data
```

The intent: clients with similar traffic share a team model, which is closer to
them than the global model would be, while the global model still lets knowledge
reach clients that have never seen a given attack.

Only parameters move. No data leaves a client.

## 3. How the three tiers are held together

Each tier is pulled toward the one above it by a **quadratic penalty**: a cost
proportional to the squared distance between the two parameter vectors. Two
coefficients set the strength of those pulls.

| Coefficient | Pulls | Effect of a large value |
|---|---|---|
| λ (lambda) | client toward its team | clients resemble their team, less personalisation |
| γ (gamma) | team toward the global model | teams resemble the global model, less team identity |

The full objective is in the paper's equation 2. In code the three updates are
at [serverPerMFL.py:407](../FLAlgorithms/servers/serverPerMFL.py#L407),
[:424](../FLAlgorithms/servers/serverPerMFL.py#L424) and
[:441](../FLAlgorithms/servers/serverPerMFL.py#L441), with the client update in
[fedoptimizer.py:89](../FLAlgorithms/optimizers/fedoptimizer.py#L89).

## 4. The defect this project addresses

λ is defined once, as the client-to-team penalty. It acquires a second role
through the derivation of the team update.

- The team update needs the gradient of the client subproblem.
- That subproblem is a **Moreau envelope**, whose gradient identity is
  `∇f̃(w) = λ(w − θ)`.
- Substituting it makes λ the coefficient on the team's average of its members.

So one number sets two things:

| Where | What λ controls | A small λ means |
|---|---|---|
| Client update | how hard a client is pulled to its team | strong personalisation |
| Team update | how much a team listens to its own members | the team ignores its members |

These want opposite values. The paper's convergence condition `γ > 2λ` then caps
the ratio λ/γ below 0.5 permanently, so a team can never weight its members more
than half as strongly as it weights the global model. At the published defaults
the figure is 3.3 per cent.

## 5. Split-λ

Split-λ makes the team-update coefficient a separate parameter, `--lamda_team`,
instead of inheriting it from the client penalty.

| | Client update coefficient | Team update coefficient |
|---|---|---|
| PerMFL | λ | λ |
| Split-λ | λ | λ_team, set independently |

Nothing else changes. One line in
[serverPerMFL.py:413](../FLAlgorithms/servers/serverPerMFL.py#L413).

**The cost.** Substituting a free coefficient breaks the Moreau envelope
identity that produced it. The resulting algorithm no longer minimises the
paper's equation 2 exactly, and the paper's convergence proof does not transfer
as written. The empirical results hold; the theoretical guarantee does not carry
over unmodified.

## 6. Why the global model matters here

Two models come out of a run.

| Model | What it is | Why it matters for intrusion detection |
|---|---|---|
| Personalised (PM) | each client's own θ | detects what that client has already seen |
| Global (GM) | the shared x | the only route by which one client's attacks inform another |

A client cannot detect an attack class absent from its own training data using
its personalised model alone. The global model is what makes federating
worthwhile, and it is the tier the defect suppresses.

## 7. Why macro F1 rather than accuracy

CICIDS2017 is 81.7 per cent benign traffic. A model predicting "benign" for
every flow scores 0.817 accuracy and detects nothing.

| Metric | Uses the true-negative cell | Behaviour on imbalanced data |
|---|---|---|
| Accuracy | yes | dominated by the majority class |
| Macro F1 | no | each class contributes equally regardless of size |

Macro F1 averages per-class F1 without weighting by class size, so a rare attack
class counts as much as benign traffic. This is why the reported floor is 0.0999
macro F1 against 0.8171 accuracy.

---

# Glossary

| Term | Meaning |
|---|---|
| Accuracy | Fraction of predictions that are correct. Dominated by the majority class on imbalanced data |
| Adjusted Rand index (ARI) | Agreement between two partitions of the same items, corrected for chance. 1 is identical, 0 is chance |
| Aggregation | Combining parameters from several models into one, here by averaging |
| Benign | Normal, non-attack network traffic. The majority class in all three intrusion datasets |
| Class | One label a classifier can output, for example BENIGN or DDoS |
| Client | One participant holding private data. Also called a device. In this code, `--tot_users` of them |
| Confusion matrix | Table counting predictions by true class against predicted class. The diagonal is correct predictions |
| Convergence | The point at which further training stops changing the model materially |
| Cross-test | Evaluating a client on classes absent from its own training data. Simulates a zero-day attack |
| Device | See Client |
| Dirichlet partition | Splitting data across clients by drawing class proportions from a Dirichlet distribution. A lower α gives more skew |
| Domain partition | Splitting clients by a real-world grouping, here the capture day or the victim host |
| Epoch | One full pass over a dataset. Not used here: `--local_iters` counts minibatch steps, not epochs |
| False positive rate (FPR) | Of the samples not in a class, the fraction wrongly assigned to it. Lower is better. The alert-fatigue measure |
| Federated learning | Training a shared model across clients that never send their data, only parameters |
| Global model (GM) | The single top-tier model `x`, shared by all clients |
| Global round | One iteration of the outermost loop, counted by `--num_global_iters` |
| Gradient | The direction of steepest increase of the loss. Training steps move against it |
| Heterogeneity | The degree to which clients' data distributions differ. Measured here by Jensen-Shannon divergence |
| Jensen-Shannon divergence (JSD) | A symmetric measure of difference between two probability distributions, bounded in [0,1]. Used here on client label distributions |
| Local step | One minibatch gradient step on a client, counted by `--local_iters` |
| Macro F1 | Unweighted mean of per-class F1 scores. Each class counts equally regardless of size |
| Macro recall | Unweighted mean of per-class recall. The detection rate treating all classes equally |
| Majority-class floor | The score a classifier achieves by always predicting the most common class. The minimum any useful model must beat |
| Minibatch | A small subset of a client's data used for one gradient step |
| Moreau envelope | A smoothed version of a function, defined as the minimum of the function plus a quadratic penalty. Its gradient is `λ(w − prox(w))`, which is where λ's second role originates |
| Non-IID | Not independent and identically distributed. Clients' data differ in distribution |
| Overfitting | Continuing to improve on training data while getting worse on unseen data. Both personalised models here peak and then decline |
| Paired t-statistic | A test of whether a difference measured on matched pairs is distinguishable from zero. Compared against a critical value set by the number of pairs |
| Parameter | A number learned during training. A model is a vector of them |
| Partition | The rule assigning data to clients |
| Personalised model (PM) | A client's own model `θ`, trained on its data and pulled toward its team |
| Personalisation | Allowing each client a different model rather than one shared model |
| Proximal operator | The exact minimiser of a function plus a quadratic penalty. Approximated here by gradient steps |
| Proximal term | The quadratic penalty itself, `(λ/2)‖θ − w‖²`, which keeps a client near its team |
| Quadratic penalty | A cost proportional to the squared distance between two parameter vectors |
| Recall | Of the samples truly in a class, the fraction found. The detection rate |
| Seed | The number initialising the random number generator. Fixing it makes a run repeatable |
| Split-λ | This project's variant: the team update's coefficient is a separate parameter rather than inheriting λ |
| Statistical heterogeneity | See Heterogeneity |
| Structural ceiling | The highest macro F1 reachable given that a client cannot predict a class it never trained on |
| Team | A group of clients sharing an intermediate model `w`. PerMFL's middle tier |
| Team round | One iteration of the middle loop, counted by `--num_team_iters` |
| Temporal split | Dividing train from test by timestamp rather than at random, so no future data leaks into training |
| True negative | A sample correctly identified as not belonging to a class. The cell macro F1 ignores |
| Zero-day | An attack unseen during training. Approximated here by cross-test |
