# What this document contains

The existing draft in `SCMoE/report/draft` covers Introduction, Literature
Review, Design, Implementation, Evaluation and Conclusion. Three parts of the
eight-chapter structure have no counterpart there and are supplied here.

| eight-chapter structure | in the existing draft? | supplied here |
|---|---|---|
| 1 Introduction | yes, chapter 1 | no |
| 2 Literature Review | yes, chapter 2 | no |
| 3 Requirements and ethics | only a short Requirements section inside Design | YES, in full |
| 4 Design | yes, chapter 3 | no |
| 5 Implementation | yes, chapter 4 | no |
| 6 Testing and verification | absent entirely | YES |
| 7 Critical and qualitative evaluation | chapter 5 presents results but has no critical section | YES, the critical part only |
| 8 Conclusion and further work | yes, chapter 6, including Limitations and Further work | no |

The existing chapter 2 already carries a "Legal, social, ethical and
professional context" section. The ethics material below is longer and
project-specific, so it either replaces that section or absorbs it, whichever
the supervisor prefers.

# Requirements, Ethics and Legal Position

This chapter states what the software had to do, the constraints it operated
under, and the ethical and legal position of the work.

## Functional requirements

  
- **F1 Reproduce the base method.**  Run PerMFL unmodified on the authors'
    own dataset and compare against the published figure. Without this, no
    later result can be attributed to a change rather than to a broken
    implementation.
  
- **F2 Load intrusion detection data.**  Parse CICIDS2017, NSL-KDD and
    TON-IoT into the four-tuple interface the existing loaders expose, so that
    every algorithm in the codebase runs on them without modification.
  
- **F3 Partition clients under controlled heterogeneity.**  Support
    attack-domain, label-skew, Dirichlet and host-based partitioning, with the
    degree of skew settable and measurable.
  
- **F4 Report metrics appropriate to imbalanced detection.**  Macro-averaged
    F1, precision, recall and false positive rate, per class and pooled, in
    addition to the accuracy the base implementation reports.
  
- **F5 Support repeated runs.**  Vary model initialisation and partitioning
    by seed so that variation across runs can be measured.
  
- **F6 Derive teams from client updates.**  Form teams during training from
    the client state rather than at load time, since a data-derived grouping
    cannot be computed before any client has trained.
  
- **F7 Evaluate on unseen classes.**  Support evaluation of clients on
    classes absent from their training data.

## Non-functional requirements

  
- **N1 Reproducibility.**  A run must be reproducible from a recorded seed
    and command. Package versions are pinned to the period of the base paper's
    publication.
  
- **N2 No information leakage.**  No statistic fitted on evaluation data may
    influence training. Scaling parameters, clipping bounds and imputation
    values are fitted on the training split only.
  
- **N3 Commodity hardware.**  Experiments must complete on a laptop CPU. No
    result may depend on access to a GPU cluster.
  
- **N4 Backward compatibility.**  Every modification defaults to the released
    behaviour, so results obtained before a change remain reproducible after
    it.
  
- **N5 Traceability.**  Each defect identified in the base implementation is
    recorded with a file and line reference and, where its effect is
    measurable, a measurement.

## Ethical position

### Data

The project uses three public datasets released for research. CICIDS2017 and
NSL-KDD were produced by the Canadian Institute for Cybersecurity, TON-IoT by
UNSW Canberra. All three were captured in purpose-built testbeds using
synthetic user profiles and scripted attacks. No traffic from a real
organisation or an identifiable individual is involved, and no attempt is made
to re-identify any host.

The IP addresses in the captures belong to testbed machines. They appear in the
raw files and are used in one partitioning scheme to define client boundaries,
which reflects how a deployed system would be organised. They are removed
before the feature matrix is constructed, both to prevent the model learning
addresses instead of behaviour and because address-derived features do not
transfer outside the testbed.

### Ethical approval

The project involves no human participants, no personal data as defined by UK
GDPR, and no interaction with live systems. Approval was obtained under
reference [TO SUPPLY: EthOS number].

### Dual use

Intrusion detection research is defensive, but any work that characterises
where a detector fails also indicates where it could be evaded. The findings
here concern the internal parameterisation of a federated learning method
rather than exploitable weaknesses in a deployed system, and the datasets are
public and well studied. The risk is low. No detection evasion technique is
developed or evaluated.

### Reporting

Results are reported whether or not they support the hypothesis. The evaluation
in Chapter~(see above) includes a configuration in which the proposed
correction makes matters worse, and a claim from an earlier stage of the work
that later measurement contradicted. Both are retained.

## Legal position

The base implementation is released by its authors under the MIT licence. This
work is a derivative and carries the same licence with attribution preserved.
Dataset licences permit academic use with citation, and all three are cited.
No dataset is redistributed; the repository contains loaders and a download
procedure.

## Scope

Three exclusions are deliberate.

No comparison against SCMoE-PFL or CFMD-i is attempted. Neither publishes an
implementation, and comparing against a reimplementation would measure the
reimplementation. This also means no claim is made about relative computational
efficiency, since establishing one would require running both.

Differential privacy and secure aggregation are out of scope. The threat model
here is statistical heterogeneity, not an adversarial server.

Attack detection is treated as multi-class classification over labelled
traffic. Unsupervised anomaly detection, as in Fed-ANIDS, is a different
problem.

# Testing and Verification

Nothing in this project is user-facing, so testing means establishing that the
pipeline produces what it claims and that measured differences come from the
change under test. This chapter records how each part was verified and which
checks caught real errors.

## Verifying the reproduction

The first test was whether the unmodified implementation reproduces its
published result. Running it on EMNIST-10 for four hundred global rounds gave a
peak personalised accuracy of 96.45 per cent against the 96.49 reported in the
paper, a difference of 0.04 percentage points.

This matters more than a sanity check. The run carries the argument binding
defect, since it is present in the released code, and it still lands on the
published figure. That is evidence the defect was present when the paper's
results were produced, and it licenses treating later measurements as changes
to the method rather than symptoms of a broken build.

## Loader invariants

Each loader is checked against six invariants before use.

No row appears in both the training and evaluation split for a client. Every
class present in the corpus appears in both the pooled training and pooled
evaluation sets. The pooled training features have approximately zero mean and
unit standard deviation, confirming the scaler was fitted and applied. No
non-finite values survive. The per-client, per-class split ratio lies within
0.70 and 0.80 of the intended 0.75. No client is empty and no team is empty.

The team check exists because it caught a real defect. One loader's sequential
assignment hardcodes its group boundaries to two specific client indices rather
than computing them, which at forty clients produces groups of eleven, twenty-six
and three and leaves the fourth team with no members. A team with no members
divides by zero when its sample proportion is computed.

## Errors the checks caught

Four errors reached a run before being caught, and the way each surfaced is
worth recording.

### A split that produced empty test classes

The first temporal split ordered each client's rows by time and cut at
seventy-five per cent. Attacks in CICIDS2017 occupy contiguous windows, so this
gave the evaluation set whichever attack ran last and placed the others entirely
in training. Wednesday clients trained on 2,913 denial-of-service samples and
were evaluated on none.

The symptom was not an exception. It was that the personalised models scored
0.45 accuracy where a classifier that always predicts benign scores 0.82. A
model below the majority-class baseline indicates a setup fault rather than a
weak method, and that comparison is the check that caught it. The correction is
to split each class on its own timeline.

### A control flow error that passed the syntax check

A guard inserted between a conditional branch and its trailing `else`
turned the construct into a `for`-`else`, which is valid Python.
The `else` then bound to the loop and raised on every call. Eight queued
runs each exited in three seconds. The parser had accepted the file.

The lesson recorded here is that a syntax check is not an execution check. Every
subsequent queue was preceded by a single short run of the same configuration.

### An undefined name from statement ordering

A module-level assignment reading an environment variable was inserted above the
import that provides the module it calls. Forty-six queued runs failed
identically. Again the file parsed.

### A metric measured against a moving target

Benign thinning was initially applied before the train and test split, so it
thinned both. The accuracy floor moved from 0.8171 to 0.7008, meaning
comparisons across thinning settings sat on different evaluation targets. The
correction, following the reference implementation, applies thinning to the
training split only, leaving the evaluation set at its natural class balance.
The floors are now constant across settings.

## Verifying the modifications

Each modification was verified by measuring the quantity it was supposed to
change.

For the argument binding fix, the optimiser's penalty weight was read directly
before and after: twenty before, matching the local iteration count, and the
supplied value after. The per-step pull changed from 0.2 to 0.05 as predicted.

For seeding, an identical configuration was run at two seeds and the results
compared. Before the change they were identical to four decimal places; after
it they differed.

For the synthetic split fix, the split was hashed across three separate
processes. Before the change the three hashes differed; after, they matched.

The clustering module was tested against synthetic data with known group
structure: twenty points drawn from five well-separated latent clusters. It
recovered them at an adjusted Rand index of 0.762 with equal group sizes, and
the trigger fired on separated data while holding off on near-identical data.

## Establishing what counts as a difference

A measurement is only interpretable against the variation of the measurement
itself. Two runs of an identical configuration, differing only in their
position in the experiment schedule, gave personalised macro F1 of 0.7318 and
0.7244, a difference of 0.0074.

Differences below roughly 0.01 are therefore not distinguishable from
run-to-run variation, and a threshold of 0.03 was adopted before results were
examined. Several comparisons fall below it and are reported as null rather than
as small effects. The largest such is 0.0016, which is an order of magnitude
inside the floor.

## What was not tested

There is no unit test suite. The base implementation ships none, and adding one
would have meant characterising behaviour the project set out to question.
Verification here is at the level of measured invariants and reproduced
published figures, which suits an experimental study but would not suit
software intended for use.

The CUDA path is untested. All experiments ran on CPU and no GPU was available.
The device selection logic is exercised only in its CPU branch.

# Critical and qualitative evaluation

This is section 7.6 of the alternate structure. It is written to follow a
results presentation, so it slots after the existing chapter 5 either as a
closing section of that chapter or as a short chapter of its own.



### What the work establishes

The reproduction is exact and the defect catalogue is verifiable line by line,
so neither depends on trusting the analysis. The tier finding is supported
across three datasets and four team counts with a mechanism derived from the
update rule rather than inferred from outcomes, and the mechanism predicts the
regression in Section~(see above) as well as the improvements. The
decoupling result is paired, seeded and significant across twelve
configurations.

### What it does not establish

The comparison against other methods is absent. Neither SCMoE-PFL nor CFMD-i
publishes an implementation, so no relative claim about accuracy or efficiency
is made. Runs completed in minutes on one CPU core where a mixture-of-experts
implementation required hours on a GPU, but the client counts, datasets and
software stacks differ, so that observation is structural rather than measured.

The baselines released with the base implementation were not run on the intrusion
detection data. Only the method under test carries the metric instrumentation
this study requires, and extending it to the others was not completed. The
consequence is that the results position PerMFL against its own configurations
and against local training, not against the federated literature.

Several exploratory results rest on single runs. The local depth and
communication sweeps are reported as exploratory for that reason, though their
effects, up to 0.29 macro F1, are far outside the noise floor.

The clustering signal is parameter-space only. CFMD-i additionally uses an
output-space divergence, which is likely the better signal under label skew,
since two clients can hold near-identical weights and behave differently. Given
that the term the clustering feeds is weighted at 3.3 per cent, a better signal
would probably not change the conclusion, but that is an argument rather than a
measurement.

### Threats to validity

The heterogeneity measure is the divergence between label distributions, which
does not capture feature-space or concept differences. A partition producing
identical label distributions with different decision boundaries would register
as homogeneous.

The structural ceiling calculation assumes a client can only predict classes it
trained on. A model that generalises across classes could exceed it, so the
bound is conservative rather than exact.

The per-client cap flattens client sizes to a narrow band, suppressing quantity
skew that real deployments exhibit. The host-based partition is the exception,
with client sizes spanning 1,984 to 37,832 rows.

Two datasets share an origin, and CICIDS2017 and NSL-KDD are both laboratory
captures with scripted attacks. Agreement between them is weaker evidence than
agreement between independent sources would be.
