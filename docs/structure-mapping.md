# Chapter structure: eight against the handbook's six

The handbook (6G7V0007, Appendix, report structure) prescribes six chapters.
The current draft has eight. Both are defensible; this records the mapping so
the choice can be made deliberately rather than by default.

## What the handbook prescribes

| Chapter | Handbook wording |
|---|---|
| 1 Introduction | aims of the work, brief overview of the remainder |
| 2 Literature Review | relate to previous work, context, technical detail, motivation, and the wider social, ethical, legal and professional context |
| 3 Design | **state the identified requirements** and show design diagrams with full explanations |
| 4 Implementation | describe the work undertaken and the results obtained, small code sections, **discuss testing strategy and show testing details** |
| 5 Evaluation | examine critically the completed work and the results, relative to the original objectives |
| 6 Conclusion | restate the work, summarise findings, acknowledge limitations, suggest further work |

Note that the handbook explicitly places requirements inside Design and testing
inside Implementation. The eight-chapter draft splits both out.

## Mapping

| Draft chapter | Words | Maps to | Action if consolidating |
|---|---|---|---|
| 1 Introduction | 966 | 1 Introduction | unchanged |
| 2 Literature Review | 1,082 | 2 Literature Review | absorb the ethics and legal material from draft ch3 |
| 3 Requirements, Ethics and Legal Position | 752 | 3 Design + 2 Lit Review | requirements to Design, ethics and legal to Literature Review |
| 4 Design | 1,203 | 3 Design | becomes Design section 3.2 onward |
| 5 Implementation | 1,444 | 4 Implementation | unchanged |
| 6 Testing and Verification | 959 | 4 Implementation | becomes Implementation section 4.5 onward |
| 7 Evaluation | 1,933 | 5 Evaluation | unchanged |
| 8 Conclusion and Further Work | 1,075 | 6 Conclusion | unchanged |

Consolidated chapter sizes would be 966, 1,500, 1,955, 2,403, 1,933, 1,075.
Implementation becomes the largest chapter at 2,403 words, which is
proportionate for a project whose contribution is a code change.

## Evidence for six

The exemplar that scored 73 uses exactly the handbook's six chapters, with
critical reflection and limitations as sections **within** Evaluation rather
than as separate chapters. The two lower-scoring exemplars were not checked.

Second readers mark against the grid. A structure that matches the grid needs
no explanation; one that does not spends the reader's attention on navigation.

## Evidence for eight

Splitting testing out of implementation gives the verification work its own
visible chapter, and the verification work here is substantial: six loader
invariants, the reproduction check, and the noise-floor derivation that
establishes what counts as a difference.

Splitting requirements out lets functional, non-functional and ethical
requirements be stated as numbered, testable items, which the Evaluation
chapter then discharges one by one.

## Recommendation

Consolidate to six. The material does not change and no words are lost; only
the heading levels move, so this is a mechanical edit of the markdown source
rather than a rewrite. The gain is that a marker reading against the grid finds
each required element where the grid says it should be.

If your supervisor has already reviewed the eight-chapter structure and
approved it, keep it. Their expectation outranks the handbook's default.
