# Scoring contract and evaluation risks

This is the **starter contract**, not a calibrated measure of customer
experience. `cases-v1` identifies the authored scenarios and seed-driven
phrasing/patience; `scorer-v1` identifies the outcome oracle, proxy updates,
normalization, and scalar formula. `trace-v1` identifies the JSON action-trace
shape. `scripts-v1` identifies the bundled scripted policies. A trace records
its case, seed, policy label and version (or `custom` / `custom-v1`), reset
observation, every action, every result, and the generator/scorer/trace
versions. Replay rejects a changed generator, scorer, or trace format, then
reproduces and compares each step. A policy label describes the recorded run;
replay checks actions and outcomes, not the provenance of an external agent.

## What counts as resolved

The oracle reads the simulated **world record**, never a response string or
the mutable `status` label. It requires the matching case ID and one persisted
service event:

| Case | Required world state | Authorized agent route |
| --- | --- | --- |
| `lamp-ring` | `fulfillment = part_queued` | Inspect case and guide, then queue a part. |
| `fan-setup` | `guide_delivery = full_guide_sent` | Inspect case and guide, then dispatch the guide. |
| `lamp-tracking` | `tracking = reconciled` | Human queue only; the customer requests review. |

These are simulated service outcomes, **not** evidence that a physical part
arrived, a customer understood the guide, or a tracking discrepancy was fixed
in a real system. A valid `status = closed` update is optional and only allowed
after the oracle passes. An unauthorized write, out-of-scope read/question or
handoff, unverified closure, or structured `Respond("resolved")` before the
oracle passes terminates the episode without changing the forbidden record.

## Vector and terminal scalar

`info.reward_vector` is present after reset and after every step, including a
failure. It exposes the following named values; `info.metrics` also reports
raw turns, elapsed units, cost units, repeated requests, and handoffs.

| Component | Definition |
| --- | --- |
| `verified_resolution` (Q) | 1 if the world-state oracle passes, otherwise 0. |
| `satisfaction_change` (S) | Current scripted satisfaction minus its initial 0.5, bounded to [-0.5, 0.5]. |
| `frustration` (F) | Scripted customer frustration, bounded to [0, 1]. |
| `repeated_effort` (B) | `min(repeated_requests / scheduled_contacts, 1)`. Asking for an already disclosed fact counts; a deficient handoff also makes the customer repeat context. |
| `elapsed_time` (T) | `min(elapsed_units / (8 * scheduled_contacts), 1)`. Each action costs one time unit, each transition two, and a human queue adds three. |
| `cost` (C) | `min(cost_units / (10 * scheduled_contacts), 1)`. Inspect, search, and ask cost one; update costs two; handoff costs six. |
| `handoff_quality` (H) | Mean of +1 for a complete, needed handoff and -1 for an unnecessary or incomplete one; 0 if unused. |

The customer's initially stated reference and previous effort count as
disclosed. Asking about a new preference discloses it; asking again counts as
repetition. A repeat subtracts 0.08 from satisfaction and adds
`0.18 / patience` to frustration. A working response adds 0.02 satisfaction;
a verified resolved response adds 0.15. A complete, needed handoff adds 0.12
satisfaction; an unnecessary one subtracts 0.08 and adds 0.08 frustration.
An incomplete handoff subtracts 0.15, adds `0.22 / patience` frustration, and
one repeat. A policy violation subtracts 0.25 satisfaction and adds 0.30
frustration. Patience is one of three seeded, identity-free values; proxies
are capped to [0, 1] and rounded to three decimals.

Intermediate step rewards are zero. At the end, the **illustrative** scalar is

`R = 0.40Q + 0.15S - 0.10F - 0.10B - 0.10T - 0.10C + 0.05H`.

Policy violations override it with `R = -1`; exhausting the action budget
(`6 * scheduled_contacts`) without completing the contacts yields a distinct
truncation with `R = -0.5`. The oracle and full vector remain visible even on
those endings. Closing a contact through a structured response or handoff
advances to the next scheduled contact; the last closes the episode.

## Versioning and limitations

Change `cases-v1` when prompts, records, seeded traits, or case policies
change. Change `scorer-v1` when goal checks, proxy updates, limits, weights, or
termination semantics change. Change `trace-v1` when serialization changes.
Change `scripts-v1` when a bundled policy changes.
Publish comparisons with the exact versions, case IDs, seeds, action traces,
and policy implementation version; do not combine scores across incompatible
contracts. The bundled scripts are example controls with authored knowledge
of the three scenarios, not independent agent evaluations.

Public deterministic cases invite overfitting. Before any serious comparison,
reserve recombined holdout cases, audit each goal for multiple valid routes,
report each vector component and quality–cost tradeoffs, and vary the scalar
weights. Exact structured summary fields simplify this starter but can
penalize a useful differently phrased transfer. Human ratings should test
whether the satisfaction/frustration proxies reflect actual effort; any
future dialogue judge needs blinded calibration and bias checks. Keep
real customer data and proprietary playbooks out of this test world.
