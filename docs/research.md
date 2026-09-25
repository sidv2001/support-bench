# Research context

SupportBench tests one deliberately narrow question: can an agent carry
customer context and an actionable case record through **separate contacts**,
including a handoff when the case needs one? These sources informed the
design; none establishes that this combination is novel, and no source code,
dataset, task, or figure was imported.

- [τ-bench](https://arxiv.org/html/2406.12045) studies tool-using agents in
  simulated user conversations and scores resulting state. The current
  [τ³-bench repository](https://github.com/sierra-research/tau2-bench)
  extends shared customer/agent state and revises grading over time. Here an
  episode spans two or three **contacts** before reset, and the generator and
  scorer are versioned so a revision cannot silently change a comparison.
- [ToolSandbox](https://arxiv.org/html/2408.04682) evaluates stateful tool
  trajectories, including forbidden transitions. Our hard failure for
  unauthorized case changes is an original, smaller application of that
  general evaluation concern. Its
  [license](https://github.com/apple/ToolSandbox/blob/c8571d7854316d2e1c5f288e59fe1e34e53f6dd1/LICENSE)
  is not a source of MIT-licensed code for this project.
- [CRMArena](https://arxiv.org/html/2411.02305v2) investigates service tasks
  with synthetic CRM records. Its published
  [CC BY-NC 4.0 license](https://github.com/SalesforceAIResearch/CRMArena/blob/6d84f3d71305af0fd3d5ed3c1936b7887464455a/LICENSE.txt)
  is noncommercial; SupportBench's three case records, product names, and
  guides were written here instead of copied from CRM tasks or data.
- [Gymnasium's environment API](https://gymnasium.farama.org/api/env/)
  motivates seeded `reset` and the distinction between termination and
  time-limit truncation. SupportBench returns observation, reward,
  `terminated`, `truncated`, and `info` from `step`, without depending on or
  claiming compatibility with Gymnasium.
- [Learning to Defer](https://arxiv.org/html/1711.06664) frames a decision
  to pass work to a person in terms of the *whole system's* outcome. A
  transfer here changes the world only when its summary carries sufficient
  context; handoff quality and human time/cost are reported separately.
- [AI Agents That Matter](https://arxiv.org/html/2407.01502) motivates
  reporting cost and reproducibility alongside success. [The
  LLM-as-a-Judge study](https://arxiv.org/html/2306.05685) documents
  judgment biases; a future subjective dialogue metric would need human
  calibration rather than replacing the record-based resolution oracle.

These are design precedents, not measured results for SupportBench. The
present scripts exercise known toy cases. Holdouts, alternative valid goals,
human feedback, sensitivity analysis, and longer-horizon behavior remain
research work; see the [scoring and risk plan](scoring.md).
