# SupportBench

SupportBench starts with a familiar support problem: someone comes back, and the
agent behaves as though their first visit never happened. Ignored history, a
wrong record, or a context-free transfer can turn one request into several.
This project gives those return visits a small, inspectable world to live in.

**Fictional vignette.** A customer opens a case because the diffuser ring is
missing from a GroveDesk lamp. They give reference `DEMO-L-041` and explain that
they have checked the packing. On the next contact, they ask whether a spare
has been queued. The agent can inspect the existing case instead of requesting
the reference again, consult the product guide, and queue the part. Saying
“resolved” without that recorded action fails the episode. A different case
has conflicting tracking records and an explicit request for a person to
review them; a useful handoff carries the earlier details forward.

The **first public starter** is runnable with Python 3.12 and the standard
library. It contains three authored cases across two invented products, with
two or three scheduled contacts per case. Customer disclosures, questions,
case history, and simulated satisfaction and frustration persist between
contacts **within** an episode; `reset` gives the next case fresh state. Agents
use structured `inspect_case`, `search_guide`, `ask_customer`, `update_case`,
`handoff`, and `respond` actions. A complete human handoff can change the
record; an incomplete one returns for context. Messages and responses are
fixed or structured rather than free-form LLM dialogue.

The scorer checks the simulated record for a specific service event: a spare
queued, a guide dispatched, or a tracking row reconciled. It does not equate
an agent's claim or a closed-status label with that event. Each step exposes a
reward vector for verified resolution, satisfaction change, frustration,
repeated effort, time, cost, and handoff quality. A scalar with **illustrative,
uncalibrated** weights is emitted when the episode ends. These customer-state
numbers are simulation rules, not measured human satisfaction.

## Run an episode

```sh
python3.12 -m supportbench run --case lamp-ring --policy context --seed 7 --output episode.json
python3.12 -m supportbench replay episode.json
python3.12 -m supportbench compare --seed 7
python3.12 -m unittest discover -s tests -v
```

GitHub Actions installs the package and runs the same offline tests on pushes
and pull requests.

`compare` runs three authored sanity-check scripts: context-aware, memoryless,
and always-handoff. These are demos, not model results. The JSON trace records
every action and observation, the seed, and the case, scorer, and script
versions; replay checks the trajectory. See the
[scoring contract](docs/scoring.md), [research and
boundaries](docs/research.md), and [episode-loop diagram with a text
alternative](docs/episode-loop.html).

Next: held-out cases, weight sensitivity, and human calibration of the
proxies. For now the cases, products, and guidelines are original fiction;
their [provenance](data/PROVENANCE.md) is documented. The code is MIT-licensed.
