import argparse
import json
from pathlib import Path

from supportbench.baselines import POLICIES
from supportbench.cases import case_ids
from supportbench.replay import record_script, replay_trace, trace_json


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="supportbench", description="Run or replay deterministic fictional cases"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("run", help="Record one scripted demonstration")
    run.add_argument("--case", choices=case_ids(), default="lamp-ring")
    run.add_argument("--policy", choices=POLICIES, default="context")
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--output", type=Path)

    compare = subcommands.add_parser("compare", help="Compare toy scripts on each case")
    compare.add_argument("--seed", type=int, default=7)

    replay = subcommands.add_parser("replay", help="Verify a recorded JSON action trace")
    replay.add_argument("trace", type=Path)

    args = parser.parse_args()
    try:
        if args.command == "run":
            text = trace_json(
                record_script(args.policy, case_id=args.case, seed=args.seed)
            )
            if args.output:
                args.output.write_text(text, encoding="utf-8")
                print(f"Recorded {args.output}")
            else:
                print(text, end="")
        elif args.command == "compare":
            rows = []
            for case_id in case_ids():
                for policy in POLICIES:
                    trace = record_script(policy, case_id=case_id, seed=args.seed)
                    final = trace["steps"][-1]
                    rows.append(
                        {
                            "case_id": case_id,
                            "policy": policy,
                            "reward": final["reward"],
                            "vector": final["info"]["reward_vector"],
                            "metrics": final["info"]["metrics"],
                        }
                    )
            print(json.dumps(rows, indent=2))
        else:
            trace = json.loads(args.trace.read_text(encoding="utf-8"))
            replay_trace(trace)
            print(f"Replay matched {len(trace['steps'])} actions for {trace['case_id']}.")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"supportbench: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
