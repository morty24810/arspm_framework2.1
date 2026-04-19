import json
from pathlib import Path

from run_experiment import main


def _load_baseline_summary(output_root: Path) -> dict:
    summary_path = Path(output_root) / "rule_coverage_baselines_v3" / "scenario_family_baseline_summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    baseline_output_root = Path(main(profile_override="thesis_sched_rule_coverage_baselines_v3"))
    baseline_summary = _load_baseline_summary(baseline_output_root)
    if not bool(baseline_summary.get("baseline_gate_passed", False)):
        print(
            "coverage v3 baseline gate failed; skipping PPO coverage training. "
            f"baseline outputs: {baseline_output_root}"
        )
    else:
        main(profile_override="thesis_ppo_sched_rule_coverage_v3")
