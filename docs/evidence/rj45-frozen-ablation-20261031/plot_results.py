"""Render the audited, recorded ablation matrix without running a policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

from audit_results import run_audit


def main() -> None:
    root = Path(__file__).resolve().parent
    load = lambda name: json.loads((root / name).read_text(encoding="utf-8"))
    audit, summary = run_audit(root)
    report, contract = load("results-report.json"), load("contract.json")
    task_path = root.parents[2] / "configs/tasks/rj45-v1.json"
    protocol = load("invalid-attempt-a/contract.json")
    packet_protocol_sha = contract["origin"]["protocol_sha256"]
    if packet_protocol_sha != protocol["origin"]["protocol_sha256"]:
        raise ValueError("repair changed the original protocol")
    origin_protocol = json.loads((root.parent / "rj45-oracle-condition-20261031/protocol-20261128.json").read_text())
    task_sha = hashlib.sha256(task_path.read_bytes()).hexdigest()
    if task_sha != origin_protocol["frozen_inputs"]["task_config_sha256"]:
        raise ValueError("task threshold source changed")
    threshold = json.loads(task_path.read_text())["evaluation"]["max_lateral_error_m"] * 1e6
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    modes = contract["mode_order"]
    labels = ["학습된 PPO", "무보정", "상수 보정", "기록 동작 재생"]
    cases = contract["case_order"]
    entries = {(e["case"]["case_id"], e["ablation_mode"]): e for e in report["episodes"]}
    fig, ax = plt.subplots(figsize=(11.8, 6.3), layout="constrained")
    data = []
    for row, case_id in enumerate(cases):
        for col, mode in enumerate(modes):
            episode = entries[case_id, mode]
            terminal = episode["evaluation"]["terminal"]
            success = terminal["success"]
            error_um = terminal["metrics"]["lateral_error_m"] * 1e6
            failure_labels = {"depth": "깊이", "lateral": "정렬", "orientation": "방향", "force": "하중", "forbidden_contact": "접촉", "grasp_live_and_proven": "파지"}
            reasons = [label for key, label in failure_labels.items() if terminal["criteria"][key] is False]
            status_label = "성공" if success else "실패 · " + ("·".join(reasons) or "유지")
            ax.add_patch(Rectangle((col, row), 1, 1, facecolor="#e5f4eb" if success else "#f9e9e7", edgecolor="white", linewidth=3))
            ax.text(col + .5, row + .40, status_label, ha="center", va="center", color="#16613d" if success else "#a53b30", fontsize=13, weight="bold")
            ax.text(col + .5, row + .72, f"{error_um:.2f} µm", ha="center", va="center", fontsize=11, color="#344054")
            data.append({"case_id": case_id, "mode": mode, "success": success, "terminal_lateral_error_um": error_um, "failed_criteria": [key for key, passed in terminal["criteria"].items() if passed is False], "terminal_metrics": terminal["metrics"], "trace_sha256": episode["evaluation"]["trace"]["sha256"]})
    ax.set_xlim(0, 4)
    ax.set_ylim(5, 0)
    ax.set_xticks([i + .5 for i in range(4)], labels, fontsize=12)
    ax.xaxis.tick_top()
    ax.set_yticks([i + .5 for i in range(5)], ["학습에 쓴 조건", "새 보정 오차 1", "새 보정 오차 2", "새 보정 오차 3", "새 보정 오차 4"], fontsize=11)
    ax.tick_params(length=0, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle("고정한 PPO와 비학습 보정의 물리 비교", fontsize=17, weight="bold")
    fig.supxlabel(f"숫자는 종료 횡오차 · 프로젝트 기준 {threshold:g} µm 외에도 파지·깊이·하중·안정 유지로 판정\n동일 train 파지 1개, 학습 seed 1개 · 탐색 진단이며 강화학습 개선 효과의 입증은 아님", fontsize=10, color="#475467")
    fig.savefig(root / "comparison.png", dpi=180, facecolor="white")
    fig.savefig(root / "comparison.svg", facecolor="white")
    svg_path = root / "comparison.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)
    paired = {}
    for baseline in modes[1:]:
        groups = {"ppo_only_success": [], "baseline_only_success": [], "both_success": [], "both_failure": []}
        for case_id in cases[1:]:
            ppo = entries[case_id, "standard79"]["evaluation"]["terminal"]["success"]
            base = entries[case_id, baseline]["evaluation"]["terminal"]["success"]
            key = "both_success" if ppo and base else "ppo_only_success" if ppo else "baseline_only_success" if base else "both_failure"
            groups[key].append(case_id)
        paired[baseline] = {"case_count": 4, **groups}
    payload = {"exploratory_only": True, "effect_proven": False, "report_sha256": hashlib.sha256((root / "results-report.json").read_bytes()).hexdigest(), "task_sha256": task_sha, "lateral_limit_um": threshold, "by_mode": summary["mode_results"], "paired_transfer": paired, "entries": data}
    (root / "comparison-data.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
