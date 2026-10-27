#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/rby1-assembly-lab"
export OMNI_KIT_ACCEPT_EULA=YES
run_dir="artifacts/rj45-paper-ablation"
mkdir -p "$run_dir"
if [[ -e "$run_dir/execution-status.log" ]]; then
  printf 'Refusing to overwrite an existing experiment status log\n' >&2
  exit 2
fi
status_log="$run_dir/execution-status.log"
python_bin="$HOME/rby1-assembly-lab/.venv-isaac/bin/python"
nominal="$run_dir/nominal.json"
held="$run_dir/held.json"
contact_proof="artifacts/rj45-parallel-probe/after-persistent-target-hook/evidence-spec.json"
a0="configs/rj45-training-target-offset-v1.json"
a1="configs/rj45-training-target-offset-a1-discounted-pose-v2.json"
a2="configs/rj45-training-target-offset-a2-keypoint-shape-v2.json"
ppo="configs/ppo-rj45-v1.json"
ppo_low="configs/ppo-rj45-low-exploration-v1.json"
run_step() {
  local label="$1"
  shift
  local log_path="$run_dir/$label.log"
  printf '%s START %s\n' "$(date -u +%FT%TZ)" "$label" | tee -a "$status_log"
  if /usr/bin/time -f 'wall_seconds=%e exit_status=%x' "$@" >"$log_path" 2>&1; then
    printf '%s PASS %s\n' "$(date -u +%FT%TZ)" "$label" | tee -a "$status_log"
  else
    local result=$?
    printf '%s FAIL %s exit=%s\n' "$(date -u +%FT%TZ)" "$label" "$result" | tee -a "$status_log"
    tail -60 "$log_path" >&2
    return "$result"
  fi
}
run_step nominal "$python_bin" scripts/run_sim.py \
  --headless --device cuda:0 --task rj45 --num-envs 1 --require-completion \
  --trace-every 1 --report "$nominal" --trace "$run_dir/nominal.jsonl" \
  --rj45-held-state-output "$held"
run_step reset_n1 "$python_bin" scripts/check_rj45_reset.py \
  --headless --device cuda:0 --num-envs 1 --reset-cycles 5 \
  --nominal "$nominal" --held "$held" --report "$run_dir/reset-n1.json"
run_step reset_n4 "$python_bin" scripts/check_rj45_reset.py \
  --headless --device cuda:0 --num-envs 4 --reset-cycles 4 \
  --nominal "$nominal" --held "$held" --parallel-contact-evidence "$contact_proof" \
  --report "$run_dir/reset-n4.json"
run_step reset_n64 "$python_bin" scripts/check_rj45_reset.py \
  --headless --device cuda:0 --num-envs 64 --reset-cycles 4 \
  --nominal "$nominal" --held "$held" --parallel-contact-evidence "$contact_proof" \
  --report "$run_dir/reset-n64.json"
run_step smoke_a1 "$python_bin" scripts/train.py \
  --headless --device cuda:0 --task rj45 --num-envs 1 --max-iterations 1 \
  --config "$ppo" --rj45-training-config "$a1" \
  --rj45-nominal-evidence "$nominal" --rj45-held-state "$held" \
  --log-root "$run_dir/smoke/a1"
run_step smoke_a2_n64 "$python_bin" scripts/train.py \
  --headless --device cuda:0 --task rj45 --num-envs 64 --max-iterations 1 \
  --config "$ppo" --rj45-training-config "$a2" \
  --rj45-nominal-evidence "$nominal" --rj45-held-state "$held" \
  --parallel-contact-evidence "$contact_proof" --log-root "$run_dir/smoke/a2"
run_step smoke_z1 "$python_bin" scripts/train.py \
  --headless --device cuda:0 --task rj45 --num-envs 1 --max-iterations 1 \
  --config "$ppo_low" --rj45-training-config "$a0" --zero-init-actor-output \
  --rj45-nominal-evidence "$nominal" --rj45-held-state "$held" \
  --log-root "$run_dir/smoke/z1"
for smoke in a1 a2 z1; do
  mapfile -t smoke_checkpoints < <(find "$run_dir/smoke/$smoke" -type f -name model_0.pt -print)
  if [[ "${#smoke_checkpoints[@]}" -ne 1 ]]; then
    printf 'Expected one smoke checkpoint for %s, found %s\n' "$smoke" "${#smoke_checkpoints[@]}" >&2
    exit 3
  fi
  smoke_profile="$a0"
  smoke_ppo="$ppo"
  smoke_expected="a0_legacy_pose_v1"
  smoke_args=()
  smoke_envs=1
  smoke_episodes=1
  case "$smoke" in
    a1) smoke_profile="$a1"; smoke_expected="a1_discounted_pose_v2" ;;
    a2) smoke_profile="$a2"; smoke_expected="a2_keypoint_shape_v2"; smoke_envs=4; smoke_episodes=4 ;;
    z1) smoke_ppo="$ppo_low"; smoke_args=(--require-zero-init) ;;
  esac
  smoke_contact=()
  if [[ "$smoke_envs" -gt 1 ]]; then
    smoke_contact=(--parallel-contact-evidence "$contact_proof")
  fi
  run_step "smoke_eval_${smoke}" "$python_bin" scripts/evaluate_policy.py "${smoke_checkpoints[0]}" \
    --headless --device cuda:0 --task rj45 --num-envs "$smoke_envs" --episodes "$smoke_episodes" \
    --seed 20261028 --evaluation-start held --baseline-mode learned \
    --config "$smoke_ppo" --rj45-training-config "$smoke_profile" \
    --rj45-nominal-evidence "$nominal" --rj45-held-state "$held" \
    "${smoke_contact[@]}" --report "$run_dir/smoke-${smoke}-evaluation.json"
  run_step "smoke_validate_${smoke}" "$python_bin" "$run_dir/validate_gpu_smoke.py" \
    "$run_dir/smoke-${smoke}-evaluation.json" --expected-profile "$smoke_expected" "${smoke_args[@]}"
done
run_step baseline "$python_bin" scripts/evaluate_policy.py \
  --headless --device cuda:0 --task rj45 --num-envs 16 --episodes 32 \
  --seed 20261028 --evaluation-start held --baseline-mode biased_zero \
  --config "$ppo" --rj45-training-config "$a0" \
  --rj45-nominal-evidence "$nominal" --rj45-held-state "$held" \
  --parallel-contact-evidence "$contact_proof" --report "$run_dir/baseline.json"
for profile in A0 A1 A2 E1 Z1; do
  train_config="$a0"
  policy_config="$ppo"
  zero_flag=()
  case "$profile" in
    A1) train_config="$a1" ;;
    A2) train_config="$a2" ;;
    E1) policy_config="$ppo_low" ;;
    Z1) policy_config="$ppo_low"; zero_flag=(--zero-init-actor-output) ;;
  esac
  run_root="$run_dir/runs/$profile"
  if [[ -e "$run_root" ]]; then
    printf 'Refusing to overwrite existing run root: %s\n' "$run_root" >&2
    exit 2
  fi
  run_step "train_${profile}" "$python_bin" scripts/train.py \
    --headless --device cuda:0 --task rj45 --num-envs 64 --max-iterations 20 \
    --config "$policy_config" --rj45-training-config "$train_config" \
    --rj45-nominal-evidence "$nominal" --rj45-held-state "$held" \
    --parallel-contact-evidence "$contact_proof" --log-root "$run_root" "${zero_flag[@]}"
  mapfile -t checkpoint_paths < <(find "$run_root" -type f -name model_19.pt -print)
  if [[ "${#checkpoint_paths[@]}" -ne 1 ]]; then
    printf 'Expected exactly one final checkpoint in %s, found %s\n' "$run_root" "${#checkpoint_paths[@]}" >&2
    exit 3
  fi
  checkpoint="${checkpoint_paths[0]}"
  run_step "evaluate_${profile}" "$python_bin" scripts/evaluate_policy.py "$checkpoint" \
    --headless --device cuda:0 --task rj45 --num-envs 16 --episodes 32 \
    --seed 20261028 --evaluation-start held --baseline-mode learned \
    --config "$policy_config" --rj45-training-config "$train_config" \
    --rj45-nominal-evidence "$nominal" --rj45-held-state "$held" \
    --parallel-contact-evidence "$contact_proof" --report "$run_dir/${profile}.json"
done
printf '%s COMPLETE development ablation\n' "$(date -u +%FT%TZ)" | tee -a "$status_log"
