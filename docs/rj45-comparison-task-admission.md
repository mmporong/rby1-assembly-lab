# RJ45 비교 과제 학습 진입 조건

상태: **G1 Isaac task-admission 3-episode smoke 통과, 전체 90-episode development admission 실행 중**. source `2e1f926` packet으로 classical tuning 18회를 완료해 후보 B를 사전등록 규칙에 따라 선택·봉인했고, admission 72회를 이어서 실행하고 있다. `classical_force_search A`, `zero`, `oracle_gt_reference`가 각 1회씩 terminal contract를 완성한 smoke에서는 hard/forbidden contact가 모두 0회였다. 이는 실행 경로·접촉 telemetry·계약 판정의 작동 확인이다. smoke는 한 bias case만 사용했고 classical은 train-03, zero와 oracle은 train-02에서 실행됐으므로 세 방식의 성능 차이나 학습 효과를 비교하는 근거가 아니다. `task_admission_completed=false`, `rl_pilot_admitted=false`이며 G1은 아직 완료되지 않았다. 근거 요약은 [`docs/evidence/rj45-task-admission-smoke-20261101/README.md`](evidence/rj45-task-admission-smoke-20261101/README.md)에 보존한다.

## 검증 범위

질문은 “같은 기본 제어에서 학습 보정이 위치·각도·파지 상태 변화의 삽입 성공 범위를 넓히면서 과도한 접촉력을 늘리지 않는가”다. G1은 이 질문을 본학습에 넣어도 되는지만 판정한다.

G1의 조작은 물체의 물리 초기 자세를 바꾸는 실험이 아니다. 두 개의 검증된 held-grasp에서 estimator에 episode 고정 calibration bias를 ±X, ±Z 0.12 mm와 ±Rx 0.2°로 넣는다. 이 값은 외부 실측 공차가 아닌 개발 탐색 설계값이며 G3 조건으로 재사용하지 않는다. 여러 오차 크기의 성공 범위 곡선은 G3에서 새 sealed 조건으로 측정한다.

reset 동등성은 warmup 안정성 통과 뒤 `_freeze_training_estimator`를 호출하기 직전의 evaluator 전용 signed GT SE(3)로 확인한다. bias는 reset에서 이미 샘플됐지만 freeze 전에는 제어에 영향을 주지 않는다. 같은 held grasp의 모든 bias·method·slot에서 socket-entry 기준 translation 차이 norm이 10 µm, rotation geodesic 차이가 0.05° 이하여야 한다. 기록 누락이나 초과는 `invalid_contract`다.

두 번째 snapshot은 base `_pre_physics_step`의 `_advance_phases`가 ALIGN→INSERT를 만든 직후, residual action을 물리에 적용하기 전에 기록한다. 이 snapshot은 **같은 grasp·같은 bias case**의 method·slot 사이에서만 translation 10 µm·rotation 0.05° 동등성을 요구한다. 서로 다른 bias case의 INSERT snapshot 차이는 처치가 만든 `realized_treatment_alignment_error`로 기록하며 동일성 gate를 적용하지 않는다. 두 snapshot의 GT는 감사·판정과 oracle 진단에만 사용하며 공정 provider로 돌려주지 않는다.

성공 기준 50 µm, 0.5°, 깊이 12 mm, 0.4초 유지, 25 N은 `configs/tasks/rj45-v1.json`에서 바꾸지 않는다. G1과 기존 공개 결과는 development이며 새 final 평가에 재사용하지 않는다.

## archive와 held-grasp 고정

실행기는 `--runtime-root`로 동결 archive root를 받고 그 아래의 `artifacts/rj45-4c-bank-generation-v2-20261028/held-grasp-bank.json`을 읽는다. 현재 저장소의 같은 상대경로에는 bank가 없으므로 로컬 경로가 있다고 가정하지 않는다.

`load_verified_held_grasp_bank(runtime_root, task, bank_path, "train", allow_final=False)`로 train 8개 전체를 먼저 검증한다. manifest SHA는 `c2fbcfea9528f4a9cefb8c4c63aba6cc561fe2f3095a0fa16b7456dda8da54d5`, `bank_source_contract_sha256(proof)`는 `38cf66d4dea11fe4253021fa57e3c0b5347974dda7490ddb9b7890e6819098d0`이어야 한다. `proof.selected_grasp_ids`가 `rj45-train-01`부터 `08`까지 순서대로인지 확인한 뒤 ID로 index를 찾는다. `train-02 → 1`, `train-05 → 4`가 아니면 중단한다. 각 case는 해당 `bank[index]`를 `held_state`로 factory에 넘긴다.

legacy 단건 `held_state`에는 grasp ID telemetry가 없으므로 `rby1_assembly.rj45_task_admission_env.make_rj45_task_admission_env_type`가 `applied_held_grasp_id`, `applied_held_grasp_index`, `applied_held_artifact_sha256`을 환경 telemetry에 넣는다. runner는 이 값과 factory closure에 넘긴 `bank[index]`의 canonical artifact SHA를 reset마다 대조한다. 같은 wrapper가 요청한 6D bias를 `_training_translation_bias_entry`, `_training_rotation_bias_entry`에 넣고 적용값을 절대오차 `1e-9` 안에서 대조한다. telemetry가 없거나 불일치하면 결과 실패가 아니라 `invalid_contract`다.

## 공정한 제어 권한

공정 provider의 유일한 외부 입력은 `observation79`와 provider가 이전 호출에서 만든 내부 state다. env 객체, `extras`, evaluator GT, calibration bias, simulator object pose에 접근하면 무효다. case ID와 부호는 runner만 보유한다. `rby1_assembly.rj45_comparison_providers.make_fair_provider`가 `zero`, `mean_constant`, `open_loop_playback`, `classical_force_search`를 만들며, 인터페이스는 `reset() -> state`, `act(observation79, state) -> (normalized_right6, next_state)`다. feasibility oracle은 runner가 `rby1_assembly.rj45_teacher_runtime.make_rj45_teacher_env_type`로 별도 구성하고 외부 action은 0만 받는다. 공정 provider API와 성과표에 포함하지 않는다.

모든 공정 방식은 79D simulation proxy와 INSERT 단계 오른손 6D tip offset만 사용한다. 위치 한도 1.5 mm, 회전 한도 1.5°이며 왼팔 지지·삽입 진행·단계 전환·과부하 중단은 같은 스크립트다. invalid proxy는 GT로 대체하지 않고 실패 분모에 남긴다.

base `_pre_physics_step`은 `_advance_phases` 뒤 action을 적용한다. 직전 ALIGN observation은 policy inactive이므로 최초 유효 INSERT observation이 provider에 전달되기 전의 전환 step에서는 모든 공정 provider의 **물리 적용 action이 0**이어야 한다. `make_rj45_task_admission_env_type`의 transition snapshot hook은 phase 전환 뒤 action 적용 전에 실행하고, requested·bounded·actual-applied 6D, actual tip-target offset과 zero 판정을 남긴다. actual-applied action과 tip-target offset이 모두 정확히 0이어야 한다. comparison 전용 physical mask는 `current active & cached_last_policy_observation_valid & estimator_valid`다. training schema 1의 `_executed_policy_transition_mask`는 valid step count 조건 때문에 최초 유효 INSERT observation 다음 action까지 버릴 수 있어 physical mask로 재사용하지 않는다. CPU 계약은 ALIGN→INSERT 전환 action이 0인 경우와 최초 유효 INSERT observation 바로 다음 action이 정상 적용되는 경우를 모두 통과했다. 실제 Isaac 실행에서 같은 telemetry를 다시 판정해야 한다.

- `zero`: 항상 0.
- `mean_constant`: 기존 development trace의 평균 6D 한 개.
- `open_loop_playback`: 같은 development trace 한 개를 activation step에 맞춰 재생한다. 기본 제어의 피드백은 유지된다.
- `classical_force_search`: 아래 수치 계약을 가진 baseline 후보. 독립 tuning·봉인 전에는 “강한 고전 제어”로 부르지 않는다.
- `oracle_gt_reference`: evaluator GT 상대 SE(3)를 받는 물리 가능성 진단이며 공정 baseline이 아니다.

`oracle_gt_reference`는 기존 `make_rj45_teacher_env_type`의 privileged target 경로를 feasibility 전용으로 사용하며 INSERT 전에는 0을 출력한다. INSERT의 committed command depth마다 실제 socket pose로 `G_true`를 만들고, 실제 plug→right-EE transform `F_actual`과 frozen 추정 transform `F_hat`으로 virtual body target `G_true · F_actual · F_hat⁻¹`을 계산한다. `teacher_tip_target_action`은 baseline/virtual body target에 plug-tip local frame을 합성하고 socket-entry frame에서 translation과 shortest rotation vector 차이를 구한다. 각 3D 벡터는 component clip 뒤 norm 1로 투영해 1.5 mm·1.5° 한도로 바꾸고, 기존 absolute tip-target hook에 매 tick 적용한다. 다음 tick에 누적하지 않고 현재 GT와 같은 command depth에서 다시 계산한다. calibration bias는 최초 frozen object-to-EE 추정에 들어간 뒤 EE/FK로 전파되므로 단순히 매 tick `bias⁻¹`을 상대 pose에 적용하는 방식은 사용하지 않는다.

## classical force-search 수치 계약

79D의 0-based half-open slice와 역스케일은 다음과 같다. 상대 위치 `obs[0:3] × [0.001, 0.012, 0.001] m`, quaternion은 `w=obs[3]`, `xyz=obs[4:7] × (π/180)/2`, 관절 위치 `obs[7:21] × π rad`, 관절 속도 `obs[21:35] × π rad/s`, grip force `obs[35:37] × 8 N`, filtered entry net force `obs[37:40] × 8 N`, command depth `obs[62] × 0.012 m`, advance `obs[63]`, joint reference error `obs[64:78] × 0.03 rad`다.

자세 항은 quaternion을 rotation vector로 바꾼 뒤 `Δx=-0.8·p_x`, `Δz=-0.8·p_z`, `Δr_x=-0.8·r_x`, `Δr_z=-0.8·r_z`로 계산한다. Y 위치 잔차와 Y 회전 잔차는 0이다.

stall은 30 Hz에서 6 tick 연속 `advance=0`이면서 command depth 증가가 tick당 0.02 mm 미만이거나, entry force norm이 8 N 이상일 때 시작한다. 최초 탐색 방향은 lateral force `[F_x,F_z]`가 0.25 N 이상이면 가장 큰 절댓값 축의 반대 방향, 아니면 `+X`다. 방향은 `[d0, +90° 회전, -d0, -90° 회전]` 순서로 6 stalled tick마다 바꾼다. 네 방향을 마치면 반경을 0.15 mm씩 늘려 최대 0.60 mm까지 반복한다. advance가 3 tick 연속 회복되면 탐색 상태를 초기화한다.

자세 항과 탐색 항을 더한 XZ translation vector는 norm 1.5 mm, XZ rotation vector는 norm 1.5°로 투영하고 각각 한도로 나눠 normalized action을 만든 뒤 각 성분을 `[-1,1]`로 자른다. entry force norm이 20 N 이상이면 provider는 0을 출력하고 기존 supervisor의 후퇴·중단에 맡긴다. 25 N은 evaluator hard failure다. gain·순서·포화는 결과를 본 뒤 바꾸지 않는다.

여기서 actor의 entry force는 `obs[37:40]`의 **filtered net 3D force vector**다. evaluator의 25 N은 gripper 접촉 peak와 접촉쌍 absolute normal-load sum peak를 포함한 독립 판정이다. 두 수치를 같은 센서나 같은 norm으로 표현하지 않는다.

### classical 독립 tuning

위 수치는 아직 설계된 baseline 후보다. admission 72 episode 전에 train-03 한 grasp와 동일한 6개 signed bias를 tuning 전용으로 고정하고 세 후보만 각 6 episode, 합계 최대 18 episode 실행한다. 후보는 `(position gain, rotation gain, force trigger N, search step mm)` 기준으로 `A=(0.8,0.8,8,0.15)`, `B=(1.0,0.8,6,0.10)`, `C=(0.6,1.0,8,0.20)`이며 나머지 수치는 위 계약과 같다.

hard/forbidden contact 0회인 후보 중 성공 수가 큰 순서, evaluator force p95가 작은 순서, 완료시간 중앙값이 작은 순서, 후보 ID 사전순으로 하나를 고른다. 목표는 6개 중 4개 이상 성공이지만 미달해도 결과에 맞춰 후보를 추가하지 않는다. 실제 tuning에서는 A·B·C가 각각 3/6 성공했고 세 후보 모두 hard/forbidden contact가 0회였다. B는 사전등록 tie-break 규칙으로 선택돼 exact controller canonical config SHA와 tuning report SHA가 봉인됐지만 목표 4/6에는 미달해 `met_target=false`다. seal과 applied controller SHA 일치는 출처·선정 절차를 검증할 뿐 강한 성능을 입증하지 않는다. 따라서 B는 **사전등록 규칙으로 선택된 목표 미달 후보**로 표시하며 “강한 고전 baseline”으로 부르지 않는다. 4/6 목표, 후보 집합과 이후 G1 admission 판정 규칙은 이 결과에 맞춰 바꾸지 않는다.

constant/playback 봉인은 원본과 파생 아티팩트를 나눠 검증한다. 먼저 raw `source_trace_gzip_sha256`가 `3ef8a4456bdfedec12996d1f78a6dc2736347a771be53137cab4263353920f59`인지 확인한다. 그다음 `mean_constant_artifact_sha256`은 canonical JSON payload `{"source_trace_gzip_sha256": source SHA, "method": "mean_constant", "normalized_right6": [0.034892102401539844, -0.010848934130210961, -0.041849699608244736, 0.03485121803133042, -0.07816294091497804, -0.02233053562947764]}`의 SHA-256으로, `playback_artifact_sha256`은 `{"source_trace_gzip_sha256": source SHA, "method": "open_loop_playback", "selected_rows": 364, "tail": "hold_final_action"}`의 SHA-256으로 만든다. 각 episode의 `applied_artifact_sha256`은 해당 method의 파생 SHA와 일치해야 한다. raw gzip SHA와 파생 method artifact SHA가 같다고 가정하지 않는다. tuning을 하지 않거나 이 SHA 계약이 맞지 않으면 `designed_untuned_classical_candidate`로만 보고 `admit_rl_pilot` 판정을 내릴 수 없다. 18 episode는 기존 72 episode와 별도이며 총 개발 실행 상한은 90 episode다.

## 실행 수와 관측 가능성

2 grasp × 6 bias case = 12 case다. provider는 5개지만 `zero`만 2회이므로 case당 실행 slot은 6개, 최대 `12 × 6 = 72 episode`다. case 교체와 성공 tolerance 완화는 금지한다.

실제 G1은 frozen `104721a`의 training core를 사용한다. `rj45_training_env.py` SHA-256은 `ecdbfed020e0751721ad834b70e895a7f9ac195f4939e97ee8d82ddf7c0bd96c`, `rj45_training.py` SHA-256은 `cc59be01737c9d7db312c83c5592f93fdb10df74ab3f77ca34cada542954b180`이며 run identity manifest와 일치한다. 이 경로의 79D simulation proxy는 estimator·controller·joint·force state로 구성되고 base가 반환하는 noisy observation을 입력으로 소비하지 않는다. 별도의 센서 latency·dropout sampler도 없다. 따라서 아래 반복 차이 `N`은 일반적인 센서 노이즈 분산 추정이 아니라, 고정 시드 프로토콜에서 calibration bias·reset·접촉 동역학을 포함한 상태와 관측 궤적의 반복성을 보는 지표다.

`zero` slot 1의 12개만 성공률·headroom 분모에 넣는다. slot 2는 동일 시드 반복 관측 차이 `N` 측정 전용이며 성과 횟수에 합산하지 않는다. 각 run을 30 Hz의 첫 entry contact tick에 맞추고 그 tick을 포함한 13 sample, 즉 12 transition의 0.4초 window로 잡는다. contact는 SI로 복원한 `||obs[37:40]×8||₂ ≥ 0.25 N`이다. contact가 없거나 13 sample이 없으면 그 case/축은 관측 불가다.

feature는 GT·추정 상대 자세를 제외한 `concat(obs[7:40], obs[63:78])` 48D normalized vector다. 각 run의 13 sample별 feature 중앙값을 `m`이라 한다. 한 grasp·오차축의 부호 간 거리는 `D=||m₊¹-m₋¹||₂`, 동일 시드 반복 차이는 `N=max(||m₊¹-m₊²||₂, ||m₋¹-m₋²||₂, 1e-6)`다. `D≥0.05`이고 `D/N≥2.0`인 축만 observable이다. 각 grasp에서 3개 축 중 2개 이상이 통과해야 한다.

## 결과 상태와 학습 진입 gate

판정은 아래 한 상태로 끝난다.

- `invalid_contract`: archive/hash/ID-index/request-applied/pre-freeze reset 동등성/같은 bias의 INSERT 전환 동등성/최초 INSERT physical-zero/provider 권한/classical·constant·playback 적용 artifact SHA/유한값/72-slot 계약 중 하나라도 어긋남.
- `infeasible_or_unobservable`: oracle GT reference가 10/12 미만, grasp별 5/6 미만, hard/forbidden contact가 1회 이상이거나 위 observability 기준 미달.
- `valid_stop_no_headroom`: 계약과 가능성은 유효하지만 공정 비학습 방식 중 하나가 10/12 이상 성공하고, 그 방식의 12 episode 전체에서 hard/forbidden contact가 0회다. 이때만 전체 성공률 관점의 headroom 부재로 판정한다.
- `valid_hard_sparse`: reference와 observability는 통과했지만 zero slot 1이 0~1/12 성공. 현 조건에서 sparse task가 지나치게 어려우므로 본학습을 시작하지 않는다.
- `valid_stop_insufficient_structured_headroom`: 모든 공정 비학습 방식이 10/12 미만이고 zero slot 1이 2~9/12이지만, headroom case 2개·부호쌍 2개·위치와 각도·두 grasp 조건 중 하나 이상이 부족하다. 전체 성공률 headroom이 없다는 뜻은 아니며, 현재 개발 조건이 명시된 RL 비교 진입 구조를 충족하지 못했다는 판정이다. pilot을 시작하지 않고 결과를 보존하며 사후에 조건을 바꾸지 않는다.
- `admit_rl_pilot`: classical tuning과 위 계약·가능성·관측 gate를 통과하고 zero slot 1이 2~9/12, classical·constant·replay가 각각 10/12 미만이며, 모든 공정 비학습 방식은 실패하지만 reference는 성공하는 case가 2개 이상이다. 또한 zero가 양쪽 부호 모두 실패하고 reference는 양쪽 모두 성공하는 부호쌍이 최소 2개여야 하며 위치·각도 두 종류와 두 grasp에 걸쳐야 한다.

어떤 상태에서도 현재 결과를 보고 bias 크기나 tolerance를 조절해 같은 계약을 다시 실행하지 않는다. `valid_stop_no_headroom`과 `valid_stop_insufficient_structured_headroom`은 v1의 terminal 상태다. classical 승리를 피하거나 구조화 headroom 부족을 해소하려고 사후에 더 어려운 v2를 만들지 않는다. v2는 계측·계약 결함을 고치거나 별도의 과학 질문을 세울 때만 독립 development envelope로 만들고 v1 결과를 보존한다. G3의 여러 크기 범위 곡선과 final 5-seed/CI/안전 비열등 기준은 pilot 뒤 별도로 봉인한다.

## G1 Isaac smoke 결과와 실행 명령

2026-10-31 UTC의 attempt-D는 고정 환경 시드 `20261121`로 아래 3개 episode를 실행했다. classical A와 zero는 24.000 s horizon에서 실패했고 privileged GT reference는 12.133 s에 성공했다. 세 episode 모두 terminal contract를 완성했고 hard/forbidden contact는 없었다. 서로 다른 grasp가 섞인 단일 case smoke이므로 방법 간 성능 비교에는 사용하지 않는다.

실행된 smoke 명령의 재현 형태는 다음과 같다. 출력 디렉터리는 기존 결과를 덮어쓰지 않는 새 경로를 사용한다.

```bash
cd "$HOME/rby1-assembly-lab"
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src <Isaac-전용-python> -u scripts/run_rj45_task_admission.py \
  --runtime-root <동결-archive-root> \
  --current-root "$PWD" \
  --config configs/rj45-comparison-development.json \
  --output-dir artifacts/rj45-task-admission-smoke-<새-ID> \
  --smoke-only \
  --headless --device cuda:0
```

전체 development admission은 `--smoke-only`를 빼고 별도의 새 출력 경로로 실행한다. 이 명령은 classical tuning 18회와 admission 72회, 최대 90회를 수행한다. 현재 source `2e1f926` packet의 tuning 18회와 후보 B 봉인은 완료됐고 admission 72회가 실행 중이다. 전체 결과가 봉인되기 전에는 `task_admission_completed=false`, `rl_pilot_admitted=false`를 유지한다.

```bash
cd "$HOME/rby1-assembly-lab"
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src <Isaac-전용-python> -u scripts/run_rj45_task_admission.py \
  --runtime-root <동결-archive-root> \
  --current-root "$PWD" \
  --config configs/rj45-comparison-development.json \
  --output-dir artifacts/rj45-task-admission-full-<새-ID> \
  --headless --device cuda:0
```

attempt-C는 환경 생성 전에 `cfg.seed`가 없어서 세 번 모두 중단됐다. 이는 물리 또는 제어 결과가 아니라 구성 실패다. 실패 기록은 보존했고, attempt-D에서는 모든 case·method·slot에 같은 nonnegative integer seed `20261121`을 factory 호출 전에 전달하고 provenance에 기록했다. runner SHA-256은 `2f5483ee2c11cd8deb8b9bd87c2b1ddc1c10e912d1b21ece6eec1fe3af96a0f8`이다. 독립 검증은 3/3 terminal, 실행 계획, 144,686개 유한 수치, bias·held identity·최초 action 계약, 동결 runtime hash 35개를 확인했다. VM은 결과 회수 뒤 `STOPPED`로 확인됐다. 실행 창 비용 `0.7288 USD`는 시간과 단가를 곱한 추정치이며 청구액이 아니다. G0 완료 근거는 [`docs/evidence/rj45-comparison-g0-20261101/README.md`](evidence/rj45-comparison-g0-20261101/README.md)에 보존한다.

기존 `single64 2/8`, `multi64 3/8`, `zero 2/8`과 고정 PPO·상수·재생 대조는 모두 효과 미입증 개발 근거다. 단일 train 조건의 reference/PPO 성공도 물리 가능성과 한 조건 학습 가능성만 보여준다.
