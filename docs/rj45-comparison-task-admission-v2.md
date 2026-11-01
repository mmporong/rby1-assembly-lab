# RJ45 G1-v2: controlled observability와 비교 가능성 회복 계획

## 1. 목표와 완료 조건

G1-v1의 물리·안전 계약과 원래 연구 질문을 유지하면서, 수동 입력 관측성 실패를 이유로 RL 파일럿 전체를 닫았던 판정만 전향적으로 교정한다. G1-v2는 공유 probe를 비교 방법의 기반 동작으로 추가하지 않고, 허용된 bounded action 아래 부호 정보가 관측될 수 있는지를 별도 controlled-observability 실험으로 측정한다.

G1-v2의 완료 조건은 다음과 같다.

1. 기존 G1-v1 결과와 진단 e를 수정하거나 소급 재판정하지 않는다.
2. 새로 freeze한 position-family controlled-observability 실험이 양 부호와 두 train grasp에서 성공한다.
3. classical 후보·튜닝 예산·선정 순서를 결과 확인 전에 freeze하고, 목표 달성 여부와 무관하게 사전등록 순서로 선정한 최선의 bounded baseline을 신규 12회로 평가한다.
4. 기존 formal 자료 중 계약이 변하지 않은 행은 provenance와 해시가 일치할 때만 불변 자료로 재사용하고, 새 decision에 필요한 자료는 별도 prospective run으로 만든다.
5. classical 4/6 목표와 position-family headroom은 비교 실행의 gate가 아니라 claim qualifier로 판정한다. 미달이면 strong/range-expansion 주장을 금지하되 전체 비교는 계속한다.
6. `rx`는 평가 범위에 남기되 성공 허용각 0.5° 안의 ±0.2° ceiling/safety 음성 대조로 계층 보고한다.
7. 독립 검수가 치명적 gate인 계약·provenance·held-out seal, oracle feasibility, blind bounded controlled observability, framework/probe safety를 통과한 뒤에만 G2를 연다.

기존 사용자 요청 범위에서 구현과 검증을 진행한다. GPU 실행은 코드·입력·시간 상한을 고정하고 독립 검수를 마친 단위부터 수행한다.

## 2. 보존할 불변 계약

- 범위: G0~G5 전체 흐름, zero / 사전등록 classical / PPO / SAC의 공정 비교, 재현 가능한 실제 시뮬레이션 영상. classical이 4/6을 충족한 경우에만 strong qualifier를 붙인다.
- 물리: lateral 50 µm, angular 0.5°, insertion depth 12 mm, dwell 0.4 s, evaluator force 25 N.
- 입력 오차: `tx`, `tz` ±0.12 mm와 `rx` ±0.2°. 물리 기준 완화나 더 큰 bias는 허용하지 않는다.
- 인터페이스: observation 79, action 6, translation 1.5 mm 및 rotation 1.5° bounds.
- 공정성: 정책·classical provider는 허용된 obs/provider state만 사용하며 env state, extras, GT bias, case/sign/grasp 식별자를 받지 않는다.
- 데이터 경계: tuning/development와 held-out final을 분리하고, final은 G3 전까지 봉인한다.
- 안전: framework/oracle/probe의 hard-force·forbidden-state는 0건이어야 한다. bounds 안에서 개별 비교 방법이 낸 hard-force 실패는 해당 방법의 안전 결과로 보존한다. 실물 로봇 실행은 금지한다.
- 부수효과: 외부 게시, push, 배포, 알림 전송을 하지 않는다.
- 주장 경계: RL 우위, 모든 family 개선, 논문급 일반화를 G1/G2의 필수 성공 조건으로 두지 않는다.

## 3. 현재 근거와 해석 경계

### 3.1 불변 참고자료

- G1-v1 formal 90: zero 6/12, classical B 6/12, oracle 12/12, passive observability는 grasp별 1/3 axis 통과.
- classical B는 3/6 tuning target 4/6을 충족하지 못한 tie-break 후보다. 이후 문서와 표에서 `strong classical`로 부르지 않는다.
- 진단 e: `docs/evidence/rj45-active-z-diagnostic-20261101e`와 `.omx/evidence/rj45-active-z-diagnostic-20261101e/independent-completed-review.json`.
- 진단 e의 active `D`: train02 0.110278, train05 1.003105. 13-contact window 8/8, success 4/8, hard/forbidden 0건. sample 13과 zero-return 후 cue도 두 grasp에서 `D > 0.05`.

### 3.2 제한

- 진단 e는 설계 근거일 뿐 G1-v1 admission 근거가 아니다.
- e의 반복은 모두 같은 seed에서 byte-identical이므로 e 문서의 `N=1e-6`은 측정 floor다. 잡음 강건성이나 seed 일반화를 주장하지 않는다.
- zero-input의 정보 부족은 허용된 모든 action에서 정보가 없다는 뜻이 아니다.
- `rx` ±0.2°는 성공 허용각 0.5°보다 작으므로 이 설정에서 angular headroom 부재는 ceiling일 수 있다. `rx`를 빼거나 키우지 않고 음성 대조로 보존한다.
- position과 angle 양 family 모두에서 headroom을 요구한 v1 규칙은 원래 목표의 필수 조건이 아니라 내부 heuristic이었다. v2는 position-family의 전향적 계약으로 교체하되 최종 평가는 모든 method와 axis를 유지한다.

## 4. 실행 전 freeze 산출물

다음 파일은 GPU 실행 전에 작성·검수·freeze해야 한다. v1 파일과 판정 함수를 덮어쓰지 않는다.

| 파일/자원 | 최소 변경 | freeze 내용 |
|---|---|---|
| `docs/rj45-comparison-task-admission-v2.md` | 신규 | v2 질문, 판정 상태, position-family headroom, `rx` 계층 보고, 실패 분기 |
| `configs/rj45-controlled-observability-v2.json` | 신규 | axis/grasp/sign/seed 행렬, probe 파형, sample 13 primary, 안전 중단, 거리 판정 |
| `configs/rj45-classical-tuning-v2.json` | 신규 | 고정 후보 목록, 후보 수, 6-case tuning matrix, 최대 예산, 선정 순서 |
| `configs/rj45-comparison-pilot-v2.json` | 후속 작성 예정 | G1-v2 gate, PPO/SAC 3 seeds, raw transition budget 필드. G1 통과 전 `frozen=false` |
| `src/rby1_assembly/rj45_controlled_observability.py` | 신규 우선 | v2 거리·집계·판정. v1 `rj45_task_admission.py` 의미 보존 |
| `src/rby1_assembly/rj45_active_axis_probe.py` | 신규 또는 검증된 공통부의 최소 확장 | x/z axis-generic probe. e archive 재현 의미를 바꾸지 않음 |
| `scripts/run_rj45_controlled_observability_v2.py` | 신규 | freeze 검증, CPU preflight, 단일 GPU ordinal 실행. 묶음 감독·복구·archive는 별도 operator가 담당 |
| `scripts/run_rj45_task_admission_v2.py` | 후속 구현 예정 | 기존 불변 행 import, 새 관측성/classical 행 결합, v2 report 생성 |
| `scripts/run_rj45_comparison_long.py` | 후속 수정 예정 | v1 gate를 약화하지 않고 별도 v2 schema/status를 엄격 검증 |
| 관련 `tests/` | 신규·보강 | v2 판정, blind provider, sample alignment, freeze/hash, runner gate, 실패 분기 |

구현 단계에서는 위 파일명과 경계를 우선하되, 이미 존재하는 범용 함수로 중복 없이 표현할 수 있으면 함수 재사용은 허용한다. 기존 evidence archive, v1 config/report, v1 status의 byte 의미는 바꾸지 않는다.

## 5. 전향적 G1-v2 실험 설계

### 5.1 Controlled-observability addendum: 24회

새 decision의 독립 근거는 `2 grasp × 2 position axis(tx,tz) × 2 sign × 3 preregistered seed = 24 episode`로 만든다.

- grasp: train02, train05.
- axis: `tx`, `tz`. 진단 e가 `tz`만 다뤘으므로 `tx`를 새 자료 없이 통과시키지 않는다.
- sign: 각 axis의 ±0.12 mm.
- seed: `i=0,1,2`에 대해 `sha256('rj45-controlled-observability-v2/environment/{i}')`의 first 4 bytes를 big-endian integer로 읽고 `& 0x7fffffff`를 적용한 3개 값 `494541735`, `903551052`, `1615423706`을 config에 기록한다. 수동 선별이나 결과 후 교체를 금지한다.
- probe provider: axis별 `active_x_probe`, `active_z_probe`. 선택된 진단 axis는 config에서 주어질 수 있지만 provider는 sign/case/grasp/GT를 알 수 없다.
- 파형: 진단 e에서 검증한 범위 안의 고정 bounded sweep를 축만 바꿔 사용한다. 구체적으로 0.15/0.30/0.45/0.60 mm, 각 단계 6 tick, +axis→-axis, separator zero를 freeze한다. 첫 INSERT physical action은 zero로 고정한다.
- 안전: runner soft stop 20 N, evaluator hard criterion 25 N. action bounds와 forbidden-state 규칙을 그대로 적용한다.
- window: episode마다 first contact ≥0.25 N 기준 13개 연속 sample이 있어야 하며, 누락되면 해당 axis/grasp는 실패다.

Primary measurement는 policy-compatible한 contact sample 13의 단일 feature48(`obs[7:40] + obs[63:78]`)이다. 이는 기존 v1 관측성 특징 계약을 유지하면서 현재 feedforward policy가 받을 한 시점의 cue를 검사한다. 다음 항목은 구분해 기록한다.

- Primary: sample 13의 feature48 부호 거리.
- Secondary A: 같은 sample 13의 full obs79 부호 거리.
- Corroboration: 같은 13-sample window의 median feature48 부호 거리.
- Secondary B only: zero-return 후 첫 observation의 cue 보존. G1-v2 gate로 사용하지 않는다.

각 axis/grasp에 대해 seed별 양·음 primary 거리 `D_seed`를 계산하고, `D_min = min(D_seed)`로 보수 집계한다. 서로 다른 config seed에서 나온 동일 부호 primary vector의 최대 pairwise 거리는 `S_max`(same-sign dispersion)로 정의하고 수치 floor는 1e-6으로 둔다. primary gate는 모든 position axis/grasp 조합에서 다음을 만족해야 한다.

- `D_min >= 0.05`
- `D_min / max(S_max, 1e-6) >= 2`
- 3 seed 모두 valid 13-contact window
- hard/forbidden 0건

비율 2는 보수적인 separation/dispersion 판정일 뿐 noise 또는 robustness 지표가 아니다. config seed와 seed bank는 독립 확률 표본이나 stochastic 독립성을 증명하지 않는다. seed 변화에도 payload가 byte-identical이면 `S_max`가 floor임을 명시하고 stochastic robustness 주장을 하지 않는다. infra/provenance 오류는 원인을 고친 뒤 해당 freeze unit 전체를 한 번 재실행할 수 있지만, 성능이 낮다는 이유로 seed나 일부 episode만 골라 재시도하지 않는다.

### 5.2 Classical baseline 사전등록: 최대 24회 tuning + 12회 admission

기존 classical B는 목표 미달 후보로 보존한다. 새 classical v2는 기존 실패 행과 코드 동역학의 원인 분석으로 후보를 설계하되, 결과를 보기 전에 다음을 freeze한다.

현재 코드·실행 자료에서 우선 검토할 설계 가설은 다음과 같다. 이는 아직 검증된 원인이 아니며 후보 설계의 출발점이다.

- 기존 search plateau 6 tick은 약 0.2 s로, 성공 dwell 0.4 s보다 짧다. stalled end-depth에서 ±100 µm offset이 50 µm 성공 공차 안에 들어와도 plateau 종료 후 방향이 바뀌면 400 ms를 채우지 못할 수 있다.
- 따라서 plateau를 contact window보다 길게 고정하는 후보(예: 최소 13 tick, 검토안 18 tick), 50 µm step, `advance_recovery=3`에 따른 offset early-reset의 상호작용을 후보 축으로 검토한다.
- 진단 e의 force peak는 약 0.55 N인 반면 기존 A/B/C trigger는 6~8 N이지만 stalled branch도 존재한다. force threshold 하나만 실패 원인으로 단정하거나 threshold만 낮춘 후보군으로 제한하지 않는다.
- 위 수치는 코드 경로·tick 주기·상태 전이를 단위 시험으로 확인한 뒤 후보 config에 freeze한다. 확인 결과 가설이 틀리면 근거를 기록하고, 첫 GPU 결과 전에만 후보 정의를 수정할 수 있다.

- 후보 수: 최대 4개.
- 후보별 평가: train-03 signed-six 6회.
- 최대 tuning 예산: 24 episode.
- 후보 parameter, tie-break ID, timeout, state reset, action bounds를 config에 모두 기록.
- 후보 추가·수정: 첫 결과 생성 뒤 금지.
- 선정 순서: provider/action 계약 유효 및 tuning hard-force·forbidden 모두 0인 후보만 → success count 최대 → force p95 최소 → completion-time median 최소 → 고정 candidate ID.
- 목표: 선정 후보 4/6 이상. 미달이면 `target_missed_safe_baseline` qualifier를 붙이고 `strong classical` 명칭을 금지한다.

목표 달성 여부와 무관하게 사전등록 순서로 선정한 최선의 safe 후보를 train02/train05 signed-six 12회에서 새로 평가한다. 기존 classical B의 12회 결과를 새 후보 성능으로 대체하거나 섞지 않는다. 새 후보 역시 obs79/provider state만 사용하고 비교 방법에 active probe를 공통 base로 추가하지 않는다. 선정 후보가 hard-force failure를 내면 이를 method outcome으로 보존하며, task physics/provenance invalid와 혼동하지 않는다.

후보의 구체 수치와 후보 수가 4보다 적을 수 있는지는 구현 전 원인 분석과 단위 시험으로 결정하되, freeze 이후 늘리지 않는다. 이 계획은 수치를 임의로 발명하지 않는다.

### 5.3 기존 formal90 재사용 판정

동일 deterministic 90회를 전부 재실행하지 않는다. 재사용 대상은 방법 계약이 변하지 않은 zero, oracle, mean constant, playback 및 v1 참고 행이다. 다음 조건을 모두 통과한 행만 v2 report에 `historical immutable row`로 참조한다.

1. archive와 payload SHA, source commit, simulator/runtime fingerprint, scene/task config hash가 원본 manifest와 일치한다.
2. v2가 요구하는 물리·성공·안전 판정 필드를 원시 행에서 다시 계산할 수 있다.
3. 독립 검수가 row count, terminal count, force, success, forbidden state를 재계산한다.
4. 보고서에서 `fresh prospective`와 `historical immutable`을 별도 열로 표시한다.

호환되지 않는 방법/행만 영향 범위대로 재실행한다. classical 후보가 바뀌므로 새 classical 12회는 필수다. blanket 90회 재실행은 근거 독립성을 높이지 않고 동일 deterministic 결과와 GPU 비용만 반복하므로 금지한다.

따라서 새 G1 GPU 예산 상한은 관측성 24 + classical tuning 24 + 선정 classical admission 12 = 60 episode다. 후보가 적거나 early invalid stop이 발생하면 더 적다.

이 60회를 한 프로세스의 원자 단위로 실행하지 않는다. 기존 formal90은 9 chunk에서 wall time이 최소 약 160분이었고, 진단 e 8회는 약 25분이었다. v2 runner는 아래 bounded chunk를 freeze하고 chunk별 archive/검수를 완료한 뒤 다음 chunk로 넘어간다.

- observability: axis × grasp 단위 4 chunk, chunk당 6 episode(2 sign × 3 seed).
- classical tuning: candidate 단위 최대 4 chunk, chunk당 6 episode.
- selected classical admission: grasp 단위 2 chunk, chunk당 6 episode.
- chunk timeout/guard: CPU preflight 측정과 첫 GPU smoke로 상한을 계산해 config에 기록한다. 기존 worker 3000 s, end verify 120 s, coordinator 28500 s, outer 30000 s, formal guard 540 min을 복사해 쓰지 않고, episode 수와 관측 wall time에 근거한 여유를 둔다.
- observability의 초기 resource envelope는 e의 8회/25분과 guard 2 h, reserve 5가 정상 종료한 사실을 참고하되, 첫 valid chunk의 실제 elapsed를 기록해 남은 chunk가 freeze된 outer guard 안에 들어가는지만 확인한다. 결과가 나쁘다는 이유로 timeout을 늘리지 않는다.
- chunk마다 expected row/terminal count, archive SHA, 완료 marker를 확인한다. 중단 복구는 같은 freeze digest의 미완료 chunk 전체에만 허용하며 완료 chunk를 다시 실행하거나 서로 다른 시도의 성공 행을 합치지 않는다.

## 6. G1-v2 판정 계약

판정은 **fatal gate**와 **claim qualifier**를 분리한다. qualifier 미달을 이유로 전체 비교를 취소하지 않는다.

### 6.1 Fatal gate

1. **계약·provenance·seal 유효성**: freeze/hash, schema, row cardinality, source/runtime binding, held-out 미접근을 검증한다.
2. **물리 가능성**: 호환성이 확인된 oracle 자료로 oracle ≥10/12, grasp별 ≥5/6을 확인한다. oracle 실행 또는 공통 task에서 hard/forbidden이 발생하면 physics/safety invalid다.
3. **blind bounded controlled observability**: 24회 prospective 자료에서 provider가 sign/case/grasp/GT를 받지 않고 bounds를 지키며, `tx`,`tz`의 양 부호·두 grasp primary feature48 gate를 통과해야 한다.
4. **framework/probe safety**: probe/action bounds, forbidden state, runner guard를 통과해야 한다. 이 공통 안전 실패는 method outcome과 구분한다.

이 네 조건 중 하나가 실패하면 G2/G3를 열지 않는다. 반대로 모두 통과하면 아래 qualifier와 무관하게 G2를 연다.

### 6.2 Claim qualifier

1. **classical target**: 사전등록 후보 중 선정한 baseline이 tuning 4/6 이상이면 `classical_target_met`; 미달이면서 provider/action 계약과 safety 우선순위를 지킨 후보면 `target_missed_safe_baseline`이다. 후자의 경우 strong 명칭만 금지하며 신규 12회 평가와 이후 비교는 계속한다.
2. **position-family structured headroom**: 각 grasp에 최소 한 개씩, 총 최소 두 개의 사전등록 position signed pair에서 zero 양 부호 실패, oracle 양 부호 성공, 공정한 비학습 방법이 양 부호를 모두 닫지 못하면 `position_range_expansion_claim_eligible`이다. 미달이면 `position_range_expansion_claim_false`로 기록하지만 full comparison은 계속한다.
3. **zero/nonlearning 난이도 맥락**: zero 2~9/12와 각 비학습 방법 <10/12는 headroom 해석 qualifier다. 범위를 벗어나도 fatal invalid로 바꾸지 않고 ceiling/sparsity 맥락을 보고한다.
4. **method hard-force**: bounds 안에서 실행된 특정 classical/PPO/SAC/zero 방법의 25 N 초과는 그 방법의 실패·안전 결과다. 공통 physics/provenance invalid로 승격하지 않는다.
5. **`rx` 음성 대조**: 모든 방법의 `rx` 결과를 최종 표·안전 집계에 유지하되 position-family claim의 필수축으로 사용하지 않는다. angular-range 개선 주장은 하지 않는다.

버전형 fatal 상태와 qualifier를 별도 필드로 저장하며 v1 status를 재사용하지 않는다.

- Fatal: `invalid_contract_or_seal_v2`, `infeasible_oracle_v2`, `controlled_feature48_not_demonstrated_v2`, `invalid_framework_or_probe_safety_v2`, `admit_training_and_final_comparison_v2`.
- Qualifier: `classical_target_met` 또는 `target_missed_safe_baseline`; `position_range_expansion_claim_eligible` 또는 `position_range_expansion_claim_false`; zero/nonlearning ceiling·sparsity 맥락.

음성 qualifier도 유효한 연구 결과다. bias 확대, tolerance 완화, 후보 사후 추가, 관측 기준 교체로 주장을 통과시키지 않는다.

## 7. G2 learnability pilot와 G3 final claim 분리

### 7.1 G2: training validity와 learnability 결과 기록

G1-v2가 fatal gate를 통과해 `admit_training_and_final_comparison_v2`가 된 뒤 `configs/rj45-comparison-pilot-v2.json`의 빈 필드를 채워 freeze한다. classical/headroom qualifier는 진입 gate가 아니다.

- PPO/SAC 각각 3개 고유 training seed.
- 같은 raw environment-transition budget과 paired evaluation schedule.
- PPO rollout/update 조건과 SAC replay warmup/UTD/minimum transitions를 수치로 고정.
- development grasp/case만 사용하고 held-out final 접근 금지.
- zero와 선정 classical은 평가 기준선이며 표본 수를 부풀리려고 반복 학습하지 않는다.
- checkpoint 선정 기준과 G3 진입 기준을 G2 결과를 보기 전에 고정한다.

G2의 질문은 “이 bounded observation/action 계약에서 PPO/SAC 학습 실행이 유효했으며 어떤 결과가 나왔는가”다. RL이 classical을 이기거나 PPO/SAC가 개선해야만 G3로 갈 수 있는 구조가 아니다. 사전등록한 sufficient transitions/updates를 소화하고, loss/action/value가 finite이며, data leakage가 없고, frozen checkpoint rule로 checkpoint가 선택되면 `training_valid`다. 개선이 0이거나 악화돼도 음성 결과를 보존하고 G3 all-method sealed final로 진행한다. 성능이 낮다는 이유로 새 학습 budget, reward, bias, tolerance를 추가하지 않는다. non-finite, transition/update 부족, leakage, checkpoint-rule 위반만 training invalid로 처리한다.

### 7.2 G3: held-out final claim

G2 전에 고정한 checkpoint rule로 선택된 checkpoint hash를 잠근 뒤, final metric, uncertainty, 최소 5 training seeds와 sealed held-out grasp/case manifest를 별도 final config에 freeze한다. G2 개선 여부로 checkpoint나 final 조건을 고르지 않는다.

- 모든 method: zero, 선정 사전등록 classical baseline, PPO, SAC. classical 4/6 미달 시 strong 명칭을 쓰지 않는다.
- 모든 axis: `tx`, `tz`, `rx`.
- success, force, time, compute cost와 uncertainty를 seed/grasp/axis/family별로 보고.
- position-family 주장은 `tx`,`tz` 계층 결과로 제한하고 pooled 평균으로 실패 subgroup을 숨기지 않는다.
- `rx`는 ceiling/safety 음성 대조로 명시한다.
- final 결과를 보고 reward, 후보, seed, checkpoint를 바꾸지 않는다.

RL 우위나 모든 family 개선이 나오지 않아도 비교는 그대로 보고한다. publication claim의 강도는 G3 evidence가 허용하는 범위로 낮춘다.

## 8. 영상과 재현성

- 비교 영상은 G3의 실제 held-out episode/run ID에서 생성한다.
- zero/classical/PPO/SAC pane은 서로 다른 실제 episode를 사용하고 복제 화면을 만들지 않는다.
- 성공 장면만 고르지 않고 각 방법의 실제 실패·hard-force 결과도 사전등록된 영상 선정 규칙에 따라 포함한다.
- run ID, seed, grasp, axis/sign, config/source/archive hash, success/force/time을 영상 manifest와 연결한다.
- active controlled-observability 장면은 보조 설명 영상으로만 사용할 수 있고 성능 비교 영상으로 섞지 않는다.
- 외부 게시와 업로드는 별도 요청 전까지 하지 않는다.

## 9. 구현·검증 순서

1. **계약 작성**: v2 문서와 두 freeze config schema를 작성하고 v1 파일 불변성을 diff로 확인한다.
2. **순수 판정 코드**: sample 13 feature48 alignment, seed별 `D`, cross-seed same-sign dispersion `S_max`, conservative aggregate, fatal gate와 qualifier를 구현한다.
3. **단위·부정 시험**: sign/case/grasp leakage, missing sample, seed 중복, floor 오해, candidate 사후 추가, final 접근, v1 status 오수용을 실패시킨다. classical/headroom qualifier 미달은 fatal이 아니며, training-valid no-improvement가 G3로 이어지는지도 회귀 시험한다.
4. **CPU preflight**: native simulator import 없이 config/schema/cardinality/hash/seed namespace를 검증한다.
5. **freeze review**: reviewer가 후보·예산·seed·probe·판정 문구가 결과 전에 고정됐는지 확인하고 freeze digest를 남긴다.
6. **관측성 GPU campaign**: 동일 freeze digest 아래 24회를 4개 bounded chunk로 실행하고, 각 archive를 검수한 뒤 campaign 전체를 독립 재계산한다.
7. **classical GPU campaign**: fatal gate가 유효한 상태에서 후보별 bounded chunk로 최대 24회 tuning을 실행하고, 목표 달성 여부와 무관하게 선정 후보 12회 admission을 grasp별 chunk로 실행한다.
8. **G1-v2 합성**: 호환된 historical row와 prospective row를 분리 결합하고 판정을 재계산한다.
9. **독립 승인**: code-reviewer/verifier가 원시 archive에서 같은 status를 재생성한 경우에만 G2 gate를 연다.

각 GPU unit은 ordinal, source commit, dirty-tree digest 또는 clean-tree proof, simulator/runtime fingerprint, config SHA, seed 목록, expected row count를 시작 전에 기록한다. 중단 시 stop/recover guard가 같은 freeze identity만 복구하도록 한다.

## 10. 검증 명령과 증거

현재 구현된 관측성·탐색기·실행기 검증 세트는 다음과 같다. G1 종합 판정과 G2 연결은 후속 구현이며, 해당 파일이 생긴 뒤 관련 검사를 추가한다.

```bash
cd $HOME/rby1-assembly-lab
.venv/bin/python -m pytest -q \
  tests/test_rj45_controlled_observability.py \
  tests/test_run_rj45_controlled_observability_v2.py \
  tests/test_rj45_controlled_observability_analysis.py \
  tests/test_rj45_active_axis_probe.py \
  tests/test_rj45_classical_v2.py \
  tests/test_run_rj45_task_admission.py
.venv/bin/python -m py_compile \
  src/rby1_assembly/rj45_controlled_observability.py \
  scripts/run_rj45_controlled_observability_v2.py \
  src/rby1_assembly/rj45_controlled_observability_analysis.py
git diff --check
```

GPU 실행 후에는 독립 검수 JSON이 최소한 다음을 포함해야 한다.

- archive/payload/source/config SHA와 runtime fingerprint.
- expected/actual episode 및 row 수, terminal 수.
- grasp/axis/sign/seed cardinality와 중복·누락 0건.
- sample 13 primary feature48 raw vector binding, sample 13 full79 secondary, median48 corroboration, zero-return secondary 분리.
- `D_seed`, `D_min`, cross-seed same-sign dispersion `S_max`, floor 사용 여부와 separation/dispersion criterion.
- candidate freeze digest와 tuning/admission 분리.
- success/force/time/hard/forbidden 재계산.
- historical/prospective 행 provenance 분리와 최종 v2 status 독립 재계산.

## 11. 실패·중단 규칙

- 계약/hash/provider leakage/row cardinality가 틀리면 결과를 판정에 쓰지 않는다.
- controlled observability가 position axis 하나 또는 grasp 하나에서 실패하면 G2를 열지 않는다.
- classical 후보가 4/6에 못 미치면 `target_missed_safe_baseline`로 표시하고 strong 명칭을 금지하지만 선정 후보 12회와 G2/G3 비교는 계속한다.
- position-family structured headroom이 두 grasp를 덮지 못하면 range-expansion claim을 false로 두되 full comparison은 계속한다.
- G2에서 개선이 없어도 sufficient transitions/updates, finite 수치, no leakage, frozen checkpoint 조건을 충족하면 training-valid 음성 결과로 G3를 계속한다.
- 공통 task/oracle/probe/framework의 hard/forbidden 위반은 fatal 안전 실패로 중단한다. bounds 안의 개별 비교 방법 hard-force는 그 방법의 결과로 기록하고 나머지 비교를 invalid 처리하지 않는다.
- infra/provenance 문제만 수정 후 같은 frozen unit의 전체 재실행을 허용한다. 결과 선택형 재시도는 금지한다.
- 어떤 실패도 bias 확대, tolerance 완화, `rx` 삭제, held-out tuning, 사후 후보/seed 추가로 회피하지 않는다.

## 12. 단계별 인계 기준

| 단계 | 필수 산출물 | 다음 단계 진입 조건 |
|---|---|---|
| G1-v2 freeze | v2 계약, observability/classical config, tests, freeze digest | 결과 생성 전 reviewer 승인 |
| G1-v2 evidence | 24 관측성 + 최대 24 tuning + 선정 classical 12 admission, 독립 review JSON | fatal gate 통과와 `admit_training_and_final_comparison_v2`; qualifier 별도 기록 |
| G2 training validity | PPO/SAC 3-seed dev 결과, transitions/updates, finite/no-leak 검증, frozen checkpoint rule | `training_valid`; 개선 여부는 claim 결과이지 진입 gate가 아님 |
| G3 final | 최소 5 training seeds, sealed held-out all-method/all-axis 결과 | 독립 통계·원시자료 검수 |
| G4 video | 실제 run manifest와 재현 영상 | G3 run binding 일치 |
| G5 closeout | 독립 최종 검수, 로컬 커밋, 잔여 위험 기록 | 요청된 저장소 범위의 완료 조건 충족 |

이 계획의 핵심 stop condition은 “G1-v2를 통과시키는 것”이 아니라, 사전등록한 최소개입 계약 아래 치명적 유효성은 엄격히 지키면서 classical 목표 미달, headroom 부재, 학습 개선 부재까지 실제 final comparison과 영상으로 정직하게 남기는 것이다.
