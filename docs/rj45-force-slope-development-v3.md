# RJ45 G1-v3 개발 가설: 힘 응답으로 X 탐색 방향 선택

## 상태와 목적

v2는 train-02/tx에서 sample13 feature48 거리 0.046849가 등록 기준 0.05에 못 미쳐 조기 종료했다. [v2 결과](evidence/rj45-controlled-observability-20261101f/README.md)와 고정 소스·설정·원시 자료는 실패 상태로 보존한다.

이 문서는 다른 제어법 하나를 개발 자료로 검증하는 별도 계약이다. 후보와 개발·확인 행렬을 CPU 독립 검수 후 `frozen=true`로 고정했다. GPU 실행은 별도 패킷과 native gate 검수를 거친다. 물리·성공 기준과 관측 분리 기준은 유지한다. 목표는 기존의 zero/classical/PPO/SAC 공정 비교와 실제 영상이다.

## 기존 자료에서 확인한 기전과 남은 가설

f의 첫 양·음 실행에서 +0.15 mm command 뒤의 네 관측을 이전 action과 맞춰 추출했다. policy observation37:40은 socket-entry frame의 Plug–Socket net contact force를 8 N으로 나눈 값이다.

| train-02 조건 | 네 응답의 마지막−첫 번째 \|Fx\| 차이 | v2 첫 접촉 offset | waveform index |
| --- | ---: | ---: | ---: |
| tx+ | −0.15931792184710503 N | +0.30 mm | 12 |
| tx− | +0.17837853729724884 N | +0.15 mm | 3 |

v2의 음의 파형은 index29에서 시작한다. 두 조건 모두 그 전에 첫 접촉에서 멈춰 13개 관측을 유지한 뒤 zero로 복귀했다. v2의 축 wrapper는 Z 제어기의 action index만 X로 바꾸며 이 첫 접촉 정지 동작을 재사용한다.

이 사실은 힘 응답의 변화 방향으로 서로 다른 탐색 방향을 선택할 근거다. 같은 방향에서 정지한 것이 거리 미달의 유일한 원인인지는 확인하지 못했다. 기존 자료에는 새로운 −X 동작의 결과가 없으며, 새 제어가 다른 grasp에서 관측 분리도를 높인다는 주장도 아직 없다.

설계 근거의 원시 archive SHA는 `2f9218e8fecb88e476690c3f39583e82663445fb2bb258656dbfbd76b0a02c1a`다. 계산 입력 행·force vector·trace/report SHA는 `evidence/rj45-force-slope-design-20261101h/source-response-analysis.json`에 보존한다. 이 자료는 개발 근거이며 새 검증 결과로 합산하지 않는다.

## 단일 후보와 시간 정렬

1. provider는 observation79와 소유 state만 받는다. axis는 X로 고정하며 case/sign/grasp/GT 입력은 없다.
2. 첫 INSERT physical action은 zero다. provider는 INSERT/active 상태이며 commanded depth가 1.3 mm에 도달할 때 zero를 한 번 출력한다.
3. 다음 호출에서 +0.15 mm impulse command를 출력한다. 이 호출에서 받은 관측은 impulse 응답에 세지 않는다.
4. 이전 +0.15 mm command에 대응하는 다음 네 유효 관측만 r1~r4로 센다. `Δ = abs(Fx(r4)) - abs(Fx(r1))`다.
5. Δ>0이면 +X, Δ<0이면 −X를 선택한다. Δ=0은 영구 invalid이며 다른 후보나 임계값을 추가하지 않는다.
6. 모든 방향에서 결정 호출은 zero를 출력한다. 다음 호출부터 선택 방향의 0.15/0.30/0.45/0.60 mm를 각 6 control tick 유지한다.
7. 접촉 검사는 selected nonzero command가 만든 다음 관측에서만 시작한다. impulse·separator의 관측으로 contact window를 시작하지 않는다. 선택 방향 d에 대해 `−d × Fx >= 0.25 N`일 때 contact sample1로 센다. Y/Z 힘이나 반대 부호 X 힘은 이 조건을 통과하지 못한다.
8. 첫 접촉 offset을 총 13개 관측 동안 유지한다. sample13이 primary feature48이다. sample13 호출부터 0.15 mm/tick으로 zero에 복귀하고 종료한다. 두 번째 접촉 window는 만들지 않는다.
9. 최대 범위까지 접촉이 없으면 zero로 복귀하되 contact window 0으로 실패를 기록한다. 새로운 방향·크기·대기 시간을 추가하지 않는다.

이 제어는 접촉 반응을 측정하기 위한 force-ascent 진단이다. 조립 보정의 성능 결과로 합산하지 않는다. 이전 관측을 state에 저장하므로 feedforward PPO/SAC가 같은 탐색을 구현한다는 근거로 쓰지 않는다. G2 학습의 유효성과 효과는 이후 별도 실험에서 판정한다.

## 고정 기준과 안전

- 물리: lateral 50 µm, angular 0.5°, depth 12 mm, dwell 0.4 s, evaluator 25 N.
- 오차: tx/tz ±0.12 mm, rx ±0.2°. obs79/action6, translation bound 1.5 mm, rotation bound 1.5°.
- probe peak 0.60 mm, 일반 step/return 0.15 mm. 안전·invalid 전환은 즉시 zero다.
- net force norm ≥20 N이면 영구 zero latch한다. phase/active 상태를 잃어도 영구 invalid zero다. total normal load·hard/forbidden 판정과 입력·소스 closure는 원래 런타임 검증을 유지한다.
- projected trigger는 `|Fx| >=0.25 N`과 norm ≥0.25 N을 모두 함의한다. 기존 접촉 문턱을 낮추지 않는다. net force의 다중 접촉 상쇄 가능성은 남으므로 contact 위치나 전체 압력을 입증한다고 주장하지 않는다.
- primary는 같은 contact sample13 feature48(`obs[7:40]+obs[63:78]`)이다. `D_min >=0.05`, `D_min/max(S_max,1e-6) >=2`, 모든 window13, hard/forbidden 0을 모두 요구한다. 가장 유리한 시점·feature·시드로 교체하지 않는다.
- 새 provider를 비교 방법들의 공통 전처리로 넣지 않는다. held-out과 실물 로봇은 사용하지 않는다.

## 실행 전 확정할 개발·확인 행렬

설정 파일은 `configs/rj45-force-slope-observability-v3.json`이다. 개발과 이후 확인 행렬을 함께 고정하며 단일 후보만 허용한다.

| 단계 | grasp × axis × sign × seed | episode | 효력 |
| --- | --- | ---: | --- |
| development | train-03 × tx × ± × 기존 3 seed | 6 | 후보의 개발 검증 |
| confirmation | train-02/train-05 × tx/tz × ± × 같은 3 seed | 24 | 새 제어의 전향 관측 검증 |

시드는 v2의 `494541735, 903551052, 1615423706`을 그대로 쓴다. 기존 deterministic byte-identical 현상을 고려해 확률적 독립성·강건성을 주장하지 않는다. train-03은 f를 설계 근거로 삼은 두 grasp 밖의 개발 조건이며 최종 held-out이 아니다.

개발은 exact6, 최대 한 번이다. schema/provenance/수집 오류도 결과가 없는 infrastructure 실패로 명확히 검증되기 전에는 재실행하지 않는다. 유효한 개발 결과가 기준 미달이면 후보 실패로 보존하고 confirmation을 열지 않는다. 개발 자료만으로 task/RL admission을 만들지 않는다.

개발 통과 후에만 같은 controller/config/source hash로 새 confirmation24를 실행한다. X는 새 provider, Z는 고정된 기존 active-axis Z provider를 사용한다. train-02/05는 이미 개발 근거가 있는 파지이므로 이 확인을 grasp-independent 일반화라고 부르지 않는다. v2의 6회와 새 24회를 섞지 않는다. 확인 통과 뒤에도 oracle·계약·seal·공통 안전과 classical 신규 평가가 포함된 G1 종합 검수가 필요하다.

## 검증과 실행 경계

CPU 검증은 네 impulse 응답의 이전 action 정렬, Δ 양/음/0, separator와 selected-action causality, projected trigger, window13, bounded return, phase/20 N 영구 latch, 입력 검증과 원본 불변성을 포함한다. CPU 테스트는 실제 접촉 force overshoot를 증명하지 못한다.

GPU 실행 전에 provider/계약/runner/raw analyzer와 회수 도구, source/config hash, exact instance identity, 단계별 ordinal, timeout·자동 종료 guard, 독립 검수와 native 실행 gate를 모두 확정한다. 첫 native 실행은 이 새 제어의 safety smoke를 포함해야 한다. 현재 문서·CPU 구현만으로 native 실행을 허용하지 않는다. 잔액 조회·외부 게시·push·알림은 수행하지 않는다.
