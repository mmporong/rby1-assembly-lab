# RJ45 높이 탐색 관측 진단 — 2026-11-01e

소스 `906fed3873a16c84f6202115e8a03fb718830bb7`의 case-blind active-z 진단을 8회 실행했다. 기존 물리 런타임, ±0.12mm 높이 오차, 성공 판정, observation79와 action bounds는 유지했다. 탐색 provider는 case·sign·grasp·GT를 받지 않는다.

## 결과

| 파지 | 높이 오차 | 반복별 조립 성공 | 접촉 관측 창 |
| --- | --- | --- | --- |
| train-02 | + | 성공 / 성공 | 각각 13개 |
| train-02 | − | 실패 / 실패 | 각각 13개 |
| train-05 | + | 성공 / 성공 | 각각 13개 |
| train-05 | − | 실패 / 실패 | 각각 13개 |

조립은 4/8회 성공했고 hard/forbidden contact는 0회였다. 이는 탐색 진단의 결과이며 zero·classical·PPO·SAC 성능표에 합산하지 않는다.

등록한 48차원 관측 요약의 부호 간 거리 D는 train-02에서 0.110278, train-05에서 1.003105였다. 두 파지 모두 D≥0.05와 D/N≥2를 통과했다. 같은 grasp/sign의 반복 trace는 바이트 단위로 동일하고 관측 요약의 반복 거리는 0이었다. N=1e-6은 등록된 계산 하한으로, 측정된 환경 잡음이 아니다. 큰 D/N을 잡음 내성·일반화·학습 가능성의 증거로 해석하지 않는다. 모든 실행의 환경 시드는 20261121이다.

train-02의 주요 차이는 z축 접촉력 관측이었다. train-05에는 삽입 진행 여부 신호도 크게 기여했다. 이 신호들은 원래 actor observation79에 포함된 정보다. evaluator GT나 latent bias를 actor에게 제공하지 않았다.

## 원시 검증

- 아카이브 104 regular members / 103 payloads: 크기·SHA 검증 PASS.
- 4,626행: hash chain, terminal 8개, source closure, provider 상태·action 재생 검증 PASS.
- commanded/requested/bounded action과 실제 tip target offset의 같은 틱 연결 검증 PASS.
- 전체 좌표계로 계산한 기하와 reported 기하 불일치 0행. legacy y-only 계산은 4,614행에서 불일치했다.
- 별도 verifier가 원시에서 재계산한 aggregate는 실행 중 생성된 판정과 일치했다.

아카이브 SHA-256: `5696a2fa44b62e9d8fa1b8c12d0a0de7079e494ab35f75491708fac4159b6999`.

로컬 원본: `/opt/homedata/rby1-storage/recovered/rj45-active-z-diagnostic-20261101e/job`.

## 한 시점 관측의 탐색 분석

13번째 접촉 관측과 zero 복귀 후 첫 관측을 고정해 별도로 추출했다. 48차원 부호 간 거리는 각각 train-02에서 0.114436 / 0.060886, train-05에서 1.003355 / 1.001295였다. 같은 시점의 반복 거리는 0이었다. 이는 개발 데이터의 추가 분석이며 사전등록한 admission이나 PPO/SAC 학습 성공 판정이 아니다. 수치는 `handoff-cue-metrics.json`에 있다.

## 해석 범위

이 진단은 허용된 case-blind 동작으로 높이 오차에 따른 관측 차이를 만들 수 있음을 보여준다. 수동 zero 궤적의 무접촉 결과만으로 모든 허용 제어에서 정보가 없다고 결론내릴 수는 없다.

기존 formal90 v1 실패와 모든 원본을 보존한다. `task_admission_completed=false`, `rl_pilot_admitted=false`를 유지한다. 새 controlled-observability 계약은 별도 사전등록·검증 대상이며, 이 결과를 기존 v1의 소급 통과로 사용하지 않는다. 모든 비교 방법과 축, 물리 성공 기준, held-out 분리, 실제 영상 범위를 유지한다.
