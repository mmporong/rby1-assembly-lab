# RJ45 전향적 관측성 — 첫 묶음 조기 종료

고정 소스 `77d64e04998c1decc7fe79f3f5a74cc4f45b9e29`로 train-02의 tx 양·음 오차를 각각 환경 시드 3개에서 실행했다. 계획한 24회 중 ordinal 1~6만 완료했다. 원래 물리·성공 기준, observation79, right-arm action6, case-blind 탐색 조건을 유지했다.

## 결과와 판정

| 항목 | 실측 | 등록 기준 | 판정 |
| --- | ---: | ---: | --- |
| primary sample13 feature48의 최소 부호 간 거리 | 0.04684900585965917 | ≥0.05 | 실패 |
| 같은 부호의 시드 간 최대 거리 | 0 | 계산 하한 1e-6 | 하한 적용 |
| 신호/시드 간 분산 비율 | 46849.00585965917 | ≥2 | 통과 |
| 유효 접촉 관측 | 6회 모두 13개 | 각각 13개 | 통과 |
| 조립 성공 | 0/6 | 진단 결과 | 성능 비교에 합산하지 않음 |
| hard / forbidden contact | 0 / 0 | 안전 기록 | 위반 없음 |

두 분리 조건은 AND다. 비율이 커도 거리 기준 미달을 보상하지 않는다. 같은 부호의 세 시드 trace가 바이트 단위로 동일하므로 시드 변경이 독립적인 확률 변동을 만들었다고 주장하지 않는다. 1e-6은 계산 하한이며 측정 잡음이 아니다.

등록된 모든 축·파지에서 통과해야 한다는 조건에 따라 `controlled_feature48_not_demonstrated_v2` 조기 실패를 기록했다. 다른 묶음은 완료된 train-02/tx의 실패를 바꿀 수 없다. [계약](../../rj45-comparison-task-admission-v2.md)의 early invalid stop 규칙에 따라 ordinal 7~24는 미실행으로 남겼다. g 패킷은 로컬 준비만 했으며 native 실행하지 않았다. classical tuning 및 신규 baseline, PPO/SAC 학습과 held-out final도 시작하지 않았다.

exact24 종합 판정기는 호출하지 않았다. `full_24_adjudication_performed=false`, `task_admission_completed=false`, `rl_pilot_admitted=false`다. 이는 현재 고정 계약의 실패이며 모든 허용 제어에서 전역적으로 관측 불가능하다는 증명은 아니다. 기준·시드·probe를 결과에 맞춰 바꾸지 않았다. 전체 방법 비교와 최종 영상 목표는 미완료다.

## 검증 근거

- 회수 아카이브: 정규 멤버 92개 / payload 91개, 전수 크기·SHA 및 추출 후 재검증 PASS.
- 원시 trace 4,362행, case 6개: 고정된 소스의 provider 상태·action 전 행 재생 PASS.
- native admission/claim, source packet 22개, worker/verifier 종료 증거 검증 PASS.
- 별도 verifier가 `math.dist`로 재계산한 D_min은 0.046849005859659176으로 동일 판정이다.
- 소스·기준 불변, held-out 미접근, 외부 게시 없음.

아카이브 SHA-256: `2f9218e8fecb88e476690c3f39583e82663445fb2bb258656dbfbd76b0a02c1a`.

회수 원본: `/opt/homedata/rby1-storage/recovered/rj45-controlled-observability-20261101f/job`.

[회수 검증](local-recovery-verification.json), [원시 재계산](chunk1-analysis.json), [독립 검수](independent-completed-review.json), [조기 종료 범위](campaign-early-stop.json)를 함께 보존한다. GPU와 자동 종료 guard의 최종 상태는 `recovery-closure.json`에 기록한다.
