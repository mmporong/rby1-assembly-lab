# RJ45 task-admission G1 smoke 근거

상태: **3-episode smoke PASS, 전체 G1 미완료**. Isaac Sim 5.1 / Isaac Lab release 2.3 환경에서 고정 시드 `20261121`로 실행 경로와 terminal contract를 확인했다. 전체 classical tuning 18회와 admission 72회는 실행하지 않았고, `task_admission_completed=false`, `rl_pilot_admitted=false`다.

## 실행 결과

| 순서 | 단계 | grasp | bias | 방식 | 결과 | 완료 시각 | hard / forbidden |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| 1 | tuning | `rj45-train-03` | `tx-positive` | `classical_force_search A` | 실패 | 24.000 s | 0 / 0 |
| 2 | admission | `rj45-train-02` | `tx-positive` | `zero` | 실패 | 24.000 s | 0 / 0 |
| 3 | admission | `rj45-train-02` | `tx-positive` | `oracle_gt_reference` | 성공 | 12.133 s | 0 / 0 |

세 실행은 terminal contract를 모두 완성했고 실행 오류가 없었다. 다만 bias는 `tx-positive` 한 종류뿐이고 classical과 나머지 방식의 grasp도 다르다. 따라서 oracle 성공과 두 방식의 실패를 방법 간 성능 차이, 강화학습 필요성, 일반화 또는 효과로 해석하지 않는다. 이 smoke가 확인한 범위는 환경 생성, 고정 bias와 held-grasp 적용, 최초 INSERT action gate, 접촉 판정, 종료 snapshot, 결과 저장 경로다.

## 구성 실패와 수정 범위

앞선 attempt-C는 `cfg.seed`가 설정되지 않아 환경 constructor가 `RJ45 residual training requires a non-negative integer cfg.seed`로 세 번 모두 중단됐다. 물리 episode는 시작되지 않았다. 실패 아티팩트는 원인 추적을 위해 보존했다.

attempt-D는 모든 case·method·slot에 같은 환경 시드 `20261121`을 task configuration과 constructor 전에 전달했다. runner SHA-256은 `2f5483ee2c11cd8deb8b9bd87c2b1ddc1c10e912d1b21ece6eec1fe3af96a0f8`, development config SHA-256은 `ce917d703735e311c44af344486eb92d3049abfbc61ce2836c42f4e6b7719386`이다. 공정 조건의 RNG를 동일하게 고정하는 구성 수정이며 threshold, physics, bias 크기, 성공 기준은 바꾸지 않았다.

## 검증과 남은 단계

독립 검증은 계획과 실행 3회 일치, terminal 3/3, 유한 수치 144,686개, bias·held identity·action 계약, 동결 runtime hash 35개, final split 미개봉을 확인했다. 원본 report SHA-256은 `33b5ab8861ba4afe635967881d00dad91092aee15cbcad8eb0dd55876fa12753`이다. 3.9 MB 원본 report는 `.omx/evidence/rj45-task-admission-20261101/attempt-d/output/report.json`에 보존하고 Git 근거에는 복사하지 않았다. 구조화된 수치와 SHA는 [`summary.json`](summary.json)에 정리했다.

다음 판정에는 봉인된 전체 90회가 필요하다. 그 결과가 admission gate를 통과한 뒤에만 PPO/SAC pilot을 시작할 수 있다. 현재 smoke만으로 학습 효과를 주장하지 않는다.

VM은 결과 회수 뒤 `STOPPED`로 확인됐다. G1 smoke 창의 계산 추정치는 `0.7288 USD`이고 예약 상한은 `1.50 USD`였다. 이 수치는 관측 단가와 실행 창을 곱한 추정치이며 청구액이 아니다.
