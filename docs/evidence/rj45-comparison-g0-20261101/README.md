# RJ45 PPO·SAC 공통 실행 경로 G0 Isaac smoke

상태: **G0 완료**. 2026-11-01 기준 4개 Isaac 환경에서 PPO와 SAC의 공통 79D 관측·오른팔 6D residual 실행 경로, 전이 수집, optimizer 실행, 유한 손실과 비영 파라미터 변화를 확인했다. 이 결과는 학습 효과나 PPO·SAC 성능 우열을 입증하지 않는다.

## 실행 결과

| 방법 | 환경 | 학습 전이 | algorithm update | optimizer step | 유한 손실 | 파라미터 변화 |
|---|---:|---:|---:|---:|---|---|
| PPO | 4 | 64 | 2 | 8 | policy/value loss 유한 | 25,484/27,532개 변경, L2 0.1705 |
| SAC | 4 | 64 | 64 | 192 | critic/entropy/policy loss 유한 | 25,484/27,532개 변경, L2 0.4455 |

PPO와 SAC의 update 수 차이는 짧은 G0 tensor·optimizer 경로를 각각 통과시키기 위한 smoke 설정이다. 같은 경험·계산 예산의 성능 비교 설정이 아니며, 이 수치로 알고리즘 효율이나 성공률을 비교하지 않는다.

실행 환경은 Isaac Sim 배포판 `5.1.0.0`, Isaac Lab **2.3 release 계열**, 설치된 `isaaclab` 배포판 metadata `0.47.2`, `skrl 2.1.0`, `torch 2.7.0+cu128`이다. `2.3`은 프로젝트가 고정한 Isaac Lab release 명칭이고 `0.47.2`는 설치 패키지 metadata이므로 같은 버전 표기로 합치지 않는다.

## 검증 경계

- 통합 CPU suite: `75 passed`.
- 독립 G0 verifier: `PASS`.
- G1 core architect: `CLEAR`.
- G1 runner CPU prelaunch review: `APPROVE`.
- G1 실제 Isaac task-admission smoke: **미실행**.

따라서 G0 공통 학습 실행 경로는 완료됐고 G1은 물리 gate 검증 중이다. G2 반복 학습, G3 봉인 평가, 영상·게시물 준비는 아직 완료되지 않았다.

## 증거 파일

- [`g0-isaac-comparison-smoke.json`](g0-isaac-comparison-smoke.json): 원본 report를 바이트 변경 없이 보존했다. SHA-256은 `cd81227390670668976dd66b26d1cd0ed576dddc2b28efd9679c94091e7e2791`이다.
- [`summary.json`](summary.json): 원본 요약을 바이트 변경 없이 보존했다. SHA-256은 `74b747a30b920aeaa4ab9fcd98e9161a481a69acdabaaf0d5f223b040da77499`다.

G0 GPU 실행의 supervisor와 child exit code는 모두 0이었다. target은 `2026-10-31T16:04:21.816443Z`에 `STOPPED`로 확인됐고 종료 guard의 timer와 service는 모두 `inactive/not-found`였다. 결과 회수도 완료됐다.

이번 실행에 잡은 `6 USD`는 이 G0 smoke 한 건의 예약 상한이다. G1의 새 GPU 예산은 별도 심사 중이며 이 예약을 G1로 확대하지 않는다. 계정·청구 관련 비공개 파일은 이 공개 evidence 묶음에 포함하지 않았다.
