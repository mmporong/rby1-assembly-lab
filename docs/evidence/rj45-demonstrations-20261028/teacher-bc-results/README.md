# 4C 교사 데이터와 BC 초기화 결과

이 폴더는 held-grasp RJ45 삽입 단계에서 교사 데이터를 수집하고, 그 데이터로 79차원 관측에서 6차원 보정 행동을 예측하는 BC actor를 학습한 결과를 정리한다. 실행 source는 `104721acf2222c1cfe8ec86cd030f5116a4fef93`이며 2026-10-28 09:08:37 UTC에 완료됐다.

## 수집 결과

| 분할 | episode | 유효·독립 성공 | 학습 가능 row | 보존된 warm-up invalid row | grasp 구성 |
| --- | ---: | ---: | ---: | ---: | --- |
| train | 256 | 256 | 95,826 | 3,385 | 8종 × 32 episode |
| dev | 64 | 64 | 23,687 | 706 | 2종 × 32 episode |

모든 계획 episode를 보존했으며 terminal invalid는 0건이다. train과 dev의 grasp ID 및 `(bias_case_id, seed)` 조합은 겹치지 않는다. HDF5에는 교사 정답 pose를 privileged group으로 분리했고, student loader가 반환한 모델 입력은 `standard79` 관측과 적용된 6차원 행동뿐이다. 관측과 행동은 전 row에서 finite였고 행동 절댓값 최댓값은 train `0.15936`, dev `0.14623`이었다.

여기서 320/320 성공은 교사 rollout이 고정된 held-grasp 삽입 범위에서 완료됐다는 뜻이다. BC 정책이나 PPO 정책이 폐루프 평가에서 성공했다는 결과가 아니다. 실패 attempt가 없었으므로 이 데이터만으로 실패 복구 능력이나 범위 밖 일반화를 주장할 수 없다.

## BC 학습 결과

- 구조: `79 → 256 → 256 → 128 → 6`, ELU
- recipe: 40 epoch, batch 1,024, learning rate `3e-4`, seed `20261120`
- optimizer update: 3,760회, sample visit: 3,833,040회
- 초기 dev MSE: `2.1678186e-3`
- 최저 dev MSE: `5.7878997e-6` (34 epoch)
- 마지막 dev MSE: `6.5741306e-6`
- 마지막 train MSE: `7.4776426e-6`

최저 dev MSE는 초기값의 약 1/374.5이며 99.73% 감소했다. 선택 규칙은 dev MSE가 가장 낮은 epoch를 고르고 동률이면 먼저 나온 epoch를 택하는 방식이다. [`learning-curve.csv`](learning-curve.csv)에 40 epoch 전체 값을 보존했다.

저장물은 `rj45_bc_actor_init_v1` BC actor 초기화 artifact다. PPO checkpoint가 아니며 critic, policy std, PPO optimizer를 학습하지 않았다. privileged 교사 pose도 student 입력에 넣지 않았다. 관측 유사 표본의 label RMS가 평균 `0.02086`으로 나타난 진단은 79차원 관측만으로 같은 보정 행동을 완전히 구분하기 어려울 수 있음을 보여 주지만, 불가능성이나 실제 정책 성능을 증명하지는 않는다.

## 계보와 봉인 상태

- BC artifact SHA-256: `f0fd65b56691748dce5bee378152524bf84f3f9daed2f1f36dd43e0e189233b5`
- actor state SHA-256: `c28cf06e444c9939c4449f00a1da28bdf69162264f050928aa12ee3b1ee22313`
- train dataset SHA-256: `ae2884f3635d6177b278156f8cfadf1d3d29db54115cc99a938927ea3d6b1efc`
- dev dataset SHA-256: `680d18be628c2eb9bb6adecf0ff719f48d99b6128ee80ca58af5485235f60830`
- bank SHA-256: `c2fbcfea9528f4a9cefb8c4c63aba6cc561fe2f3095a0fa16b7456dda8da54d5`
- task SHA-256: `7ac32092e6e723be066c80c53aecbbd873792e849ce42b742dbafd92cb3af88b`
- training config SHA-256: `0d5f5a07dcdfe74dfbee58852af0ad1076480cc4f91eaeb990f5e535fc61a179`

교사 수집 단계에서는 final policy seed를 열지 않았고 PPO도 시작하지 않았다. HDF5·actor·로그 원본은 Git에 넣지 않고 검증한 로컬 retrieval에 보존했다. archive 21개 파일은 생성 전후 해시가 같았으며, 안전 추출 뒤 크기와 SHA-256을 다시 대조했다.

## 파일 안내

- [`derived-summary.json`](derived-summary.json): HDF5 row·grasp·phase 분포와 BC 핵심 수치의 재검증 결과
- [`bc-training-report.json`](bc-training-report.json): 40 epoch 원본 학습 보고서
- [`learning-curve.csv`](learning-curve.csv): epoch별 train/dev MSE와 update 수
- [`run-summary.json`](run-summary.json): 원격 단계 완료 상태와 데이터셋·artifact SHA
- [`teacher-stage-manifest.json`](teacher-stage-manifest.json), [`pre-bc-manifest.json`](pre-bc-manifest.json): 수집과 BC 실행 전 고정 계약
- [`retrieval-verification.json`](retrieval-verification.json): archive·HDF5·BC artifact 재검증 범위
- [`source-manifest.json`](source-manifest.json): 원본 21개 파일과 공개 파일의 크기·SHA-256
