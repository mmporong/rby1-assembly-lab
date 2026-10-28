# RJ45 BC 폐루프 정량 진단

## 결론

확인된 train/eval 계약 불일치는 없었다. 학습과 평가는 같은 `standard79` 관측 순서, 무정규화 입력, 6차원 무차원 action, 같은 물리 스케일(이동 1.5 mm, 회전 0.02618 rad), 같은 absolute tip-target 적용 경로를 사용한다. teacher 수집의 유효 행은 정책 관측과 적용 action의 timestamp 차이가 전부 0초였고, BC 평가의 action saturation도 64회 모두 0이었다.

우선 검증할 가설은 **작은 one-step MSE가 정책 롤아웃 안정성을 보장하지 않을 수 있다는 점**이다. 79차원 관측에는 이전 12차원 action이 포함되고, teacher label은 한 step 사이에 매우 조금 변한다. dev에서 이전 오른팔 action을 그대로 label로 쓰는 MSE가 `6.2425e-6`, BC MSE가 `5.7879e-6`이다. BC는 이전 action 복사보다 MSE를 7.28% 줄였고, zero actor 대비로는 99.73% 줄였다. 7.28%라는 차이는 폐루프 분포 이동 가설을 우선 시험할 근거이지만, autoregressive shift나 부분 관측성을 증명하지는 않는다.

또한 teacher label은 실제 socket·plug·오른손 EE와 frozen estimated plug-to-EE 관계로 만든 privileged target을 쓴다. 학생의 standard79에는 episode마다 숨겨진 calibration bias와 teacher target pose가 없다. 학습 데이터의 다른 episode 최근접 관측 RMS가 `0.00920`인데 대응 label RMS가 평균 `0.02086`, 최대 `0.04871`이었다. 이는 같은 수준으로 보이는 학생 관측에 서로 다른 보정값이 존재할 수 있음을 보여 주지만, 식별 불가능성의 수학적 증명은 아니다.

## 1. 계약과 구현 대조

| 항목 | 학습/수집 | 평가 | 판정 |
|---|---|---|---|
| 관측 | `build_policy_observation`의 79차원 | 같은 함수의 `observations["policy"]` | 일치 |
| 정규화 | 관측 구성 단계의 고정 단위 스케일만 사용, 추가 통계 정규화 없음 | 동일 | 일치 |
| 관측 순서 | 상대 pose 7, 관절 위치 14, 속도 14, 힘 5, grip 2, phase 8, 이전 action 12, depth 1, advance 1, joint error 14, active 1 | 동일 | 일치 |
| action | bounded 6D, 오른팔 action으로 확장 | 같은 6D actor 출력 | 일치 |
| 물리 스케일 | 이동 `0.0015 m`, 회전 `0.02617994 rad` | `_apply_policy_mating_target_pose_offsets`의 같은 cfg 값 | 일치 |
| 시간 정렬 | 이전 정책 관측을 cache한 뒤 같은 pre-physics step에서 teacher action 생성 | 관측을 받아 다음 `env.step(action)` 수행 | 일치 |
| 정책 관측 timestamp | 유효 train/dev 행 모두 global observation timestamp와 차이 0초 | 같은 environment common-step 관측 | 지연 불일치 근거 없음 |
| action clipping | dataset label 최대 절댓값 0.159 | BC 64회 saturation rate 0 | clipping 원인 아님 |
| actor 구조 | 79→256→256→128→6 ELU | 같은 actor artifact strict load | 일치 |

코드 감사 범위에서는 normalization, 관측 ordering, action scale, phase, timestamp, previous-action 정렬 오류가 확인되지 않았다.

## 2. 데이터와 actor

- train: 256 episode, 95,826개 유효 성공 행, 8개 train grasp
- dev: 64 episode, 23,687개 유효 성공 행, 2개 dev grasp
- 유효 phase: train `insert 95,570 + done 256`, dev `insert 23,623 + done 64`
- BC best epoch: 34/40
- train MSE 재계산: `6.7157e-6`
- dev MSE 재계산: `5.7879e-6`
- zero actor dev MSE: `2.1678e-3`
- 이전 action 복사 dev MSE: `6.2425e-6`
- actor와 label의 차원별 상관계수: `0.9982~0.9989`
- actor 출력 norm: 중앙값 `0.1121`, 90% `0.1605`, 최대 `0.1934`
- 평균 절대 action 오차의 물리 환산:
  - 이동: `1.61 / 1.00 / 0.93 µm`
  - 회전: `0.0195 / 0.0194 / 0.0214 mrad`

label은 매 step 천천히 변한다. dev label 변화 norm은 중앙값 `0.000987`, 90% `0.00255`다. 이 관찰은 previous-action 지속성만으로도 낮은 one-step MSE가 나올 가능성과 양립하지만, actor가 shortcut을 학습했다는 증거는 아니다.

## 3. 학생 관측 식별성과 dev shift

standard79에는 추정 상대 pose, 관절/힘, phase, 이전 action, command depth가 들어간다. 다음 값은 포함되지 않는다.

- episode-fixed calibration bias 자체
- 실제 socket/plug/right-EE로 만든 teacher virtual target pose
- privileged `teacher_gt_pose` 7차원

privileged pose는 train 95,826행과 dev 23,687행에 별도 group으로 보존됐고 학생 loader에는 들어가지 않았다. GT 누수는 확인되지 않았다.

train/dev 관측 평균 이동은 전체적으로 작지만 grasp 차이가 보인다.

- grip force: train 표준편차 기준 RMS `0.585`
- arm joint position: `0.294`
- joint reference error: `0.150`
- relative quaternion: `0.139`
- previous action: `0.101`
- relative position: `0.092`

오프라인 dev는 두 dev grasp에서 잘 맞지만 평가 seed는 수집과 다른 calibration latent case를 만든다. 숨은 bias에 대한 새 조합 일반화는 낮은 dev MSE만으로 검증되지 않는다.

## 4. 실제 폐루프 64개 비교

아래 8/64는 [보존한 사전 확인 실행](bc-only-evaluation.json)의 결과다. 같은 64개 조건·설정·actor·source를 사용한 [별도 반복 실행](bc-only-repeat-evaluation.json)은 12/64였다. 6건은 실패에서 성공으로, 2건은 성공에서 실패로 바뀌었다. 두 보고서만으로 차이의 원인을 특정할 수 없으므로 한 실행의 성공률을 반복 불변인 성능으로 해석하지 않는다. 이 절의 분석 입력과 수치는 첫 실행으로 고정하고, 본 PPO 비교에는 해당 비교에서 함께 실행한 기준선 결과를 사용한다.

| 지표 | biased zero | BC-only |
|---|---:|---:|
| 성공 | 38/64 | 8/64 |
| timeout | 26 | 56 |
| invalid | 0 | 0 |
| saturation 발생 | 0 | 0 |
| 실패 lateral 중앙값 | 73.1 µm | 66.8 µm |
| 실패 depth 중앙값 | 12.289 mm | 12.362 mm |
| 실패 orientation 중앙값 | 2.466 mrad | 4.449 mrad |
| 실패 dwell 중앙값 | 0 s | 0 s |
| 실패 insertion load 중앙값 | 0.0115 N | 0.218 N |

성공 판정은 깊이 12~13 mm, lateral 50 µm 이하, orientation 8.727 mrad 이하에 더해 상대 선속도 5 mm/s 이하, 상대 각속도 0.05 rad/s 이하, 양손 파지·리프트 유지와 하중 조건을 확인한다. 240 Hz 물리 계산에서 제어 스텝의 시작과 각 물리 스텝 끝을 포함한 9개 표본이 안정성 조건을 만족해야 연속 유지 시간 0.4초를 누적한다. BC 실패의 중앙 depth와 orientation은 각 한계를 만족하지만 lateral 중앙값은 50 µm를 넘고 최종 dwell 중앙값은 0초다. 종료 시점의 geometry와 dwell만으로 어느 중간 표본이나 속도 조건이 유지를 끊었는지는 특정할 수 없다. 힘은 25 N 한계보다 훨씬 낮고 forbidden contact는 0이므로 하중 초과가 주원인이라는 근거는 없다.

동일 64개 paired case의 성공 전이는 다음과 같다.

- zero 성공 → BC 실패: 34
- zero 성공 → BC 성공: 4
- zero 실패 → BC 성공: 4
- zero 실패 → BC 실패: 22

BC는 zero가 실패한 26개 중 4개를 구했지만 zero가 성공한 38개 중 34개를 잃었다. zero 성공→BC 실패 집단의 translation bias norm 중앙값은 `39.3 µm`였다. 이 결과는 작은 bias에서 nominal controller가 성공하던 조건에 BC action을 더했을 때 성공이 줄어든 현상까지 보여 준다. action 시계열이 없어 비영 action이 lateral/orientation 조건을 흔든 원인이라고 단정할 수는 없다.

paired BC-zero 중앙 변화:

- elapsed: `+10.10 s`
- depth: `+42.6 µm`
- lateral error: `+28.4 µm`
- orientation error: `+2.50 mrad`
- grip peak: `+0.301 N`
- insertion load peak: `+0.179 N`
- dwell: `-0.4 s`

## 5. 확인된 사실과 남은 가설

### 확인된 사실

1. 학습/평가 관측·정규화·action scale·timestamp 계약 오류는 발견되지 않았다.
2. BC action은 작고 saturation되지 않았지만, 50 µm lateral gate에는 충분히 큰 지속 편향을 만들 수 있다.
3. one-step BC MSE는 zero보다 99.73% 좋고 이전 action 복사보다 7.28% 좋다. 후자는 폐루프 분포 이동을 우선 시험할 근거이며 shortcut이나 부분 관측성의 증명은 아니다.
4. teacher label은 학생에게 없는 privileged geometry에 의존한다.
5. 실제 paired 평가에서는 BC가 34개 기존 성공을 잃고 4개 실패만 회복했다.

### 아직 검증되지 않은 가설

1. **autoregressive observation shift**: actor 오차가 previous-action 필드를 통해 다음 입력을 바꾸며 teacher rollout 밖으로 이동한다. 평가 action 시계열이 없으므로 현재 보고서만으로 궤적을 증명할 수 없다.
2. **숨은 bias의 부분 관측성**: standard79만으로 새 calibration case의 teacher correction을 안정적으로 식별하기 어렵다. 최근접 관측의 label 충돌이 이를 지지하지만 불가능성 증명은 아니다.
3. **held-grasp dev shift**: 8개 train grasp와 2개 dev grasp의 관절·grip 분포 차이가 기여할 수 있다. 다만 offline dev MSE가 낮아 grasp와 숨은 bias의 영향을 분리할 수 없다.

첫 PPO-only·BC→PPO 비교는 BC→PPO의 초기 파지 안정화 검사에서 중단됐다. PPO-only의 200-update 결과는 32/64·27/64·31/64였으며, 두 학습 방법의 비교 결론은 아직 없다. [중단 기록](../paired-first-attempt/README.md)을 보존하고, 제한적 초기화 복구를 검증한 새 소스에서 두 군을 모두 다시 실행한다. 같은 개발 64조건으로 PPO가 BC의 폐루프 한계를 보완하는지 확인한다. 현재 결과만으로 BC dataset이나 teacher 구현 자체가 잘못됐다고 결론 내릴 근거는 없다.

### 이전 행동 입력에 대한 오프라인 민감도

성공한 교사 시연의 INSERT 관측에서 다른 입력을 고정하고, 오른손의 이전 행동 6개 값만 조금 바꿔 actor 출력을 다시 계산했다. 중앙 차분 간격은 무차원 행동값 `0.001`이며, train과 dev에서 일정 간격으로 최대 1,024행을 골랐다. 물리 시뮬레이터를 돌린 시험은 아니다.

| 항목 | train | dev |
| --- | ---: | ---: |
| INSERT 관측 행 | 95,570 | 23,623 |
| 국소 미분 계산에 사용한 행 | 1,017 | 985 |
| actor의 교사 행동 MSE | `6.726e-6` | `5.798e-6` |
| 이전 행동을 그대로 복사한 MSE | `6.892e-6` | `6.259e-6` |
| 이전 행동 입력을 0으로 바꾼 MSE | `1.072e-3` | `1.088e-3` |
| 국소 Jacobian 최대 특잇값 중앙값 | 0.9950 | 0.9949 |
| 최대 특잇값이 1을 넘은 표본 비율 | 32.9% | 34.7% |

이 입력을 0으로 바꾸면 예측 오차가 커졌으므로 학습된 actor가 이전 행동값에 민감하다는 점은 확인했다. 다만 0으로 바꾼 입력은 학습 분포를 벗어날 수 있다. 국소 미분이 1을 넘는 표본도 전체 로봇의 불안정이나 폐루프 실패 원인을 입증하지 않는다. 실제 행동이 다음 관측과 접촉에 미치는 영향은 rollout 시계열로 따로 확인해야 한다. 계산 결과와 입력·소스 해시는 [`action-sensitivity.json`](action-sensitivity.json)에 보존했다.

## 근거와 재현 범위

상세 실행시간은 [`diagnostic-results.json`](diagnostic-results.json)에 기록했다. CPU에서 HDF5 119,513개 유효 행을 읽고 actor 추론과 두 평가 보고서의 64개 paired outcome을 분석했다.

실제 평가 실행의 provenance는 현재 분석 환경과 분리해 검증했다.

- biased-zero 평가는 commit `104721acf2222c1cfe8ec86cd030f5116a4fef93`, evaluator SHA-256 `3a3b96325ecfd32c9e4603eda04713646d35f88fc11cd73dbf9ef23842d6c238`에서 실행됐다.
- BC-only 평가는 commit `8512a5932547b5495325b5ff0b6894fb072db98e`, evaluator SHA-256 `ca578d7db304781e382aea33d1154545723eac5e5ae4bc512d3e0186c25aa1c8`에서 실행됐다.
- 각 보고서에 내장된 36개 source SHA를 해당 commit의 `git show` byte와 전부 대조했다. 두 실행의 `src/` 물리·환경 소스는 모두 같았고, runtime source 차이는 BC 입력을 TensorDict에서 `policy` tensor로 선택하도록 고친 `scripts/evaluate_policy.py`뿐이다. commit 사이의 나머지 변경은 이 수정의 회귀 테스트다.
- 현재 분석 환경의 evaluator SHA-256 `a7b8935a40d64452bcf19437c3cf4616c7ff9b1521be3106491700c47153c1e3`는 분석 코드 provenance이며 두 실제 평가 실행에 귀속하지 않았다.

주요 근거 SHA-256:

- train HDF5: `ae2884f3635d6177b278156f8cfadf1d3d29db54115cc99a938927ea3d6b1efc`
- dev HDF5: `680d18be628c2eb9bb6adecf0ff719f48d99b6128ee80ca58af5485235f60830`
- BC actor: `f0fd65b56691748dce5bee378152524bf84f3f9daed2f1f36dd43e0e189233b5`
- BC training report: `32aa18808dfb9ae2ccc6bb4019f24e9c5782005ee69c99889af7ee2816467bbf`
- BC-only 평가: `5c61a1ca92cda3f473f61888d70f308eb6a920f3d26acae47d4413cfbc90413f`
- biased-zero 평가: `ea6c92c0cd2ac70f04429842244ffadd65a1bf12af455c965efbeaa2c3b3aea5`

## 공개 파일과 재현

Git에는 대용량 HDF5와 actor checkpoint를 넣지 않았다. 네 입력은 검증한 retrieval archive `rj45-4c-teacher-bc-collector-recovery-evidence-20261028.tar.gz`에서 가져오며, archive SHA-256은 `a5f6ce84a0534931302cc000b2d74ce87ea775398ed9b2f785b4f2f23953ad0c`이다. 세 평가 보고서는 이 폴더에 byte 보존 사본으로 포함했다.

- [`bc-only-evaluation.json`](bc-only-evaluation.json): BC-only 64개 평가 원본
- [`bc-only-repeat-evaluation.json`](bc-only-repeat-evaluation.json): 같은 64개 조건의 별도 BC-only 반복 평가 원본
- [`biased-zero-evaluation.json`](biased-zero-evaluation.json): biased-zero 64개 평가 원본
- [`diagnostic-results.json`](diagnostic-results.json): 재계산한 상세 통계와 실행 provenance
- [`action-sensitivity.json`](action-sensitivity.json): 교사 관측에서 이전 행동 입력의 국소 민감도
- [`source-manifest.json`](source-manifest.json): 공개 파일과 미포함 입력의 SHA-256, 크기, 출처
- [`../../../../scripts/analyze_rj45_bc_closed_loop.py`](../../../../scripts/analyze_rj45_bc_closed_loop.py): 입력 SHA와 평가 실행 source map을 fail-closed로 확인하는 재현 스크립트

archive를 안전한 위치에 푼 뒤 다음처럼 실행한다. 스크립트는 여섯 입력의 SHA-256이 검토된 값과 다르거나, 두 평가 보고서의 source map이 지정 commit의 `git show` byte와 다르면 중단한다.

```bash
cd "$HOME/rby1-assembly-lab-rj45-data"
ARTIFACT_ROOT="$HOME/rby1-assembly-lab/artifacts/cloud/retrieved-4c-teacher-bc-collector-recovery-20261028/payload/artifacts/rj45-4c-teacher-bc-collector-recovery-v1-20261028"
PYTHONPATH=src python scripts/analyze_rj45_bc_closed_loop.py \
  --repo "$PWD" \
  --train-h5 "$ARTIFACT_ROOT/teacher-train.h5" \
  --dev-h5 "$ARTIFACT_ROOT/teacher-dev.h5" \
  --actor "$ARTIFACT_ROOT/bc/bc_actor_init.pt" \
  --training-report "$ARTIFACT_ROOT/bc/training_report.json" \
  --bc-eval docs/evidence/rj45-demonstrations-20261028/bc-diagnostic/bc-only-evaluation.json \
  --zero-eval docs/evidence/rj45-demonstrations-20261028/bc-diagnostic/biased-zero-evaluation.json \
  --output /tmp/rj45-bc-closed-loop-diagnostic.json

PYTHONPATH=src python scripts/analyze_rj45_bc_action_sensitivity.py \
  --repo "$PWD" \
  --teacher-root "$ARTIFACT_ROOT" \
  --output /tmp/rj45-bc-action-sensitivity.json
```

오프라인 MSE가 낮다는 사실은 폐루프 성공률 개선을 뜻하지 않는다. 이 결과에서는 BC dev MSE가 zero actor보다 99.73% 낮았지만, 동일 64개 조건의 폐루프 성공은 biased-zero `38/64`, BC-only `8/64`였다. 원인으로 제시한 autoregressive observation shift, 숨은 bias의 부분 관측성, held-grasp dev shift는 후속 시험이 필요한 가설이다.
