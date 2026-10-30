# RJ45 개선·일반화·포트폴리오 실행 계획

2026-10-30 사용자 지시에 따라 공개 마감보다 강화학습의 추가 기여 검증을 우선한다. 다음 실행의 원인 분리 순서, 비용·런타임 확인과 BC-only 대비 PPO 효과 판정은 [강화학습 기여 검증 재개 계획](rj45-rl-contribution-plan.md)을 따른다. 아래 기존 실험 계약과 동결 protocol·결과는 당시 조건으로 보존한다.

## 목표와 현재 위치

사용자가 2026-10-28 승인한 4·5·6단계를 하나의 Codex goal로 수행한다. 기존 제어에 추가한 학습 보정의 개선 여부를 검증하고, 개발 기준을 통과한 후보가 생기면 여러 파지 상태와 미사용 조건에서 재검증한다. 후보가 기준을 넘지 못하면 실패 원인과 한계를 결과로 남긴다. 실물 이동이나 공개 게시·알림은 범위에 포함하지 않는다.

| 단계 | 현재 상태 | 근거 |
| --- | --- | --- |
| 4B 병렬 규모 측정 | 완료. 단일 GPU에서 64·128·256환경 처리량은 223.39·460.75·868.43 전이/초였다. | [규모 확대 결과](evidence/rj45-refinement-20261028/scale-results/README.md) |
| 4B 설정 비교 | 완료. 96 update에서 F가 11/32로 가장 높았다. | [설정 비교](evidence/rj45-refinement-20261028/scale-results/refinement-summary.json) |
| F 새 seed 반복 | 완료. 200 update 결과가 7/32·10/32·8/32로 승격 기준을 통과하지 못했다. 최종 조건은 봉인 상태다. | [후속 결과](evidence/rj45-refinement-20261028/scale-results/f-followup-summary.json) · [독립 검증](evidence/rj45-refinement-20261028/scale-results/independent-audit.json) |
| 256환경 병렬 PPO 영상 | 완료. 16초·60 fps 개요와 근접 영상을 별도 동일 설정 실행으로 기록했다. | [미디어 검증](evidence/rj45-refinement-20261028/parallel-media.json) |
| generation v2 pilot | 통과. 시뮬레이터 물리 파지 4개가 최대 clique 4를 이뤘다. | [생성 계약](evidence/rj45-demonstrations-20261028/grasp-generation-v2-protocol.json) · [실제 pilot 결과](evidence/rj45-demonstrations-20261028/generation-v2-pilot-results/README.md) |
| production 파지 | 완료. 후보 20개 중 train 8·dev 2·final 2의 총 12개 파지를 선정했다. | [선정·거리 검증](evidence/rj45-demonstrations-20261028/generation-v2-bank-results/README.md) |
| teacher·BC | 완료. train 256회·dev 64회 시연이 유효·독립 성공 판정을 받았다. BC dev MSE는 `5.7879e-6`이다. | [수집·학습 결과](evidence/rj45-demonstrations-20261028/teacher-bc-results/README.md) |
| BC-only 폐루프 평가 | 완료. 기본 평가 12/64, 앞선 같은 계약 평가 8/64. 기준 제어는 38/64다. | [실패 진단과 반복 차이](evidence/rj45-demonstrations-20261028/bc-diagnostic/README.md) |
| 첫 PPO·BC 초기화 PPO | 중단. PPO 3개 seed는 200 update를 마쳤고 32/64·27/64·31/64였다. BC 학습 중 warmup 검사가 실패했다. | [실패·부분 결과 보존](evidence/rj45-demonstrations-20261028/paired-first-attempt/README.md). 복구 계약 검증 뒤 두 군을 모두 다시 실행한다. |

4B에서 성능 승격이 막혔으므로 4C는 여러 파지 상태의 시연으로 actor를 초기화한 뒤 같은 학습량의 PPO 대조군과 비교한다. 아래 절은 완료된 4B 계약과 진행 중인 4C·5·6단계의 검증 기준을 함께 보존한다.

## 4. 원인 분석과 개선 실험

### 먼저 고정할 사실

- 평가 시드와 파지 artifact를 맞춘 비교를 사용한다. 성공 기준 strict-v1의 12~13 mm 깊이, 50 µm 횡오차, 0.5° 방향 오차, 0.4초 유지 및 파지·하중 조건은 그대로 유지한다. 50 µm는 프로젝트 기준이며 제조사 규격이 아니다.
- 기존 79D 관측은 단위별 수동 scaling을 한다. empirical normalization이 꺼져 있다는 사실만으로 입력 정규화가 없다고 해석하지 않는다.
- 79D는 관절 기반 고정 파지 추정기와 episode 고정 보정 오차를 사용한다. 삽입 접촉의 3축 합력 proxy와 손별 파지 proxy가 포함되며 손목 6축 F/T 측정값이 아니다.
- 기존 보고서의 1.5 mm·1° observation-noise 값은 상속된 기본 환경 설정이다. RJ45 wrapper의 정책 관측에는 그 per-step 잡음이 적용되지 않았다. 새 실행 기록에는 실제 적용 여부와 episode 고정 bias를 구분한다.
- 고정 파지와 scripted ALIGN의 관절 자세에는 보정 오차의 단서가 남을 수 있다. 현재 관측이 본질적으로 비식별이라고 단정하지 않고 rollout 데이터로 확인한다.

### 4A: 기존 학습의 실패를 재현 가능한 수치로 정리

모델·보고서·TensorBoard 원본 해시와 함께 seed별 learning rate, 탐색 표준편차, reward 항, 마지막 깊이·횡오차·각도·유지시간, 기준 제어와의 성공 교집합을 계산한다. 관찰과 원인 가설을 분리한다. 최종 시점만 기록된 값으로 전 구간의 실패 원인을 단정하지 않는다. Adaptive learning rate 상승은 라이브러리의 의도된 동작이며 성능 저하의 인과성은 대조 실험으로 확인한다. 기존 saturation 0은 결정론적 평가의 기록일 뿐 학습 중 clipping이 없었다는 증거가 아니다. 새 학습에는 wrapper clip 전후 분포와 활성 단계의 clip 비율을 기록한다.

### 4B: 작은 설정 차이를 순서대로 비교

첫 개선 후보는 관찰된 adaptive learning-rate 급상승을 분리하는 고정 learning rate다. 그다음 탐색 진폭, 횡방향 보상 해상도를 한 항목씩 바꾼다.

| 군 | 이전 군에서 바뀌는 항목 | 질문 |
| --- | --- | --- |
| Z1 재현 | 기존 설정 그대로 | 동일 실행 조건의 기준이 재현되는가 |
| F | adaptive 대신 fixed learning rate 0.0003 | 급격한 step-size 변화가 성능을 제한했는가 |
| N | F의 초기 탐색 표준편차 0.1 → 0.02 | 50 µm 정렬에 비해 큰 탐색이 성공 경험을 줄였는가 |
| R | N의 lateral potential scale 0.5 mm → 0.05 mm | 마지막 미세정렬에 대한 보상 구분이 부족했는가 |

이 순서는 각 인접 군의 조건부 효과를 비교한다. 모든 조합을 시험한 factorial 효과나 독립적인 주효과로 해석하지 않는다. entropy·성공 기준·기준 제어·action bound·평가 예산은 이 비교에서 함께 변경하지 않는다. 네 군 모두 같은 seed로 96회까지 연속 학습하고 20·96회 모델을 평가한다. 20회 결과는 안정성과 학습 신호를 보는 중간 기록이며 성능 선별에는 쓰지 않는다. 성능 후보는 유효한 96회 모델만 비교하고 최대 하나를 선택한다. [규모 확대·비교 규칙](evidence/rj45-refinement-20261028/scale-refinement-protocol.json)에 seed·동률 규칙·승격 조건과 설치를 포함한 6시간 마감을 고정한다. 성공 여부로 인프라 재시도를 결정하지 않는다. R의 50 µm는 결과를 보기 전에 고정한 가설이며 보상 곡선·유한값·개선 방향을 먼저 검사한다.

### 4B 실행 규모와 병렬 촬영

사용자의 규모 확대 지시에 따라 L40S 4장 AWS 장비에서 실행했다. 첫 Nebius 128코어 장비는 공급 단계에서 실패해 AWS의 96코어 구성으로 대체했다. 실제 사용 장비와 마감·단가는 실행 계약으로 고정했다. 64·128·256환경에서 PPO 갱신당 총 8,192 transition을 맞춰 처리량을 측정했고, 환경당 rollout은 각각 128·64·32 step이었다. 첫 3회 갱신을 준비 구간으로 제외하고 다음 5회 학습의 처리량을 비교했다. 물리·접촉·유한값 검사를 통과한 구성 중 최고 처리량의 90% 이상인 가장 작은 환경 수를 고르는 규칙에 따라 256환경을 선택했다. 성공률과 보상은 규모 선택에 쓰지 않았다.

벤치마크의 원시 갱신 시간·유효성 검사·선택 결과를 저장하고 `num_envs × num_steps_per_env = 8192`를 확인했다. 네 군의 실제 설정, 소스, 명목 실행·파지·접촉 증거, GPU·CPU 배정과 마감은 별도 실행 계획과 입력 SHA manifest로 고정했다. 전체·근접 촬영도 명령, seed, 설정·소스·영상·JSONL 해시를 기록했다.

환경 수와 GAE rollout 길이가 바뀌면 이전 64환경 실험의 완전 재현은 아니다. 새 Z1·F·N·R은 같은 환경 수, rollout 길이, 갱신 횟수와 미니배치 크기를 적용한다. 기준 제어를 먼저 평가하고, 각 GPU에 프로필 하나씩 배치해 네 실험을 동시에 실행한다. 하나의 프로세스 안에서는 여러 독립 환경의 경험을 모아 정책 하나를 갱신한다. 서로 다른 GPU의 네 프로세스는 다른 설정을 비교하는 독립 학습이다.

Isaac Sim 5.1과 Lab 2.3에서는 GPU를 숨기는 환경변수 대신 `--device cuda:N`으로 GPU를 고르고, AppLauncher의 renderer multi-GPU를 끈다. 실제 프로세스 PID와 GPU UUID를 대조한다. 촬영은 처리량 측정에서 분리하며, 한 시뮬레이터의 전체 환경 격자와 근접 구도를 실제 프레임으로 기록한다. 환경 ID·물리 시각·PPO 갱신 수·삽입 상태를 같은 시각의 JSONL에 저장한다. GPU 네 개의 영상이나 순차 실행을 합성해 한 시뮬레이터의 병렬 장면처럼 표시하지 않는다.

같은 개발 32조건을 반복 사용하므로 선택 편향이 생길 수 있다. 개발 결과는 후보 선택용이며 일반화 성능으로 표현하지 않는다. 후보 수·순서·공통 checkpoint·모델 SHA를 고정하고 미사용 최종 조건은 한 번만 평가한다.

### 4C: 성공 시연과 관측 진단

시연을 사용하기 전에 여러 유효한 물리 파지 상태를 만들고 상태 ID와 보정 seed로 train/dev/final을 분리한다. 같은 파지의 다른 timestamp를 독립 파지로 세지 않는다. 좌우 wrist↔object 변환, 관절 벡터, 접촉 body/patch와 force proxy, 두 부품 상대 자세로 상태를 기술한다. 안정화 프레임 수·미끄럼·복원 오차 허용치, 파지 간 최소 거리, split별 상태·조건 수는 데이터 생성 전에 별도 계약으로 고정한다. 채택에는 정책 성능을 쓰지 않고 물리 유효성만 사용한다. 한 state의 모든 rollout·교사 레이블·normalization 통계는 한 split에만 속한다. 소규모 표본은 여러 파지 사례 검증으로 표현한다.

교사는 시뮬레이터 정답 상태로 제한된 6D 삽입 보정을 계산할 수 있다. 학생의 입력은 실제 정책 시각의 79D 관측으로 제한하고, 정답 pose·bias는 교사 레이블과 평가용 별도 필드에 보관한다. 적용 전 제안 action과 제한 후 적용 action, 관측 시각·유효성, 접촉, 단계, 파지·설정·소스 해시, 독립 성공 판정을 기록한다. 성공 시연에만 BC 손실을 적용하되 전체 시도와 실패 비율은 보존한다.

`기준 제어 / PPO / BC-only / BC 초기화 후 PPO`를 비교한다. BC의 교사 시뮬레이션 비용과 gradient update는 PPO rollout 수와 별도 보고한다. BC의 개발 손실만으로 성공을 판단하지 않고 물리 rollout으로 확인한다. 실패 상태의 분포 이동이 확인되면 교사 재질의나 감쇠 시연 손실을 별도 변경으로 검증한다.

정답 bias를 예측하는 probe와 85D privileged 대조는 원인 분리 진단이며 배포 관측이나 성능 후보에 추가하지 않는다. 진단을 먼저 수행해 최적화와 관측 정보의 문제를 구분한다. 79D의 과거 접촉 정보가 필요한지는 동일 데이터·행동·보상으로 causal history 대조를 통해 확인한다. 현재 입력에 이미 직전 action이 있으므로 중복 추가하지 않는다. Normalization과 probe fitting에는 train split만 사용한다.

## 5. 미사용 조건과 양팔 협조 검증

1. train/dev/final 파지 상태의 물리 유효성, artifact ID, 접촉 안정성과 소스 계보를 확인한다. 각 split의 calibration seed도 분리한다.
2. 모든 비교군은 같은 task·파지·조건 ID·실제 bias vector·초기 상태·시간/재시도 예산을 사용한다. 평가 actor는 결정론적으로 실행한다. PPO 군의 환경 transition·update·checkpoint 나이를 맞추고 기준 제어도 새 환경에서 함께 평가한다.
3. 후보 선택은 개발 결과와 공통 checkpoint 규칙으로만 한다. 세 학습 seed의 성능·편차를 모두 보고한다.
4. 먼저 오른손 6D 보정 효과를 확인하고, 같은 6D 상대 보정을 양손에 배분하는 방식과 지지팔 고정을 비교한다. 상대 SE(3) 보정 권한과 양손을 더한 Cartesian 이동 예산을 동일하게 유지하며, 손마다 기존 bound를 주어 권한이 두 배가 되지 않도록 한다. action 차원을 늘리는 12D 정책은 별도 실험이며 여러 요소를 동시에 바꾸지 않는다.
5. 후보·모델 SHA·집계·성공 판정·실패 처리 규칙을 고정한 뒤 미사용 파지와 보정 조건을 한 번 평가한다. 최종 결과를 본 뒤 같은 평가에 맞춰 재학습하거나 모델을 고르지 않는다.
6. 마찰·배치·지연 등의 범위는 근거와 단위를 설정 파일에 기록한다. 가정한 물성을 실물 보정값이라고 표현하지 않는다.

기존 단일 파지 개발 실험과 다중 파지 실험은 서로 다른 검증 범위다. 한 범위의 조건부 성공률을 다른 범위의 일반화 성능으로 합치지 않는다. 모든 조건을 통과한 후에도 실제 센서·실물 전이는 별도의 검증 대상이다.

## 6. 검증된 실행의 포트폴리오 산출물

- 같은 조건의 기준 제어와 학습 정책 비교 영상 1편. 전체 시도와 깊이·횡오차·힘 proxy를 동기 표시하고, 표시한 정책 SHA와 실제 실행 모델을 연결한다.
- 원인과 측정값을 함께 볼 수 있는 실제 정책 실패 영상 2편. 개발 실패 유형별 대표 episode는 해당 유형의 오류 크기 중앙값에 가까운 순서로 고르고 ID와 선택 규칙을 남긴다. 두 유형이 없으면 동일 유형의 서로 다른 조건 2개로 표시한다. 명령 오류 주입 시험은 실제 정책 실패와 구분한다.
- 성공률, 힘 proxy 최대값, 완료 시간, 시드별 편차와 조건별 결과 그래프. 실패·무효 사례를 분모에서 숨기지 않는다.
- 논문 아이디어, 프로젝트에서 바꾼 부분, 개선 여부·한계, 재현 명령을 설명하는 문서.
- 렌더 프레임 시각·누락 검사와 소스·설정·모델·영상 SHA. 60 fps 공유본은 실제 기록 프레임에 근거한다.
- 완료: 하나의 Isaac Sim에서 256개 환경을 동시에 학습하는 개요·근접 영상. 두 구도는 별도 동일 설정 실행이며 병렬 환경 수, 시뮬레이션 시각, 현재 rollout 전 완료 update 수를 표시한다. [미디어 검증](evidence/rj45-refinement-20261028/parallel-media.json)

최종 평가로 승격하지 못하면 개발 실험 영상으로 표시한다. 개선 실패를 성공으로 포장하지 않고 원인·방법·측정 결과·한계를 담은 포트폴리오를 만든다.

## 실행·검증·종료

일반 Codex native goal과 저장소 계획을 사용한다. 장기 GPU 프로세스는 실험별 마감·예상 compute 비용·정확한 인스턴스 ID·산출물 회수 절차를 기록한다. 기존 다른 프로젝트 인스턴스는 조작하지 않는다. 로컬 파일과 원격 산출물 SHA를 대조하고 인스턴스 STOPPED를 확인한다.

각 변경은 작성과 독립 리뷰를 분리하고 위험에 맞는 테스트·물리 검사를 통과해야 실험에 사용한다. 필요한 source·config·asset 증거가 바뀌면 명목 동작과 파지 복원·접촉 검사를 다시 실행한다. 검증한 변경만 경로별로 커밋하고 승인된 origin/main에 push한 뒤 SHA를 대조한다.

## 적용 근거

- [RSL-RL 3.0.1 PPO](https://github.com/leggedrobotics/rsl_rl/blob/2fc1f78bc1d796ffa8f07ce6b09898227db284bb/rsl_rl/algorithms/ppo.py): KL에 따른 adaptive learning rate 구현. fixed와 adaptive 비교의 구현 근거다.
- [DAPG, RSS 2018](https://www.roboticsproceedings.org/rss14/p49.pdf): BC 초기화와 시연을 활용한 정책 최적화. 이 프로젝트의 BC→PPO는 논문 전체 재현이 아니다.
- [AutoMate, RSS 2024](https://roboticsproceedings.org/rss20/p064.pdf): 조립 시연과 학습의 결합 방향. 다른 로봇의 공개 행동을 RB-Y1 시연으로 표시하지 않는다.
- [Isaac Sim 5.1 GPU 선택](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_faq.html): renderer GPU 인덱스와 multi-GPU 설정. 고정 [Lab AppLauncher 소스](https://raw.githubusercontent.com/isaac-sim/IsaacLab/3c6e67bb5c7ada942a6d1884ab69338f57596f77/source/isaaclab/isaaclab/app/app_launcher.py)를 함께 대조한다.
- [Isaac Lab v2.3.0](https://github.com/isaac-sim/IsaacLab/releases/tag/v2.3.0): 고정 런타임의 버전 근거.

진행 상태: 4B 비교와 병렬 PPO 촬영, generation v2의 물리 파지 12개 선정, 교사 시연과 BC 학습·물리 평가를 마쳤다. 낮은 BC 예측 오차가 폐루프 성공률 개선으로 이어지지 않아 PPO와 BC 초기화 PPO를 비교했으나, BC 학습 중 초기화 검사 실패로 캠페인이 중단됐다. 동일 조건의 제한적 초기화 복구를 검증한 뒤 두 군을 새 소스에서 다시 실행한다. 지지팔 제어 비교와 실제 정책 실패 영상 분석은 후속 작업이다. 미사용 최종 조건은 봉인했으며, 개발 기준을 넘은 후보가 있을 때만 평가한다.
