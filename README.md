# RB-Y1 Assembly Lab

휴대형 LAN 어댑터를 왼손으로 지지하고 오른손으로 RJ45 플러그를 삽입하는 양팔 조립 실험이다. 실제 규격의 작은 커넥터를 기본 RB gripper로 잡고, 접촉으로 막힌 상황에서 제한된 양팔 보정이 도움이 되는지 확인하는 것이 목표다.

RJ45 장면은 TE 공식 잭 CAD와 플러그 도면 치수를 사용한다. 어댑터 외함과 픽업 받침은 설계 형상이며, 질량·마찰은 가정이다. Brev L40S의 단일 환경에서 기본 그리퍼의 마찰 접촉으로 두 부품을 15 mm 들어 올리고, 약 12.3 mm 삽입한 상태를 0.4초 유지해 독립 성공 판정을 통과했다. 물리 계산을 120 Hz에서 240 Hz로 바꾼 실행도 같은 성공 조건을 통과했다. 두 실행은 동일 seed의 명목 조건 검증이며, 정책 학습 성능이나 물성의 실물 일치를 뜻하지 않는다. 범위·치수 출처·실행 방법은 [RJ45 과제 문서](docs/rj45-task.md)에 있다.

기존 toy 과제는 같은 Isaac Sim·PhysX 환경에서 검증했다. 단일 seed의 zero-residual 기준 제어기는 고정 조인트나 물체 자세 덮어쓰기 없이 양쪽 그리퍼의 마찰 접촉으로 두 부품을 들어 올리고, 30.66 mm 깊이와 1.08 mm 횡오차로 삽입한 상태를 0.4초 유지해 44.07초에 독립 성공 판정을 통과했다. 별도로 기록한 3개 seed도 독립 평가와 데이터 계약을 통과했으며 train 2개·eval 1개로 분리했다. 이 결과는 여유 끼워맞춤 toy 기준선이며 RJ45 삽입, 잼 회복, 잔차 정책 성능, 카메라 기반 제어 또는 실물 전이의 검증 결과가 아니다. 기존 MuJoCo 트레이 운반 프로젝트와 별도 저장소다.

## 현재 결과

완료된 검증과 진행 중인 작업을 구분했다. 수치는 모두 시뮬레이션 결과이며 센서 성능이나 실물 전이를 뜻하지 않는다.

| 항목 | 결과 | 상태와 근거 |
| --- | --- | --- |
| 단일 GPU 병렬 처리량 | 64환경 223.39, 128환경 460.75, 256환경 868.43 전이/초 | 완료. update당 전이 수를 8,192개로 맞춰 비교했다. [결과 문서](docs/evidence/rj45-refinement-20261028/scale-results/README.md) |
| 4개 설정의 96-update 개발 비교 | 기준 제어 7/32, Z1 7/32, F 11/32, N 10/32, R 9/32 | 완료. 사전 규칙으로 F를 후속 후보로 골랐다. [요약](docs/evidence/rj45-refinement-20261028/scale-results/refinement-summary.json) |
| F의 새 seed 3개, 200 update | 7/32, 10/32, 8/32 | 완료. 평균 8.33/32로 승격 기준 10/32에 못 미쳤다. 최종 조건은 열지 않고 봉인했다. [결과](docs/evidence/rj45-refinement-20261028/scale-results/f-followup-summary.json) · [독립 검증](docs/evidence/rj45-refinement-20261028/scale-results/independent-audit.json) |
| generation v2 파지 pilot | 시뮬레이터 물리 파지 4개, 최대 clique 4 | pilot 통과. 고정 후보 20개를 평가해 학습 8개·개발 검증 2개·최종 검증 2개를 선정하는 중이다. [생성 계약](docs/evidence/rj45-demonstrations-20261028/grasp-generation-v2-protocol.json) · [실제 pilot 결과](docs/evidence/rj45-demonstrations-20261028/generation-v2-pilot-results/README.md) |
| teacher·BC·BC 초기화 PPO 비교 | 결과 없음 | production 파지 검증 뒤 실행한다. 아직 완료되지 않았다. [실행 계획](docs/rj45-next-stage-plan.md) |

## 256환경 병렬 PPO 영상

한 Isaac Sim 프로세스에서 256개 환경의 경험을 모아 정책 하나를 갱신하는 장면을 기록했다. 개요와 근접 영상은 같은 설정을 사용한 별도 실행이며, 각각 16초·60 fps·960프레임이다. 전체 학습 호출은 20 update를 마쳤고 영상 자막에는 프레임 시점의 시뮬레이션 시간과 현재 rollout 전 완료 update 수를 표시한다.

- [256환경 전체 구도 영상](docs/media/rj45/parallel-ppo/rj45-parallel-ppo-overview.mp4)
- [환경 0 근접 영상](docs/media/rj45/parallel-ppo/rj45-parallel-ppo-closeup.mp4)
- [전체 구도 썸네일](docs/media/rj45/parallel-ppo/rj45-parallel-ppo-overview-thumbnail.png)
- [근접 썸네일](docs/media/rj45/parallel-ppo/rj45-parallel-ppo-closeup-thumbnail.png)
- [촬영 조건·원본·인코딩·프레임 검증](docs/evidence/rj45-refinement-20261028/parallel-media.json)

두 영상은 실제 벡터화 PPO rollout을 보여주지만, 학습을 마친 삽입 성공 정책의 시연은 아니다. 개요와 근접 화면도 한 실행의 동기화된 두 카메라 영상이 아니다.

## 에셋과 데이터

- 공식 RB-Y1 M v1.2와 기본 RB gripper를 source revision·파일 SHA-256으로 고정한다.
- 원본 그리퍼 USD의 빠진 충돌 설정을 원본 손가락·손바닥 mesh의 convex hull로 보완한다. RJ45 실행은 기본 그리퍼 형상과 마찰 접촉을 사용하며, 물체와 손 사이 고정 조인트는 없다.
- RJ45 픽업 지그는 위가 열린 로케이터다. 플러그와 측면·전면 사이에 각각 0.15 mm의 초기 틈을 두어 미끄러짐과 회전을 줄이되, 위로 들어 올리는 동작은 막지 않는다. 공급사 부품이 아닌 프로젝트 설계 형상이다.
- 기존 toy 조립 부품은 치수·질량·마찰을 명시한 자유 강체다. 하우징의 구멍을 네 개의 벽 collider로 보존한다.
- NVIDIA AutoMate `00015`의 peg/socket, grasp, 분해 궤적 100개를 참고 데이터로 확보했다. Franka 시연을 RB-Y1 action 데이터로 바꾸어 표시하지 않는다.
- NVIDIA Simple Room은 실행 영상의 시각 배경으로 사용했으며, 충돌과 성공 판정에는 관여하지 않는다. 병렬 물리 학습의 필수 의존성으로도 두지 않는다. 에셋 목록과 사용 상태는 [자산 문서](docs/assets-and-data.md)에 있다.
- 데이터 기록기로 수집한 에피소드에는 관측·행동·관절·물체 자세·접촉·독립 평가와 출처를 저장한다. 단위 테스트의 합성 데이터는 학습 시연이 아니다.
- 기존 toy 기준선 HDF5는 3개 seed가 모두 독립 성공·유효 판정을 받았고, 학습 가능 train 2개와 격리한 eval 1개를 포함한다. 파일·episode hash와 원본 source archive 검증 결과의 정본은 [기준선 데이터 근거](docs/evidence/baseline-dataset.json)다. 기록 당시의 소스·설정·에셋 해시와 원본 소스 묶음을 보관한다. 이 데이터는 RJ45 정책 학습 근거로 사용하지 않는다.

다운로드한 바이너리 에셋은 Git에 포함하지 않는다. 기본 로봇·배경 에셋은 `configs/assets.json`, RJ45 CAD·도면은 `configs/rj45-assets.json`의 URL·크기·SHA-256으로 재현한다. 공식 잭 STEP은 KiCad 10의 `pcbnew` GLB export를 거쳐 정규화된 mesh로 변환하며, 변환 절차는 [RJ45 과제 문서](docs/rj45-task.md#재현)에 적었다.

## 로컬 검사

Python 3.11 환경에서 실행한다. 이 단계는 GPU 없이 가능하다.

```bash
cd "$HOME/rby1-assembly-lab"
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[assets]'
.venv/bin/python scripts/fetch_assets.py
.venv/bin/python scripts/build_scene.py
.venv/bin/python scripts/audit_assets.py --generated --output artifacts/asset-audit.json
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

`assets/generated/scene.usda`는 정적 장면이다. USD 생성이나 단위 테스트 통과는 물리 과제 성공을 뜻하지 않는다.

## Brev 실행

실행 대상은 RTX 렌더링을 지원하는 GPU다. Isaac Sim 5.1.0 / Isaac Lab 2.3.0 / Python 3.11 / RSL-RL 3.0.1을 기준으로 구현했다. 최신 버전을 뜻하는 조합은 아니며, Sim 5.1은 현재 공식 문서에서 지원 종료로 표시된다. 이 프로젝트의 고정 버전으로 먼저 검증하고, 버전 변경은 별도 재검증한다.

설치·실행·종료 절차와 검증 순서는 [Brev 실행 문서](docs/brev-run.md)에 있다. 실행기는 단계 변화와 물리 상태를 JSONL trace로 남기고, 단일 환경 실행을 MP4로 기록한다. 아래 기존 toy 근접 촬영은 1280×960에서 물리 상태를 초당 120장 기록했고, 공유용 4배속 영상과 정상 속도 전체 구도 영상은 모두 60 fps다. 물리 120 Hz·제어 30 Hz는 촬영 전후에 동일하다. RJ45 명목 영상과 병렬 PPO 영상은 서로 다른 계약으로 촬영했으며, 각 근거 문서에 프레임과 원본 해시를 기록했다.

전용 Brev L40S에 고정 버전 환경을 설치하고 사용자가 승인한 NVIDIA Omniverse EULA 설정으로 GPU 연산·PhysX·렌더링을 실행했다. 컴퓨팅 예산과 중지 원칙은 [예산 설정](configs/compute-budget.json)에 기록한다.

## 기존 toy 과제 미디어

- [근접 조립 영상 · 4배속 · 60 fps](docs/media/nominal-baseline-4x-60fps.mp4)
- [전체 구도 영상 · 정상 속도 · 60 fps](docs/media/nominal-overview-60fps.mp4)
- [동작·접촉 trace 그래프](docs/media/nominal-evidence.png)
- [파지 해제 오류와 부품 낙하 · 0.5배속 · 60 fps](docs/media/failure-gripper-open-60fps.mp4)
- [양손 충돌과 금지 접촉 정지 · 0.5배속 · 60 fps](docs/media/failure-hand-collision-60fps.mp4)
- [실패 원인·측정값·재현 조건](docs/evidence/failure-cases.json)
- [미디어 출처와 SHA-256](docs/media/media-manifest.json)
- [촬영 주기·프레임 누락·물리 상태 대조](docs/evidence/video-cadence.json)

실패 영상은 검증을 위해 잘못된 명령을 넣은 음성 시험이다. 파지한 손을 여는 명령으로 부품 낙하를 유발하고, 양손에 같은 목표를 주어 충돌을 유발한다. 물체 자세·센서값·종료 사유는 덮어쓰지 않는다. 영상에는 원인과 실제 측정값, 안전 모니터의 정지 이유를 표시한다. 학습 정책의 실패율이나 실물 로봇의 안전 성능을 뜻하지 않는다.

명목 조건의 양팔 파지·리프트·정렬·삽입은 [60fps 영상](docs/media/rj45/nominal40-pick-insert-60fps.mp4)으로 확인할 수 있다. 시뮬레이터 상태를 사용하는 기준 제어 기록이며 학습 정책의 성능 영상은 아니다. [촬영·프레임·원본 해시](docs/media/rj45/nominal40-pick-insert-60fps.json)를 함께 남겼다.

## RJ45 파지 실패 측정

픽업 받침을 개선하기 전의 무학습 기준 제어에서, 오른손 접촉이 사라진 뒤 플러그가 129.84° 회전하고 초기 위치보다 36.925 mm 내려간 실패를 기록했다. 이 실행은 `grip_close_timeout`으로 종료됐다. 그래프는 같은 물리 조건을 재현한 trace의 높이·회전·손가락 접촉을 시간순으로 표시한다. 특정 학습 정책의 실패율이나 실물 결과를 뜻하지 않는다.

- [파지 상실·플러그 회전·낙하 영상 · 60fps, 7.13초](docs/media/rj45/failure19-grip-loss-60fps.mp4)
- [영상 프레임·촬영·원본 해시](docs/media/rj45/failure19-grip-loss-60fps.json)
- [실패 측정 그래프](docs/media/rj45/failure19-evidence.png)
- [벡터 그래프](docs/media/rj45/failure19-evidence.svg)
- [시계열·계산식·출처 해시](docs/media/rj45/failure19-evidence-data.json)

## 검증 범위

제어기 진행 상태를 복원하기 전의 staged 초기화 구현은 4환경 부분 초기화와 64환경·1,000 제어 스텝의 임의 보정 검사를 통과했다. 모든 환경이 움직인 뒤 한 번씩 재초기화됐고, 측정 접촉 분리 조건을 통과했으며, 종료 시 모든 환경의 파지 안정화와 정책 활성화 완료를 확인했다. 부분 초기화에서는 다른 환경의 실제 관절·물체 상태가 보존됐다([검증 근거](docs/evidence/rj45-validation.json)). 이 기록은 이전 소스의 초기화 검사이며, 현재 제어 동작이나 학습 정책의 성능 평가가 아니다. 수정 환경의 파지 상태 시작 기준 제어는 4환경·각 1회에서 모두 삽입에 성공했다.

RJ45 잔차 학습 환경은 검증된 파지 상태를 초기화 때 복원하고, 실제 접촉·리프트·절대/상대 속도·관절 추종 오차를 다시 검사한 뒤 정렬과 삽입을 시작하도록 구현했다. 기존 기본값인 `post_integral_step`은 정책 출력을 적분 제어 뒤의 Cartesian twist 잔차로 적용한다. 새 `absolute_tip_pose_target_insert_only`는 INSERT 단계에서 오른팔 6D 출력만 받아 소켓 입구 좌표계의 플러그 팁 목표 자세를 바꾼다. 팁 위치를 회전 중심으로 다시 계산하므로 손목 회전만 덧붙이지 않으며, 기존 twist 잔차는 중복 적용하지 않는다.

표준 설정은 79차원 관측을 유지한다. 진단 설정은 episode에 고정된 보정 오차 6D를 추가한 85차원 privileged 관측이며 실제 센서 성능을 뜻하지 않는다. 정책과 기준 제어는 같은 고정 파지 보정값과 로봇 링크 자세 기반의 물체 위치 추정을 사용한다. 정책 활성화 시점의 추정 관계는 시뮬레이터 정답 자세에 보정 오차를 더해 초기화하며, 이후 물체 정답 자세는 안전 감시·보상·독립 평가에도 쓰인다. 테이블에서 집어 드는 전체 과제 평가의 초기 단계는 정답 상태를 사용하는 스크립트 제어다. 카메라 기반 제어나 실물 전이 검증은 아니다. 평가는 `--evaluation-start table`로 테이블 파지부터 수행하거나 `--evaluation-start held`로 검증된 파지 상태의 정렬·삽입만 분리할 수 있다. 두 시작 방식은 비교 키에 구분하며, held 평가에서는 초기화 통과와 실제 유효 정책 입력 횟수도 확인한다.

고정 100 µm 보정 오차를 상쇄하는 분석용 명령으로 제어 권한을 비교했다. 기존 모드는 3초 유지 시험을 통과하지 못했고 실제 x 오차가 -99.14 µm였으며, 새 모드는 연속 2.7333초 동안 조건을 유지한 뒤 전체 엄격 평가도 통과했다. 새 모드의 유지 종료 x 오차는 +1.67 µm였다. 이 A/B는 같은 명목 실행 42와 파지 상태를 사용한 제어 명령 유지 검사이며 강화학습 성능이나 센서 성능의 근거가 아니다. [제어 방식 비교 그래프](docs/media/rj45/controller-authority42-comparison.png)와 [시계열·원본 해시](docs/media/rj45/controller-authority42-comparison-data.json)를 제공하며, 원시 보고서와 trace 해시는 [RJ45 검증 근거](docs/evidence/rj45-validation.json)에 기록했다.

관측 정보의 영향을 분리하는 85차원 진단 설정도 구현했다. 기존 79개 관측값 뒤에 시뮬레이터가 정한 보정 오차 6개를 추가한다. 실제 센서값이나 추정 결과가 아닌 정답 파라미터를 사용하는 원인 분리 실험이다. 1환경·1 iteration·128 transition GPU smoke와 64환경·20-update 학습, 32개 조건의 짝지은 평가를 마쳤다. 기준 제어는 12회 성공했지만 85D 정책은 0회 성공했고 모두 24초 제한으로 끝났다. 무효 episode와 금지 접촉 하중은 0이었고 정책의 최대 삽입 접촉 하중은 2.968129873 N이었다. 이번 단일 held·단일 seed·20-update 진단에서는 privileged 6D를 추가해도 결과가 나아지지 않았다. 이 결과는 학습 원리의 한계를 증명하지 않으며 실물 센서, 추정된 보정 오차, 생산용 정책의 검증도 아니다. 표준 79D도 scripted ALIGN이 만든 관절 자세에 보정 오차 단서가 남을 수 있으므로, 85D 결과만으로 79D의 관측 가능 여부를 단정하지 않는다.

손목 F/T 센서 대신 PhysX 접촉값을 사용한다. 합력 크기와 개별 법선력의 합, 손가락 접촉점·법선·간격을 구분해 기록한다. 62개 관절의 위치·속도·제어 목표는 회전 관절 rad와 손가락 직선 관절 m를 나누어 저장한다. 로봇은 고정 베이스이며 좌우 손 사이 9개 강체 접촉쌍을 감시한다. 나머지 내부 로봇 접촉은 필터링하므로 전신 자체 충돌 회피를 검증한 것으로 표시하지 않는다. 모든 물성·힘 한계는 시뮬레이션 설계 가정이다.

기존 toy 4환경 병렬 장면은 일반 USD clone과 GPU PhysX를 유지하면서 fast physics replication을 끄고, 환경 간·전역 지면 충돌 필터와 금지 접촉의 환경별 종료를 검사했다. 한 환경에 넣은 금지 접촉이 그 환경만 중단시키고 나머지 세 환경은 계속 진행했으며, 다른 환경의 plug와 palm을 겹친 음성 대조에서는 필터 기반 금지 접촉 하중과 미필터 palm net force가 모두 0 N이었다. 근거는 [병렬 장면 검사](docs/evidence/parallel-scene-probe.json)에 묶는다. RJ45 접촉 감시도 현재 `after-persistent-target-hook` 증거에서 4환경 양성 접촉·비접촉·reset·환경 간 음성 대조와 실제 body pose SAT 검사를 통과했다([RJ45 검증 근거](docs/evidence/rj45-validation.json)). 전체 물리 보고서의 최대 접촉 하중은 20.08854294 N이다. 5 step 주기와 최초 접촉·끝점만 남긴 trace의 최대값 19.72353172 N과 구분해 기록한다. 다환경 실행은 이 증거 specification과 보고서·trace·SAT·오프라인 toolchain의 해시, 실제 센서의 환경별 순서를 확인해야 시작된다. 이 시험은 접촉 감지 경로의 검증이며, 전체 RJ45 과제의 병렬 성공이나 학습 성능을 뜻하지 않는다.

## 이전 학습 실험

초기 79D·85D 단일 seed 실험과 논문 기반 예비 비교는 현재 결과를 만들기 전의 개발 기록이다. 수치와 원본 해시는 [강화학습 연구·실험 문서](docs/rj45-rl-research.md), [관측 방식 비교 자료](docs/media/rj45/target-offset42-observation-comparison-data.json), [과거 장기 결과](docs/evidence/rj45-paper-ablation/long-results/manifest.json)에 보존했다. 현재 판단에는 위의 `현재 결과` 표와 [규모 확대 결과](docs/evidence/rj45-refinement-20261028/scale-results/README.md)를 사용한다.

다중 파지 production, teacher 시연, BC와 PPO 비교, 미사용 조건 평가는 아직 남아 있다. 한 파지 상태의 개발 성공률을 일반화 성능으로 해석하지 않는다.

[실험 계약](docs/experiment-design.md)은 기준 제어 비교, 데이터 분리, 성공 판정과 남은 검증을 정의한다.
