# RB-Y1 Assembly Lab

한 손으로 하우징을 지지하고 다른 손으로 플러그를 삽입하는 양팔 조립 실험이다. 접촉으로 막힌 상황에서 후퇴·재정렬·재삽입하는 기준 제어와 제한된 잔차 강화학습을 비교한다.

전용 Brev L40S에서 Isaac Sim·PhysX 장면과 RTX 렌더링을 실행했다. 단일 seed의 zero-residual 기준 제어기는 고정 조인트나 물체 자세 덮어쓰기 없이 양쪽 그리퍼의 마찰 접촉으로 두 부품을 들어 올리고, 30.66 mm 깊이와 1.08 mm 횡오차로 삽입한 상태를 0.4초 유지해 44.07초에 독립 성공 판정을 통과했다. 별도로 기록한 3개 seed도 독립 평가와 데이터 계약을 통과했으며 train 2개·eval 1개로 분리했다. 이 결과는 여유 끼워맞춤 nominal 기준선이며 잼 회복, 잔차 정책 성능, 카메라 기반 제어 또는 실물 전이의 검증 결과가 아니다. 기존 MuJoCo 트레이 운반 프로젝트와 별도 저장소다.

## 에셋과 데이터

- 공식 RB-Y1 M v1.2와 기본 RB gripper를 source revision·파일 SHA-256으로 고정한다.
- 원본 그리퍼 USD의 빠진 충돌 설정을 원본 손가락·손바닥 mesh의 convex hull로 보완한다. 단일 환경 PhysX 실행에서 양손 마찰 파지·리프트·삽입 완료를 확인했다.
- 초기 조립 부품은 치수·질량·마찰을 명시한 자유 강체다. 하우징의 구멍을 네 개의 벽 collider로 보존한다. 물체와 손 사이 고정 조인트는 없다.
- NVIDIA AutoMate `00015`의 peg/socket, grasp, 분해 궤적 100개를 참고 데이터로 확보했다. Franka 시연을 RB-Y1 action 데이터로 바꾸어 표시하지 않는다.
- NVIDIA Simple Room은 성공 실행 영상의 시각 배경으로 사용했으며, 충돌과 성공 판정에는 관여하지 않는다. 병렬 물리 학습의 필수 의존성으로도 두지 않는다. 에셋 목록과 사용 상태는 [자산 문서](docs/assets-and-data.md)에 있다.
- RB-Y1에서 새로 실행하는 에피소드는 관측·행동·관절·물체 자세·접촉·독립 평가와 출처를 기록한다. 단위 테스트의 합성 데이터는 학습 시연이 아니다.
- 기준선 HDF5는 3개 seed가 모두 독립 성공·유효 판정을 받았고, 학습 가능 train 2개와 격리한 eval 1개를 포함한다. 파일·episode hash와 원본 source archive 검증 결과의 정본은 [기준선 데이터 근거](docs/evidence/baseline-dataset.json)다. 현재 코어와 병렬 충돌 설정으로 재기록했으며 저장한 소스·설정·에셋 해시가 모두 현재 파일과 일치한다.

다운로드한 바이너리 에셋은 Git에 포함하지 않는다. `configs/assets.json`의 URL·해시로 재현한다.

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

설치·실행·종료 절차와 검증 순서는 [Brev 실행 문서](docs/brev-run.md)에 있다. 실행기는 단계 변화와 물리 상태를 JSONL trace로 남기고, 단일 환경 실행을 MP4로 기록한다. 근접 촬영 원본은 1280×960, 30 fps, 약 44초이며 공유용 4배속 편집본은 약 13초다. 대규모 학습 전에 기준선 HDF5 기록 계약과 병렬 환경 격리를 별도로 확인한다.

전용 Brev L40S에 고정 버전 환경을 설치하고 사용자가 승인한 NVIDIA Omniverse EULA 설정으로 GPU 연산·PhysX·렌더링을 실행했다. 로컬 일반 단위 테스트 54개와 Torch 기반 제어·접촉 계산 테스트 19개가 통과했다. 컴퓨팅 예산과 중지 원칙은 [예산 설정](configs/compute-budget.json)에 기록한다.

## 미디어

- [근접 조립 영상](docs/media/nominal-baseline-4x.mp4)
- [전체 구도 영상](docs/media/nominal-overview.mp4)
- [동작·접촉 trace 그래프](docs/media/nominal-evidence.png)
- [초기 접근 실패 1](docs/media/failure-early-01.mp4) · [초기 접근 실패 2](docs/media/failure-early-02.mp4) — 이전 제어기 디버깅 기록
- [미디어 출처와 SHA-256](docs/media/media-manifest.json)

## 검증 범위

잔차 정책에는 정답 물체 자세에 잡음을 더한 상태 관측을 제공한다. 기준 제어기와 단계 전환은 시뮬레이터 정답 상태를 사용한다. 따라서 시스템 전체의 센서 오차 강건성이나 카메라 기반 제어를 입증하는 단계는 아니다. 손목 6축 F/T 센서 대신 PhysX 접촉을 사용하며, 삽입부의 합력 크기와 개별 접촉 법선력 절댓값의 합을 구분해 기록한다. 확인된 nominal 실행은 삽입 하중이 0 N인 여유 끼워맞춤이므로 접촉 잼이나 복구 능력의 근거가 아니다. 금지 접촉은 29개 로봇 부위와 대상 물체의 접촉 쌍에서 법선 하중을 합산한다. 로봇은 고정 베이스이고 자체 충돌은 비활성화되어 있다. 모든 물성·힘 한계는 시뮬레이션 설계 가정이며 제조사 실측 교정값이 아니다.

4환경 병렬 장면은 일반 USD clone과 GPU PhysX를 유지하면서 fast physics replication을 끄고, 환경 간·전역 지면 충돌 필터와 금지 접촉의 환경별 종료를 검사했다. 한 환경에 넣은 금지 접촉이 그 환경만 중단시키고 나머지 세 환경은 계속 진행했으며, 다른 환경의 plug와 palm을 겹친 음성 대조에서는 필터 기반 금지 접촉 하중과 미필터 palm net force가 모두 0 N이었다. 근거는 [병렬 장면 검사](docs/evidence/parallel-scene-probe.json)에 묶는다.

PPO는 4환경·1 iteration·96 transition에서 프로세스 정상 종료, 유한한 체크포인트 가중치와 optimizer 상태만 확인했다. 이는 학습 파이프라인 스모크이며 정책 성능이나 개선을 뜻하지 않는다. 남은 검증은 잼 난이도 고정, 여러 seed의 규모 있는 학습, 학습에 쓰지 않은 조건의 기준 제어 대비 평가다.

[실험 계약](docs/experiment-design.md)은 기준 제어 비교, 데이터 분리, 성공 판정과 남은 검증을 정의한다.
