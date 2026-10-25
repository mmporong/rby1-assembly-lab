# RB-Y1 Assembly Lab

한 손으로 하우징을 지지하고 다른 손으로 플러그를 삽입하는 양팔 조립 실험이다. 접촉으로 막힌 상황에서 후퇴·재정렬·재삽입하는 기준 제어와 제한된 잔차 강화학습을 비교한다.

**현재는 GPU 물리 검증 전 구현 단계다.** 공식 로봇·그리퍼와 공개 조립 에셋을 확보했고 USD 장면 생성, 물리 구성 감사, 데이터 기록, 제어·평가 로직을 작성했다. 실제 파지·조립 성공, 학습 성능, 실물 전이는 아직 입증하지 않았다. 기존 MuJoCo 트레이 운반 프로젝트와 별도 저장소다.

## 에셋과 데이터

- 공식 RB-Y1 M v1.2와 기본 RB gripper를 source revision·파일 SHA-256으로 고정한다.
- 원본 그리퍼 USD의 빠진 충돌 설정을 원본 손가락·손바닥 mesh의 convex hull로 보완한다. 실제 접촉 안정성은 GPU 검증 대상이다.
- 초기 조립 부품은 치수·질량·마찰을 명시한 자유 강체다. 하우징의 구멍을 네 개의 벽 collider로 보존한다. 물체와 손 사이 고정 조인트는 없다.
- NVIDIA AutoMate `00015`의 peg/socket, grasp, 분해 궤적 100개를 참고 데이터로 확보했다. Franka 시연을 RB-Y1 action 데이터로 바꾸어 표시하지 않는다.
- NVIDIA Simple Room은 시각 배경 후보이며, 병렬 물리 학습의 필수 의존성으로 두지 않는다. 에셋 목록과 사용 상태는 [자산 문서](docs/assets-and-data.md)에 있다.
- RB-Y1에서 새로 실행하는 에피소드는 관측·행동·관절·물체 자세·접촉·독립 평가와 출처를 기록한다. 단위 테스트의 합성 데이터는 학습 시연이 아니다.

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

설치·실행·종료 절차와 검증 순서는 [Brev 실행 문서](docs/brev-run.md)에 있다. 대규모 학습 전에 하나의 환경에서 낙하·개폐·파지·정렬·삽입을 확인해야 한다.

전용 Brev L40S에 고정 버전 환경을 설치하고 GPU 연산·패키지 검사를 마쳤다. 새 환경의 NVIDIA 이용약관 동의가 남아 있어 PhysX 과제 검증과 학습은 대기 중이다. 컴퓨팅 예산과 중지 원칙은 [예산 설정](configs/compute-budget.json)에 기록한다.

## 검증 범위

잔차 정책에는 정답 물체 자세에 잡음을 더한 상태 관측을 제공한다. 현재 기준 제어기와 단계 전환은 시뮬레이터 정답 상태를 사용한다. 따라서 시스템 전체의 센서 오차 강건성이나 카메라 기반 제어를 입증하는 단계는 아니다. 실제 손목 F/T 대신 PhysX 접촉력 대용값을 기록하며, 금지 접촉 판정도 접촉력 프록시에 기반한다. 로봇 자체 충돌은 비활성화되어 있다. 모든 물성·힘 한계는 시뮬레이션 설계 가정이며 제조사 실측 교정값이 아니다.

[실험 계약](docs/experiment-design.md)은 기준 제어 비교, 데이터 분리, 성공 판정과 남은 검증을 정의한다.
