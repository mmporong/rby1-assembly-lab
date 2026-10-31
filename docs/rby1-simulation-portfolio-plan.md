# RB-Y1 Isaac Lab 양팔 조립·병렬 강화학습 포트폴리오 계획

상태: 2026-10-31 사용자 정정을 반영한 방향 계획. 기존 커밋 `a279c1d`의 MuJoCo 복귀안을 대체한다. 구현·학습·게시 완료를 뜻하지 않으며, 학습 실행은 아래 G0 계약 검증 뒤에 진행한다.

## 1. 결정과 프로젝트 주제

**Isaac Sim 5.1 / Isaac Lab 2.3 / PhysX를 유지한다.** 기존 MuJoCo Tray Lab은 운반·경로 개선·하중 분석의 근거로 두고, 이번 assembly는 USD 로봇·작업물 구성, GPU 병렬 접촉 환경, 관측·행동·보상·평가 계약, 학습 비교로 역량을 확장한다. 같은 작업을 MuJoCo로 다시 만드는 것은 이번 필수 범위가 아니다.

주제는 **“Isaac Lab 기반 RB-Y1 양팔 커넥터 조립: 병렬 학습 환경 구축과 PPO·SAC 비교”**다. 물리 과제는 기본 그리퍼로 테이블의 LAN 어댑터와 RJ45 플러그를 잡아 들고 상대 자세를 맞춰 삽입하는 것이다. full-task 영상과 안정 파지 상태에서 시작하는 held-grasp 학습 평가는 구분한다.

기존 구조는 왼팔과 단계 전이를 스크립트가 담당하고, 학습 정책이 오른팔의 6차원 보정 명령을 낸다. 이 범위에서는 “양팔 조립 시스템의 삽입 보정 학습”으로 표현하며 양팔 모두의 협동학습 성과로 확대하지 않는다. 양팔 행동을 함께 학습하는 실험은 필요성과 비용을 검토한 뒤 별도 확장한다.

## 2. 두 프로젝트에서 보여줄 서로 다른 역량

| 항목 | 기존 Tray Lab | 이번 Assembly Lab |
|---|---|---|
| 실행 기반 | MuJoCo 3.11.0, MJCF | Isaac Sim 5.1 / Lab 2.3, USD·PhysX |
| 해결한/해결할 문제 | 이동 양팔의 트레이 운반, 충돌 경로 개선, 부하 분석 | 작은 부품의 물리 파지·접촉 삽입, 병렬 학습·평가 |
| 모델링 | 관절·구동·충돌·센서 모델과 제어·기록 | USD 조합·의존성·collider·관성·mount frame, 환경별 물리 격리 |
| 실험 확장 | 정한 조건·seed의 반복 비교 | 벡터화된 다수 환경의 조건 무작위화·정책 학습·독립 평가 |
| 시각·데이터 | 기존 RGB-D와 하중 동기 기록 | RTX 렌더·카메라/깊이 데이터 생성 기능의 활용 후보. 현재 정책은 카메라 정책이 아닌 79D simulation proxy |

이 표는 이번 두 구현의 선택을 비교한다. MuJoCo가 강화학습·GPU 병렬·카메라를 지원하지 않는다는 뜻이 아니다. MuJoCo에도 MJX/Warp GPU 실행 경로가 있다. Isaac 선택의 이유는 현재 구축한 USD·PhysX·벡터 환경·렌더/센서 도구를 함께 활용하는 것이며, PhysX의 실물 정확도가 더 높다는 주장은 하지 않는다.

## 3. 현재 근거와 새 기여

### 현재 재사용할 근거

- 런타임과 로봇 출처는 `configs/runtime.json`에 고정돼 있다. Python 3.11, Isaac Sim 5.1.0, Lab 2.3.0, Torch 2.7.0/cu128, RSL-RL 3.0.1, RB-Y1 M v1.2 USD다. 이전 MuJoCo 기획의 별도 SB3 2.9.2/Torch 2.8 후보를 여기에 자동 적용하거나 환경을 업그레이드하지 않는다.
- `docs/assets-and-data.md`와 `src/rby1_assembly/scene.py`에 USD 출처·hash·의존성 감사, 기본 gripper collider 보완과 mount 기록이 있다. 기존 보완 자체를 새 성과로 세지 않는다.
- `docs/rj45-task.md`와 독립 평가기에 파지·리프트·삽입 깊이·정렬·힘·속도·dwell 계약이 있다. TE 도면/CAD를 참고한 rigid 모델이며 인증 mating pair, 실제 삽입력, 탭 탄성·케이블 장력·전기 연결 성과는 아니다. 50µm 횡오차는 프로젝트 기준이다.
- `docs/evidence/rj45-refinement-20261028/scale-results/README.md`에 단일 GPU의 64/128/256환경 처리량 223.39/460.75/868.43 transitions/s가 있다. 렌더링 없이 update당 8,192전이를 맞춘 수집·갱신 측정이다. 환경별 rollout 길이도 바뀌었으므로 성공률 개선이나 순수한 병렬화 인과로 해석하지 않는다.
- 최근 `docs/evidence/rj45-multibias-20261031/README.md`의 single2/8, multi3/8, zero2/8은 1seed 진단이며 RL 개선 근거로 승격하지 않는다.

### 새로 만들 두 기여

1. **신뢰할 수 있는 병렬 조립 실험 환경:** contact/solver/timestep·초기 파지·환경 간 충돌 격리·관측 유효성·동작 적용·reset/종료를 검증하고, 실패를 물리·관측·정책·실행기 문제로 구분한다. 다수 환경이 실행된다는 사실과 각각 독립된 유효 실험이라는 근거를 함께 만든다.
2. **같은 물리 환경의 고전 제어·PPO·SAC 비교:** 관측과 행동 권한, 저수준 제어, 보상, 초기조건, 평가기를 맞추고 학습 효율·접촉 안정성·새 조건 대응을 비교한다. 정책 이름 교체나 한 성공 영상만으로 개선을 주장하지 않는다.

## 4. 다음 실행의 순서와 코드 경계

### G0 — 기존 환경 재현과 비교 계약 확정

`src/rby1_assembly/isaac_env.py`, `rj45_training_env.py`, `rj45_training.py`, 기존 recording/evaluation 경로를 기준으로 현재 baseline을 재현한다. 기존 코드를 보존한 채 새 비교 설정·관측 adapter·runner·검증 테스트를 별도 경로로 둔다. `scripts/analyze_rj45_paired.py`는 수정하지 않는다.

PPO/SAC 공통 실행 스택은 이 단계에서 현재 Isaac Lab pin과 GPU tensor 인터페이스에 맞춰 결정한다. 현재 extras는 `.cache/upstream/IsaacLab/source/isaaclab_rl/setup.py`에 SB3/skrl/rl-games와 RSL-RL 경로를 제공한다. SAC의 warmup·replay·UTD, PPO의 rollout, 두 알고리즘의 checkpoint/normalizer/termination 처리를 실제 짧은 smoke로 검증한 뒤 정확한 버전·wheel hash·설정·소유 파일을 동결한다. 라이브러리별 Python↔GPU tensor 복사, auto-reset의 final observation, time-limit bootstrap, policy/critic 권한도 확인한다. 현재 RSL PPO와 새 라이브러리 SAC를 설정 차이 그대로 비교하지 않는다.

**G0 완료 조건:** baseline report/hash, 공통 observation/action/reward/termination/normalization schema, 실행 스택과 dependency lock, 두 알고리즘 smoke, 정확한 환경 전이 계수기, runner/checkpoint/provenance 명세가 모두 있어야 한다. 이 gate 전에는 본학습을 시작하지 않는다. 학습 경험 수는 벡터 환경 N개의 제어 한 step당 N전이로 세고 physics substep과 구분한다.

### G1 — 모델·접촉·병렬 실행의 유효성

RB-Y1 M v1.2, gripper와 작업물의 source·hash·질량·관성·collider·frame을 고정한다. 시각 mesh와 collision을 함께 확인하고 socket 공동이 막히는 형상을 거부한다. 유효한 timestep/solver 범위, 파지 유지, 금지 침투, 유한한 속도·힘, 조건별 실패를 확인한다. 두 접촉 근사를 비교한다면 손가락 geom만 바꾸고 나머지 변수를 고정한다. 고의로 망가뜨린 collider는 감사기 음성 대조에만 사용한다.

full-task의 물체 state write는 episode reset에서만 허용한다. 이후 object pose 덮어쓰기·mocap·kinematic 전환·접착 토글 없이 동역학과 제어로 진행한다. 고정 robot mount/base의 모델 정의와 실행 중 상태 조작은 구분한다. held-grasp snapshot도 reset에서만 복원하고 안정성 검사 뒤 실행한다.

actor, classical controller, 단계 전이는 같은 encoder/FK·자세/접촉 proxy 권한을 사용한다. simulation proxy에는 가정한 noise·latency·dropout을 명시하고 evaluator GT와 분리한다. 관측 invalid를 무노이즈 GT로 대체하지 않으며, 관측 실패로 중단한 episode는 평가 실패 분모에 남긴다. 기존 estimator의 GT 초기화/fallback 경로는 감사하되 기존 PPO 실패 원인으로 단정하지 않는다.

환경 수를 늘릴 때 교차 환경 접촉, reset 독립성, 종료 후 상태, 접촉 buffer 용량과 누락을 검사한다. 처리량·VRAM·GPU 사용량과 유효 episode 수를 함께 기록한다. 실제 시뮬레이션을 여러 환경으로 배치한 영상은 원 run과 연결하고 복제 편집으로 대신하지 않는다.

### G2 — 공정한 학습과 봉인 평가

강한 고전 힘 제어, base+zero residual, PPO, SAC를 같은 조건으로 비교한다. classical은 같은 추정값과 행동 한도 안의 결정적 PD/force-limited search·후퇴/재정렬이다. PPO/SAC의 policy·critic 관측 권한과 전처리는 맞추고, 알고리즘 고유 구조 차이는 기록한다. 튜닝 조건 접근과 trial/계산 예산도 동등하게 제한한다.

train / pilot-tuning / sealed-final의 condition·grasp·reset seed·snapshot hash를 분리한다. 파지는 정책 비의존 수용 규칙으로 생성하고 정책별 사후 선별을 금지한다. pilot은 3 training seed부터 시작하고 최종은 최소5 seed와 실측 분산에 따른 독립 평가 수를 실행 전에 동결한다. pilot 데이터는 최종 효과 추정에 재사용하지 않는다.

주 비교는 같은 환경 경험 수다. 업데이트·UTD/replay/warmup·wall-clock·VRAM·GPU-hour를 함께 기록하고 같은 계산시간 비교는 보조 결과로 둔다. 최종 checkpoint 선택, 실용 성공률 차이 경계·안전 비열등 margin·CI·다중비교 방법은 final 실행 전에 수치 고정한다. classical/zero를 training seed 수만큼 복제해 표본을 늘리지 않는다. 성공률·끼임·힘 제한 초과·완료 시간·미사용 조건 결과를 원 report에서 재생성한다. 불확정/우월성 미입증/우월성 지지를 구분한다.

## 5. 링크드인 한 편의 산출물과 JD 연결

다음 대표 결과는 **“Isaac Lab에서 RB-Y1 양팔 접촉조립의 독립 병렬 환경을 구축하고, 같은 조건에서 학습 방법의 효율·안정성을 비교했다”**라는 내용을 검증한 뒤 작성한다. 아직 수행하지 않은 부분을 완료형으로 게시하지 않는다.

- 기본 자세·테이블 파지·리프트·정렬·삽입이 이어지는 전체 장면.
- 실제 병렬 환경이 서로 다른 초기조건을 수행하는 장면과 유효성·처리량 기록.
- 같은 초기조건의 classical/PPO/SAC 비교 화면과 성공률·접촉력·학습비용 표.
- 원인이 드러나는 실패2개와 힘·정렬·상태 전이의 동기 기록. 수치 폭주는 물리적 고장과 구분한다.
- source/asset/config/runtime/checkpoint hash, seed와 재현 명령.

JD의 MuJoCo 모델 개발은 기존 Tray Lab의 검증 근거로 보여준다. 이번은 RB-Y1 simulation 설계, USD/asset 관리, PhysX 접촉과 GPU 병렬 학습, Isaac Sim 우대 경험을 보강한다. 기존 ROS2 경력은 해당 별도 프로젝트에서 설명한다. RGB-D·state/action/time/episode export와 실제 loader 검증은 VLA 지원의 선택 확장으로 두며 VLA 학습 성과로 바꾸어 쓰지 않는다.

MuJoCo–PhysX 전체 task 정합, 실물 식별·Sim-to-Real, 유연 케이블, 양팔 모두 학습은 이번 필수 범위가 아니다. 모델·병렬 실행 완료와 RL 개선 입증은 별도로 판정하며 하나가 다른 목표의 완료를 대신하지 않는다.

## 6. 자원·검증·완료 경계

이번 정정은 계획 문서만 변경한다. 학습·설치·Brev 시작·게시는 수행하지 않는다. 실행 단계에서 프로젝트용 Brev 한 대와 실제 비용·디스크 상한을 확인하고, 렌더 없는 학습과 촬영을 분리한다. home과 /data의 실제 peak 여유를 검사하고 원본을 삭제해 조건을 억지로 맞추지 않는다. 사용하지 않는 GPU·프로세스는 작업 후 종료한다.

문서 검증은 기존 근거 대조, 공식 지원 기능과 현재 구현 구분, 수정 diff, 별도 verifier 검수로 수행한다. G0에서 실행 스택/adapter/테스트와 수치 protocol이 확정되기 전에는 이 문서를 구현 착수용 완성 명세로 취급하지 않는다. MuJoCo 복귀안의 기술스택·새 MJCF 파일 계획은 활성 후속 작업에서 제외한다.

## 근거

- `configs/runtime.json`, `docs/assets-and-data.md`, `docs/rj45-task.md`
- `docs/evidence/rj45-refinement-20261028/scale-results/README.md`
- `docs/evidence/rj45-multibias-20261031/README.md`
- [Isaac Lab 2.3 RL 실행 경로](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/reinforcement-learning/rl_existing_scripts.html)
- [Isaac Lab 2.3 카메라](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/core-concepts/sensors/camera.html)
- [MuJoCo 3.11 개요와 MJX/Warp 경로](https://mujoco.readthedocs.io/en/3.11.0/overview.html)
