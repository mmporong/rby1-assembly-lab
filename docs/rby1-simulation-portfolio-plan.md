# RB-Y1 Isaac Lab 양팔 조립·병렬 강화학습 포트폴리오 계획

상태: 2026-11-01 기준 **G0 완료, G1 전체 task-admission 실행 중**이다. 기존 커밋 `a279c1d`의 MuJoCo 복귀안을 대체한다. G0 공통 PPO/SAC 실행 경로는 실제 Isaac GPU smoke를 통과했고, G1의 3-episode Isaac 사전 검증도 계약 3/3으로 통과했다. 같은 동결 조건의 classical tuning 18회와 admission 72회는 실행 중이며 G1 완료, RL pilot 진입, 강화학습 효과와 게시 준비는 아직 확정되지 않았다.

## 1. 결정과 프로젝트 주제

**Isaac Sim 5.1 / Isaac Lab 2.3 release 계열 / PhysX를 유지한다.** 실제 G0 환경의 `isaaclab` 배포판 metadata는 `0.47.2`다. `2.3`은 프로젝트가 고정한 release 명칭이고 `0.47.2`는 설치 패키지 metadata이므로 같은 버전 표기로 합치지 않는다. 기존 MuJoCo Tray Lab은 운반·경로 개선·하중 분석의 근거로 두고, 이번 assembly는 USD 로봇·작업물 구성, GPU 병렬 접촉 환경, 관측·행동·보상·평가 계약, 학습 비교로 역량을 확장한다. 같은 작업을 MuJoCo로 다시 만드는 것은 이번 필수 범위가 아니다.

주제는 **“Isaac Lab 기반 RB-Y1 양팔 커넥터 조립: 병렬 학습 환경 구축과 PPO·SAC 비교”**다. 물리 과제는 기본 그리퍼로 테이블의 LAN 어댑터와 RJ45 플러그를 잡아 들고 상대 자세를 맞춰 삽입하는 것이다. full-task 영상과 안정 파지 상태에서 시작하는 held-grasp 학습 평가는 구분한다.

이 프로젝트가 답할 질문은 **“학습 보정이 초기 위치·각도·파지 오차에서 삽입에 성공하는 범위를 넓히면서 과도한 접촉력을 늘리지 않는가”**다. PPO와 SAC의 차이 또는 학습 보정의 효과는 같은 평가 계약으로 반복 실험하기 전까지 완료형으로 쓰지 않는다.

기존 구조는 왼팔과 단계 전이를 스크립트가 담당하고, 학습 정책이 오른팔의 6차원 보정 명령을 낸다. 이 범위에서는 “양팔 조립 시스템의 삽입 보정 학습”으로 표현하며 양팔 모두의 협동학습 성과로 확대하지 않는다. 양팔 행동을 함께 학습하는 실험은 필요성과 비용을 검토한 뒤 별도 확장한다.

## 2. 두 프로젝트에서 보여줄 서로 다른 역량

| 항목 | 기존 Tray Lab | 이번 Assembly Lab |
|---|---|---|
| 실행 기반 | MuJoCo 3.11.0, MJCF | Isaac Sim 5.1 / Lab 2.3 release 계열(배포판 metadata 0.47.2), USD·PhysX |
| 해결한/해결할 문제 | 이동 양팔의 트레이 운반, 충돌 경로 개선, 부하 분석 | 작은 부품의 물리 파지·접촉 삽입, 병렬 학습·평가 |
| 모델링 | 관절·구동·충돌·센서 모델과 제어·기록 | USD 조합·의존성·collider·관성·mount frame, 환경별 물리 격리 |
| 실험 확장 | 정한 조건·seed의 반복 비교 | 벡터화된 다수 환경의 조건 무작위화·정책 학습·독립 평가 |
| 시각·데이터 | 기존 RGB-D와 하중 동기 기록 | RTX 렌더·카메라/깊이 데이터 생성 기능의 활용 후보. 현재 정책은 카메라 정책이 아닌 79D simulation proxy |

이 표는 이번 두 구현의 선택을 비교한다. MuJoCo가 강화학습·GPU 병렬·카메라를 지원하지 않는다는 뜻이 아니다. MuJoCo에도 MJX/Warp GPU 실행 경로가 있다. Isaac 선택의 이유는 현재 구축한 USD·PhysX·벡터 환경·렌더/센서 도구를 함께 활용하는 것이며, PhysX의 실물 정확도가 더 높다는 주장은 하지 않는다.

## 3. 현재 근거와 새 기여

### 현재 재사용할 근거

- 런타임과 로봇 출처는 `configs/runtime.json`에 고정돼 있다. Python 3.11, Isaac Sim 5.1.0, Isaac Lab 2.3 release 계열과 `isaaclab` 배포판 metadata 0.47.2, Torch 2.7.0/cu128, RSL-RL 3.0.1, RB-Y1 M v1.2 USD다. 이전 MuJoCo 기획의 별도 SB3 2.9.2/Torch 2.8 후보를 여기에 자동 적용하거나 환경을 업그레이드하지 않는다.
- `docs/assets-and-data.md`와 `src/rby1_assembly/scene.py`에 USD 출처·hash·의존성 감사, 기본 gripper collider 보완과 mount 기록이 있다. 기존 보완 자체를 새 성과로 세지 않는다.
- `docs/rj45-task.md`와 독립 평가기에 파지·리프트·삽입 깊이·정렬·힘·속도·dwell 계약이 있다. TE 도면/CAD를 참고한 rigid 모델이며 인증 mating pair, 실제 삽입력, 탭 탄성·케이블 장력·전기 연결 성과는 아니다. 50µm 횡오차는 프로젝트 기준이다.
- `docs/evidence/rj45-refinement-20261028/scale-results/README.md`에 단일 GPU의 64/128/256환경 처리량 223.39/460.75/868.43 transitions/s가 있다. 렌더링 없이 update당 8,192전이를 맞춘 수집·갱신 측정이다. 환경별 rollout 길이도 바뀌었으므로 성공률 개선이나 순수한 병렬화 인과로 해석하지 않는다.
- 최근 `docs/evidence/rj45-multibias-20261031/README.md`의 single2/8, multi3/8, zero2/8은 1seed 진단이며 RL 개선 근거로 승격하지 않는다.

### 새로 만들 두 기여

1. **신뢰할 수 있는 병렬 조립 실험 환경:** contact/solver/timestep·초기 파지·환경 간 충돌 격리·관측 유효성·동작 적용·reset/종료를 검증하고, 실패를 물리·관측·정책·실행기 문제로 구분한다. 다수 환경이 실행된다는 사실과 각각 독립된 유효 실험이라는 근거를 함께 만든다.
2. **같은 물리 환경의 고전 제어·PPO·SAC 비교:** 관측과 행동 권한, 저수준 제어, 보상, 초기조건, 평가기를 맞추고 학습 효율·접촉 안정성·새 조건 대응을 비교한다. 정책 이름 교체나 한 성공 영상만으로 개선을 주장하지 않는다.

| gate | 현재 상태 | 완료 근거 또는 남은 조건 |
|---|---|---|
| G0 공통 실행 경로 | 완료 | 4환경 PPO/SAC GPU smoke, 유한 손실과 비영 파라미터 변화, 독립 verifier PASS |
| G1 task admission | 전체 실행 중 | 3-episode Isaac 사전 검증은 계약 3/3으로 통과. 같은 동결 조건의 classical tuning 18회 뒤 admission 72회 실행 중이며 전체 판정 전에는 G1 완료나 pilot 진입을 주장하지 않음 |
| G2 반복 학습 | 미완료 | G1 admission 뒤 같은 경험 예산의 PPO/SAC pilot과 반복 학습 필요 |
| G3 봉인 평가 | 미완료 | 독립 조건·seed·CI·안전 비열등 기준을 실행 전에 봉인해야 함 |
| 영상·게시물 | 미완료 | G2/G3 결과와 실패 근거를 연결한 미디어가 필요 |

## 4. 다음 실행의 순서와 코드 경계

### G0 — 기존 환경 재현과 비교 계약 확정

`src/rby1_assembly/isaac_env.py`, `rj45_training_env.py`, `rj45_training.py`, 기존 recording/evaluation 경로를 기준으로 현재 baseline을 재현한다. 기존 코드를 보존한 채 새 비교 설정·관측 adapter·runner·검증 테스트를 별도 경로로 둔다. `scripts/analyze_rj45_paired.py`는 수정하지 않는다.

PPO/SAC 공통 실행 스택은 이 단계에서 현재 Isaac Lab pin과 GPU tensor 인터페이스에 맞춰 결정한다. 현재 extras는 `.cache/upstream/IsaacLab/source/isaaclab_rl/setup.py`에 SB3/skrl/rl-games와 RSL-RL 경로를 제공한다. SAC의 warmup·replay·UTD, PPO의 rollout, 두 알고리즘의 checkpoint/normalizer/termination 처리를 실제 짧은 smoke로 검증한 뒤 정확한 버전·wheel hash·설정·소유 파일을 동결한다. 라이브러리별 Python↔GPU tensor 복사, auto-reset의 final observation, time-limit bootstrap, policy/critic 권한도 확인한다. 현재 RSL PPO와 새 라이브러리 SAC를 설정 차이 그대로 비교하지 않는다.

G0는 skrl 2.1.0 공통 실행 경로로 완료했다. PPO와 SAC 모두 기존 단위 스케일을 유지한 79D 비특권 actor/critic 관측과 오른팔 6D residual action을 사용하며, 왼팔과 단계 전이는 기존 scripted 경로를 유지한다. 통합 CPU suite 75개가 통과했고 독립 G0 verifier가 PASS로 판정했다.

동결된 `104721a` runtime과 train held-bank를 사용한 4환경 Isaac smoke에서 PPO는 64전이·2 algorithm update·8 optimizer step, SAC는 64전이·64 algorithm update·192 optimizer step을 완료했다. 두 경로 모두 손실이 유한했고 27,532개 actor 파라미터 중 25,484개가 바뀌었다. report SHA-256은 `cd81227390670668976dd66b26d1cd0ed576dddc2b28efd9679c94091e7e2791`이다. 두 방법의 update 수 차이는 G0 경로 검증 설정이며 성능 공정 비교가 아니다.

**G0 완료 판정:** baseline report/hash, 공통 observation/action/reward/termination/normalization schema, 실행 스택과 dependency lock, 두 알고리즘 smoke, 정확한 환경 전이 계수기, runner/checkpoint/provenance 명세를 확보했다. 상세 근거는 `docs/evidence/rj45-comparison-g0-20261101/README.md`에 보존한다. 이 완료는 tensor·optimizer·Isaac 통합 경로를 확인한 것이며, 알고리즘 성능이나 학습 효과를 뜻하지 않는다. 학습 경험 수는 벡터 환경 N개의 제어 한 step당 N전이로 세고 physics substep과 구분한다.

### G1 — 모델·접촉·병렬 실행의 유효성

RB-Y1 M v1.2, gripper와 작업물의 source·hash·질량·관성·collider·frame을 고정한다. 시각 mesh와 collision을 함께 확인하고 socket 공동이 막히는 형상을 거부한다. 유효한 timestep/solver 범위, 파지 유지, 금지 침투, 유한한 속도·힘, 조건별 실패를 확인한다. 두 접촉 근사를 비교한다면 손가락 geom만 바꾸고 나머지 변수를 고정한다. 고의로 망가뜨린 collider는 감사기 음성 대조에만 사용한다.

full-task의 물체 state write는 episode reset에서만 허용한다. 이후 object pose 덮어쓰기·mocap·kinematic 전환·접착 토글 없이 동역학과 제어로 진행한다. 고정 robot mount/base의 모델 정의와 실행 중 상태 조작은 구분한다. held-grasp snapshot도 reset에서만 복원하고 안정성 검사 뒤 실행한다.

actor, classical controller, 단계 전이는 같은 encoder/FK·자세/접촉 proxy 권한을 사용한다. simulation proxy에는 가정한 noise·latency·dropout을 명시하고 evaluator GT와 분리한다. 관측 invalid를 무노이즈 GT로 대체하지 않으며, 관측 실패로 중단한 episode는 평가 실패 분모에 남긴다. 기존 estimator의 GT 초기화/fallback 경로는 감사하되 기존 PPO 실패 원인으로 단정하지 않는다.

환경 수를 늘릴 때 교차 환경 접촉, reset 독립성, 종료 후 상태, 접촉 buffer 용량과 누락을 검사한다. 처리량·VRAM·GPU 사용량과 유효 episode 수를 함께 기록한다. 실제 시뮬레이션을 여러 환경으로 배치한 영상은 원 run과 연결하고 복제 편집으로 대신하지 않는다.

현재 `scripts/run_rj45_task_admission.py`, `make_fair_provider`, `make_rj45_task_admission_env_type`, classical/constant/playback seal과 G1 판정기는 구현돼 있다. 통합 CPU suite 79개, G1 core architect CLEAR, runner CPU prelaunch APPROVE를 확인했고, [3-episode Isaac 사전 검증](evidence/rj45-task-admission-smoke-20261101/README.md)은 계약 3/3으로 통과했다. source `2e1f926`과 로깅·source 계약을 보존한 채 `configs/rj45-comparison-development.json`의 동일 조건으로 classical tuning 18회 뒤 2 grasp × 6 bias × 6 slot admission 72회를 실행 중이다. 전체 결과가 나오기 전에는 task admission 완료, RL pilot 진입 또는 학습 효과를 주장하지 않는다.

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

계획 단계와 G0 구현·검증·GPU smoke는 끝났고, G1은 3-episode Isaac 사전 검증을 통과한 뒤 같은 동결 조건의 classical tuning 18회와 admission 72회를 실행 중이다. 프로젝트의 GPU 인스턴스는 한 대만 사용하며 전체 실험은 사용자가 정한 700 USD 한도 안에서 관리한다. `6 USD`는 완료된 G0 smoke에 잡았던 과거 예약 상한이고 실제 청구액이 아니다. 현재 G1 full90의 예약 상한은 `7.50 USD`이며 240분 자동 stop guard와 최대 13,800초 process timeout을 둔다. 실제 process 한도는 guard deadline 300초 전을 넘지 않도록 동적으로 줄이며, 현재 guard deadline은 `2026-10-31T20:48:44.655689Z`다. 비용 수치는 관측 단가와 실행 시간을 사용한 추정치이며 청구액으로 쓰지 않는다. G2 실행 예산은 승인되지 않았고, G1 결과만으로 다음 단계의 실행 권한이 생기지 않는다. 비용·프로세스·실행 결과는 기술 기록에 남기되 비공개 계정·청구 자료는 문서와 게시물에 넣지 않는다.

렌더 없는 학습과 촬영은 분리한다. home과 `/data`의 실제 peak 여유를 검사하고 원본을 삭제해 조건을 억지로 맞추지 않는다. 사용하지 않는 GPU·프로세스는 작업 후 종료한다. G0 target은 `2026-10-31T16:04:21.816443Z`에 `STOPPED`로 확인됐고 종료 guard의 timer와 service는 모두 `inactive/not-found`였다. 결과 회수도 완료됐다. 후속 실행 전에는 구체 자원 상태를 런타임에서 다시 확인한다.

문서 검증은 기존 근거 대조, 공식 지원 기능과 현재 구현 구분, 수정 diff, 별도 verifier 검수로 수행한다. G0의 실제 Isaac 결과와 provenance는 확인됐다. G1의 강한 classical·learnability gate와 G2/G3의 반복·봉인 평가를 통과하기 전에는 PPO/SAC 차이, 성공 범위 확대, 접촉력 안전성을 입증했다고 게시하지 않는다. MuJoCo 복귀안의 기술스택·새 MJCF 파일 계획은 활성 후속 작업에서 제외한다.

## 근거

- `configs/runtime.json`, `docs/assets-and-data.md`, `docs/rj45-task.md`
- `docs/evidence/rj45-refinement-20261028/scale-results/README.md`
- `docs/evidence/rj45-multibias-20261031/README.md`
- `docs/evidence/rj45-comparison-g0-20261101/README.md`
- `docs/rj45-comparison-task-admission.md`, `configs/rj45-comparison-development.json`
- [Isaac Lab 2.3 RL 실행 경로](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/reinforcement-learning/rl_existing_scripts.html)
- [Isaac Lab 2.3 카메라](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/core-concepts/sensors/camera.html)
- [MuJoCo 3.11 개요와 MJX/Warp 경로](https://mujoco.readthedocs.io/en/3.11.0/overview.html)
