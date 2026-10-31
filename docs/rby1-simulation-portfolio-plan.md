# RB-Y1 MuJoCo 접촉조립 모델 검증과 PPO·SAC 비교 계획

상태: 실행 전 계획 문서. 아래 내용은 구현·검증 계약이며 공개 완성품이나 실험 결과가 아니다.

## 1. 결정과 목표 결과

**추천: B안 — MuJoCo 중심 RB-Y1 접촉조립 환경·모델/에셋 검증을 주제로 삼고, 동결된 환경에서 classical/zero/PPO/SAC를 핵심 평가한다.** 목표 문장은 “RB-Y1 M 양팔과 RJ45 작업물을 MJCF로 구성해 접촉 민감도를 검증하고, 동일 권한의 residual PPO와 SAC가 기준 제어의 남은 오차를 줄이는지 판정한다”이다. 이는 JD의 MuJoCo 모델·환경·asset 개발을 전면에 두면서 사용자의 강화학습 비교 목적을 유지한다. 아직 만들어지지 않은 모델 성능이나 학습 우위는 완료 성과처럼 쓰지 않는다.

물리 과제는 **“왼손으로 작은 LAN 어댑터를 들고, 오른손으로 RJ45 플러그를 잡아 구멍에 맞춰 천천히 끼우는 작업”**이다. “고정되지 않은 왼손 지지 오차가 오른손 정렬·삽입에 영향을 준다”를 검증 가설로 둔다. rigid 삽입이며 탭 탄성, 케이블 장력, 전기 연결, 실제 삽입력 예측은 범위 밖이다. 50 µm 횡오차도 프로젝트 판정값이다 (`docs/rj45-task.md:1-32`).

### 대안 판단

| 대안 | 채용 신호 | 장점 | 결정 |
|---|---|---|---|
| A. PPO–SAC 알고리즘 benchmark | RL 신호는 강하나 MuJoCo 모델 개발 신호가 좁음 | 정면 비교가 명확함 | **주제 전면 선택 보류.** 유효한 대안이지만 JD 핵심을 덜 보여준다. 기존 한 seed 음성 결과는 기각 근거가 아니라 새 실험 설계의 경고다 (`docs/evidence/rj45-multibias-20261031/README.md:3-7`, `:49-80`). |
| B. MuJoCo 모델/접촉 검증 + 학습 효과 비교 | JD 핵심과 RL 목표를 함께 충족 | 모델링, 검증, 실패 분석, 학습을 한 기술 질문으로 연결 | **필수 범위로 채택.** |
| C. MuJoCo–PhysX full-task 정합 + sim2sim + 학습 | 폭은 가장 넓음 | 엔진 차이를 다루는 고급 신호 | **선택 확장.** 현재 동일 로봇·충돌·관성·작업물·구동·좌표계 하네스가 없어 원인 혼합이 크다. cross-engine 일치는 실물 정확성 증거도 아니다. B의 게이트를 모두 통과한 뒤 한 개의 고정된 접촉 미니장면에서만 검토한다. |

## 2. 새 기술 기여와 JD 매핑

새 성과는 두 개만 주장한다.

1. **RB-Y1 M MuJoCo RJ45 모델 패키지와 검증 하네스:** 공식/고정 출처, MJCF include 구조, 관절·구동기·질량·관성·충돌·좌표계 manifest를 만들고, 접촉에 실제로 사용하는 두 타당한 근사와 timestep/solver 민감도를 같은 판정기로 비교한다. 고의로 망가뜨린 collision은 검출기의 negative control일 뿐 성능 기준선이 아니다. 기존 Isaac 쪽 gripper convexHull 6개 보완 (`docs/assets-and-data.md:46`, `src/rby1_assembly/scene.py:63-103`)과 기존 MuJoCo tray 저장소의 CAD mesh 교정 (`/home/lim/rby1-tray-lab/docs/modeling.md:15-21`)은 출처·참고 구현으로 명시하고 새 성과로 세지 않는다.
2. **동결 환경의 강화학습 판정:** 같은 오른손 6D residual action, actor observation, control, reward 권한으로 classical/zero/PPO/SAC를 비교한다. 왼손과 단계 전이는 script이므로 양팔 협동학습으로 부르지 않는다. 양팔 모두 학습은 선택 확장이다 (`configs/rj45-training-v1.json:1-10`, `docs/evidence/rj45-multibias-20261031/README.md:13-24`).

| 구분 | 현재 확인된 구현 | 이번에 새로 만드는 증거 |
|---|---|---|
| Isaac/PhysX | RB-Y1 USD, gripper collider 6개, RJ45 task와 strict evaluator, 기존 PPO 자료가 있음 (`docs/assets-and-data.md:9-17`, `:46-56`) | 새 성과로 세지 않고 요구사항·회귀 참고로만 사용 |
| MuJoCo | 별도 tray 저장소에 RB-Y1 M과 gripper CAD 접촉 교정이 있으나 RJ45 양팔 환경은 없음 (`/home/lim/rby1-tray-lab/docs/modeling.md:1-21`) | 이 저장소의 RJ45 MJCF 모델, 출처 manifest, 두 타당 접촉 근사, 독립 평가와 민감도 증거 |
| 학습 | PhysX 오른손 6D residual 한 seed 비교는 효과 미입증 | MuJoCo에서 classical/zero/PPO/SAC의 동일 권한·다중 seed 짝지은 판정 |

| JD | 포트폴리오 증거 |
|---|---|
| RB-Y1 Simulation 환경 설계·개발 | 양팔 파지–정렬–삽입 task contract, reset 분포, 실패 조건, 재현 seed |
| MuJoCo 모델 개발·고도화 | MJCF 모델, actuator/contact/inertia 검증, 접촉 근사·timestep 민감도 |
| Robot Model / Asset 관리 | 출처 URL·commit·SHA-256·license·A형/M형 구분 manifest. Menagerie pin과 모델 구분은 `/home/lim/rby1-tray-lab/src/rby1_tray/assets.py:8-13`, `/home/lim/rby1-tray-lab/docs/modeling.md:1-10`을 참고하되 새 RJ45 패키지에 독립 기록 |
| Python/C++, kinematics/dynamics, Git 리뷰 | Python 검증기·FK/frame 검사·동역학 지표·테스트·작은 리뷰 가능한 커밋. C++/ROS2는 실제 사용하지 않으면 주장하지 않음 |
| VLA 지원 | state/action/timestamp, camera/episode/task metadata를 내보낼 수 있는 schema만 선택 범위로 설계. loader 호환 검증과 VLA 학습 성공은 구분 |
| system identification / Sim-to-Real | 이번에는 합성 파라미터 민감도와 미식별 항목 목록까지만. 실물 데이터가 없으므로 식별·sim-to-real 성과로 표현하지 않음 |

## 3. 범위와 코드 접점

### 필수 범위

- `assets/mujoco/rby1_rj45/` 또는 저장소의 기존 asset 관례에 맞는 새 디렉터리: RB-Y1 M, 양 그리퍼, table, adapter/socket, plug의 MJCF include와 mesh 참조. 큰 원본 바이너리는 저장소에 복제하지 않고 manifest로 고정한다. 현재 Isaac 런타임 pin과 robot source는 `configs/runtime.json:1-25`, 자산 관리 원칙은 `docs/assets-and-data.md:1-28`을 따른다.
- `configs/mujoco-rj45-model.json`: 모델 variant, timestep, solver, contact 근사, actuator, 질량·관성·마찰, 출처와 미식별 가정을 한곳에 둔다. 기존 `configs/tasks/rj45-v1.json:126-177`의 PhysX 값은 자동 이식하지 않고 비교 출처로만 사용한다.
- `src/rby1_assembly/mujoco_rj45_model.py`: MJCF 로드, A/M variant 검사, actuator/joint/frame/geom manifest, 정적 모델 감사.
- `src/rby1_assembly/mujoco_rj45_env.py`: task reset, 스크립트 제어, residual action 적용, 센서 관측을 담당한다. 성공 판정은 포함하지 않는다.
- `src/rby1_assembly/mujoco_rj45_gym.py`: Gymnasium `Box(79,)` observation과 `Box(6,)` residual action, 공통 reward, `terminated`/`truncated`, terminal observation, 정확한 environment-transition/physics-substep counter를 제공한다. 한 environment의 한 control step만 1 transition이며 vector N개면 N을 더한다.
- `src/rby1_assembly/mujoco_rj45_evaluator.py`: simulator ground truth만 읽는 독립 평가기. 깊이, 횡오차, 각도, 상대속도, dwell, 접촉 하중, 금지 접촉을 판정한다. 현재 독립 판정 구조는 `src/rby1_assembly/recording.py:408-470`, 임계 계약은 `docs/rj45-task.md:28-32`를 기준으로 재현한다.
- `scripts/audit_mujoco_rj45.py`, `scripts/run_mujoco_rj45_baseline.py`, `scripts/evaluate_mujoco_rj45_paired.py`: 각각 모델 감사, 명목/full-task 실행, 짝지은 평가만 담당한다. `scripts/analyze_rj45_paired.py`와 기존 증거 폴더는 수정하지 않는다.
- `configs/mujoco-rj45-rl-common.json`, `configs/mujoco-rj45-ppo.json`, `configs/mujoco-rj45-sac.json`, `scripts/train_mujoco_rj45_sb3.py`: 공통 계약과 알고리즘별 runner를 분리한다. checkpoint, frozen normalizer, SAC replay/warmup/UTD·resume state, source/config/G2 report/sealed manifest hash를 run manifest에 묶는다.
- `tests/test_mujoco_rj45_model.py`, `tests/test_mujoco_rj45_contract.py`, `tests/test_mujoco_rj45_evaluator.py`, `tests/test_mujoco_rj45_gym.py`, `tests/test_mujoco_rj45_sb3.py`: model/GT 경계 외에 두 알고리즘 headless smoke, exact transition counter, GT 혼입 거부, TimeLimit truncation과 terminal observation, final checkpoint, 네 방법 표 재생성을 검사한다.
- `pyproject.toml`, `requirements-mujoco-lock.txt`: `.venv-mujoco`를 Isaac 환경과 분리하고 Python 3.11, `mujoco==3.11.0`, `stable-baselines3==2.9.2`, `gymnasium==1.4.0`, NumPy 1.26.0, h5py 3.14.0을 고정한다. Torch 2.8.x는 resolver 뒤 정확 patch·wheel·CPU/CUDA build와 hash를 lock한다. 근거는 [SB3 metadata](https://pypi.org/project/stable-baselines3/2.9.2/), [Gymnasium metadata](https://pypi.org/project/gymnasium/1.4.0/), [PPO API](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), [SAC API](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)이며 실제 설치 호환성은 아직 미검증이다.

### 선택 범위

- LeRobot Dataset v3 형태 export와 loader round-trip. state/action/timestamp, multi-camera, episode/task metadata만 검증하며 VLA 학습은 하지 않는다.
- B 완료 뒤, plug–socket만 놓은 고정 접촉 미니장면의 MuJoCo–PhysX 지표 대조. 전체 양팔 궤적이나 sim2real 주장으로 확장하지 않는다.
- ROS2 bridge, C++ 구현, 실제 RB-Y1 SDK 제어, 실물 시험, 양팔 모두 학습은 완료 조건이 아니다.

## 4. 순차 게이트와 판정

### G0 — 실험 계약과 출처 동결

모델 variant를 **RB-Y1 M 1.2**로 고정하고 A형 혼입을 실패시킨다. 자산 URL·revision·SHA-256·license를 기록한다. controller, observation, 독립 GT 평가 필드를 동결한다. actor 입력은 관측 원천별로 `encoder/FK`, `contact proxy`, `pose proxy`를 구분하고 noise·latency·dropout 가정을 기록한다. simulator의 무노이즈 object pose·latent bias는 금지한다. noisy pose proxy는 허용하되 GT와 같은 이름을 쓰지 않는다. 신규 SB3 PPO/SAC의 actor와 critic은 모두 같은 79D를 사용하며 critic privilege를 두지 않는다. 기존 RSL PPO의 actor 79D/critic 92D checkpoint와 PhysX 결과는 `legacy_gt_diagnostic`으로 격리하고 신규 결과와 합치지 않는다 (`src/rby1_assembly/rj45_training.py:543-620`, `:1001-1048`; `src/rby1_assembly/recording.py:34`, `:618-623`). 기존 `rj45_training_env.py:1316-1355`의 실제 상대 자세 기반 estimator 초기화와 `:1405-1437`의 일부 valid row 실제 object pose fallback은 기존 실패 원인으로 단정하지 않고 새 경로 제거 감사 항목으로 등록한다.

`train`, `pilot-tuning`, `sealed-final` manifest를 분리한다. 각 manifest는 condition ID/값, grasp ID와 snapshot hash, reset seed, 생성 코드·config hash를 가진다. 세 pool은 ID와 원시 상태가 겹치지 않으며 pilot 결과는 최종 효과 추정에 넣지 않는다. final grasp는 정책 평가 전에 full-task 실행으로 생성하고, 접촉 안정·힘 한계·금지 접촉·snapshot 유한성이라는 정책 비의존 수용 규칙으로만 채택한다. 성공 정책에 유리한 grasp 사후 선별과 정책별 grasp 재추출을 금지하며 sealed manifest hash가 달라지면 final 평가 전체를 무효화한다.

**통과:** manifest 누락 0개, 모든 hash 재계산 일치, A/M 혼입 0개, action/observation/critic/GT schema 테스트 통과. 실패하면 물리 실행으로 넘어가지 않는다.

### G1 — 모델 구조와 물리 기본 검증

정적 감사에서 joint/actuator 수, limit, gear/control range, mass·inertia 양의 정부호, mesh scale, 좌표축, 양손 TCP와 socket-entry/plug-tip frame을 검사한다. 자유 낙하·정지·관절 step·FK 교차검사로 단위와 구동 방향을 확인한다. 움직임 고정, kinematic object, weld/equality 접착, qpos 덮어쓰기는 감사 전용 fixture 외에는 금지한다.

contact variant는 **그리퍼 손가락 geom만** 바꾸는 단일 변수로 제한한다. reference는 손가락 부품별 단일 convex hull, sensitivity는 같은 손가락의 고정된 convex decomposition으로 사전 지정한다. 변경 part/mesh hash/생성 파라미터와 geom pair·margin·gap을 manifest에 기록한다. socket cavity, plug, palm, 나머지 robot geom과 visual, 질량, 관성, 마찰, actuator, controller는 고정한다. socket 공동 전체에 단일 convex hull을 씌우는 구성은 geometry gate에서 거부한다. 가능한 횡·각도·깊이를 먼저 검사한 뒤 접촉·solver/timestep을 평가한다. 둘 중 어느 쪽도 더 물리적으로 정확하다고 미리 단정하지 않고 full-task 성공만으로 정확성을 주장하지 않는다. 일부 collision 삭제 모델은 negative control일 뿐 성능 표에 넣지 않는다.

**통과:** NaN/Inf 0, 초기 자가충돌·금지 침투 0, frame 위치/방향 오차가 사전등록 허용치 안, 명령 부호와 joint limit 위반 0. 허용치는 실행 전에 config에 고정하며 결과를 본 뒤 넓히지 않는다.

### G2 — 강한 비학습 기준선과 민감도

full-task nominal은 기본 자세에서 양손 파지, 실제 lift, 정렬, 삽입까지 실행한다. 작업물과 동적 robot 실행 state write는 첫 `mj_step` 전 episode reset에서만 허용하고 held-grasp snapshot도 reset 안에서 한 번만 복원한다. 이후에는 actuator 경로만 쓰며 qpos/qvel/xpos 덮어쓰기, mocap, kinematic 전환, equality/weld toggle을 금지한다. 선언된 robot 고정 mount/base와 정적 fixture는 모델 정의로 허용하되 실행 중 바꾸지 않는다. proxy가 invalid이면 GT fallback 없이 episode를 중단하고 `sensor_or_observation_failure`로 동결 평가 분모에 실패 포함한다. 이를 로그/hash 불일치·실행기 오류처럼 전체 실험을 무효화하는 provenance failure와 구분하며 정책별 관측 실패를 사후 제외하지 않는다. phase transition도 actor와 같은 proxy 권한만 쓴다. **held-grasp 평가**는 full-task에서 안정성이 확인된 snapshot에서 시작하고 RL 구간은 정렬–삽입으로 제한한다. 두 결과를 합산하거나 held-grasp를 full-task 성공으로 부르지 않는다.

두 접촉 근사 × 최소 2개 timestep × 사전등록 solver 설정에서 같은 seed와 초기조건을 반복한다. 기준선은 속도 제한 삽입 + force/jam 중단 + 후퇴·재정렬을 갖춘 스크립트다. 지표는 strict 성공률, 최초 실패 사유, 완료 시간, 최종/최대 횡오차·각도·깊이, dwell, 순접촉력과 법선력 절댓값 합, 금지 접촉, 침투 깊이, 에너지/속도 폭주다. 각 조건은 같은 **3개 repeat seed**로 재실행하고 deterministic 설정에서 지표 편차와 trace hash를 기록한다.

**통과:** geometry gate를 통과한 근사 중 하나 이상이 nominal full-task와 repeat 판정을 통과해야 한다. 둘 다 통과하면 민감도가 낮고 계산비용이 합리적인 근사를 사전 기준으로 선택한다. 하나만 통과하면 통과 근사를 채택하고 다른 근사의 실패 범위를 보고한다. 둘 다 실패하면 RL로 진행하지 않는다. timestep/solver에 따라 판정이 바뀌면 안정 구간을 찾기 전 학습을 막는다.

### G3 — 학습 입장 검사와 짝지은 평가

학습 전, classical과 zero residual을 미사용 초기 오차·파지에서 평가한다. 비교군은 **classical/zero/SB3 PPO/SB3 SAC**이다. zero는 base + 0 residual, classical은 같은 79D proxy·6D bound·update rate를 쓰는 bounded PD/force-limited search + 후퇴·재정렬 결정 규칙이며 학습법과 같은 pilot condition 접근·trial/GPU-hour 상한에서 gain/threshold를 조정한다. G2 report hash와 sealed manifest hash가 다르면 runner가 시작되지 않는다. 네 방법은 같은 adapter, reward, termination, counter를 쓰며 train pool의 base rollout으로 만든 normalizer를 hash 고정해 공유하고 pilot/final에서 update하지 않는다. PPO/SAC actor는 모두 `[256,256]` ReLU MLP로 고정해 기본 activation 차이를 제거하고, critic 구조 차이는 알고리즘 특성으로 manifest에 기록한다.

주 공정성 축은 **동일 environment transition 수**다. PPO rollout overshoot가 없도록 총 budget을 PPO rollout batch와 두 알고리즘 batch 분할의 공배수로 protocol에 고정하고 runner 종료 시 exact counter 일치를 요구한다. 같은 trial 수, condition 접근, tuning GPU-hour 상한을 주고 gradient update, SAC replay/warmup/UTD·resume, wall-clock, memory를 기록한다. 동일 GPU-time 결과는 보조 비교다. final checkpoint는 사전 규칙으로 하나만 선택하며 checkpoint·normalizer·runner config·dependency lock·G2/sealed hash 계보가 완전해야 평가한다.

3 training seed와 최소 8개 미사용 조건은 **pilot**이다. pilot의 분산으로 최종 최소 5 training seed, condition별 독립 reset, 총 episode 수와 비용 상한을 별도 protocol에 수치로 동결한다. 효과 경계 `δ_success`와 unsafe-rate 비열등 margin `m_safety`는 pilot 결과와 무관한 실무 최소효과·허용위험으로 protocol에 수치 고정한다. 주 대조는 PPO−classical과 SAC−classical이다. 두 성공률 차이의 one-sided CI 하한이 각각 `δ_success`를 넘고, unsafe-rate 차이의 one-sided CI 상한이 `m_safety` 미만이어야 우월성을 주장한다. 두 주 대조는 Holm 방식으로 family-wise α=0.05를 통제한다. seed·condition 계층을 보존하고, deterministic classical/zero 결과를 training seed마다 복제해 표본 수를 늘리지 않는다. precision admission을 못 채우면 **불확정**, CI가 경계를 못 넘으면 **우월성 미입증**, 모두 통과하면 **우월성 지지**로 결론을 닫는다. 평균 reward나 한 영상은 근거가 아니다.

### G4 — 공개 산출물과 단일 대표 결과

대표 결과는 **“MuJoCo에서 검증한 RB-Y1 양팔 RJ45 접촉조립 모델과, 동일 환경에서 residual RL이 강한 기준선의 남은 오차를 줄였는지에 대한 재현 가능한 판정”** 한 문장으로 통일한다. LinkedIn 한 편에는 ① 모델/출처 도식, ② 두 물리 근사·민감도 표, ③ 짝지은 RL 결과 표, ④ 하나의 split-screen 대표 영상, ⑤ commit/config/artifact hash와 재현 명령을 넣는다. 주 1회 게시 습관은 결과를 나누는 편집 주기일 뿐, 한 주 안에 모든 게 성공한다는 일정 약속으로 쓰지 않는다.

필수 실패 영상은 사전등록 평가에서 자연스럽게 나온 두 종류다. **F1:** full-task에서 파지/정렬/접촉 중 최초 물리 실패가 보이고 실패 사유·힘·침투/오차가 함께 표시된 영상. **F2:** held-grasp의 동일 시작조건에서 기준선 또는 정책이 jam, force limit, 횡오차, dwell 미달 중 하나로 실패하는 영상. 성공한 조건만 재촬영하지 않고 원 평가 report/trace hash에 연결한다. 기존 gripper 감사 영상, 기존 PhysX multi64/zero 영상 (`docs/evidence/rj45-multibias-20261031/README.md:88-103`)은 배경 자료일 뿐 새 결과 영상으로 사용하지 않는다.

**완료:** 모든 수치가 원 report에서 재생성되고 영상 두 개가 manifest와 일치한다. **모델 핵심 완료**는 G0–G2 통과로, **RL 목표 완료**는 G3 sealed-final 실행과 위 결론 규칙 적용으로 따로 판정한다. RL이 불확정 또는 우월성 미입증이면 모델 분석은 가치가 있어도 사용자 RL 성과 목표는 **미충족**이다. 그때는 “모델 검증 + 불확정/음성 학습 결과”로만 게시한다. G2 실패 시 G3를 실행하지 않는다.

게시 문구는 실제 통과한 게이트와 최종 통계 판정만 반영한다.

## 5. 검증, 자원 경계, 중단 규칙

검증 순서는 dependency lock preflight → CPU schema/unit test → MuJoCo headless smoke → 모델 감사 → nominal repeat → 민감도 → PPO·SAC 각 headless smoke → 승인된 GPU 학습 → sealed 평가 → 네 방법 표 재생성이다. lock 설치·import·짧은 step이 통과하기 전 호환성을 주장하지 않는다. 각 게이트는 config, code/model/asset hash, runtime, seed, counter를 manifest에 남긴다. MuJoCo는 `3.11.0` 기능만 사용한다 (`/home/lim/rby1-tray-lab/pyproject.toml:5-10`, `/home/lim/rby1-tray-lab/requirements-lock.txt:1`). 최신 `3.13` API를 전제하지 않는다.

현재 단계에서는 구현, 다운로드, Brev, GPU, 게시를 하지 않는다. 향후 실행 전 home과 `/data`의 실제 여유와 peak 용량을 계산한다. MuJoCo physics는 CPU headless로 실행하고 렌더링과 분리한다. CPU pilot 처리량을 잰 뒤 학습 update device와 Brev 필요성을 결정하며 PPO MLP에 고가 GPU를 기본 요구하지 않는다. GPU 물리 병렬화로 표현하지 않는다. 어느 볼륨도 예상 peak의 2배와 추가 2 GB가 없으면 실행하지 않고 기존 원본을 삭제하지 않는다. Brev는 `configs/compute-budget.json:29-38`의 순서·유휴 금지를 지킨다.

중단 조건은 provenance/hash·variant·GT 권한·pairing 위반, G2 전 근사 실패, 자원 초과, 최종 admission 미충족이다. 대안은 plug–socket 미니장면으로 축소하거나 RL 없이 모델 감사·민감도·실패 분석을 내는 것이다. PhysX와 MuJoCo 수치를 합치거나 임의 개선률을 만들지 않는다.

## 6. 실행 인계와 최종 수용 기준

1. G0 계약 PR을 먼저 검토하고, 그 뒤에만 모델 코드를 작성한다.
2. G1/G2 구현과 테스트를 작은 커밋으로 분리해 model source, physics, evaluator 변경의 원인을 추적 가능하게 한다.
3. 학습 코드는 G2 통과 report hash를 요구하도록 fail-closed로 연결한다.
4. fresh Architect가 모델 경계·접촉 근사·GT 분리를 승인하고, fresh Critic이 기준선·통계·실패 공개·stop rule을 승인한 뒤 실행 계획으로 넘긴다.

최종 수용 기준: B안 유지, 새 기여 2개 이하, 모든 게이트의 report/hash 재현, reset 뒤 actuator-only, proxy invalid의 fail-closed, 단일 contact 변수, train/pilot/sealed manifest 분리, four-method 동일 계약, equal-transition 주 비교와 compute 보조 비교, 수치 고정된 δ·안전 margin·다중비교, pilot 뒤 최종 5 seed 이상 admission, 실패 영상 2개, G0–G2/G3 완료 분리, 기존 성과 재포장 금지다.
