# 양팔 케이블 조립 프로젝트 의미 검증: 문헌 근거 메모

조사 기준일은 2026-10-31이다. 범위는 공개 원문·저자/기관 코드로 확인 가능한 **양팔 또는 케이블 결합 조작**, **케이블 커넥터 삽입**, **힘 인지 잔차 학습**이다. 문헌의 성공률은 각 저자의 과제·초기화·센서·평가 조건에서만 성립하며, RB-Y1의 양손 male/female 결합에 성능 향상이나 현실적 유용성을 보장하지 않는다.

## 1. 이 과제가 실제로 두 팔의 동시 적응을 요구하는가?

### Takeaway

양 끝을 모두 공중에서 잡고 male/female 커넥터의 상대 자세와 삽입력을 맞추는 설정은, 두 케이블의 장력·비틀림·테이블 접촉이 반대편 끝의 자세를 바꾸면 두 팔의 동시 적응이라는 연구 질문을 만들 수 있다. 다만 가장 가까운 wire-harness 연구는 다섯 개의 평면 mover와 케이블 결합을 다루며, 두 로봇 팔이 서로 마주 잡은 커넥터를 삽입하지는 않는다. 따라서 현재 과제는 문헌에 의해 이미 검증된 표준 과제가 아니라, **양쪽 끝의 능동 제어가 필요한 조건을 실험으로 분리해야 하는 새 변형**이다.

### Cited Findings

| 핵심 자료·상태 | 실제 과제 / 학습 대상 | 양팔 능동 권한·센서 | 비교 기준과 효과 근거 | RB-Y1 과제와의 적용 차이 |
|---|---|---|---|---|
| Streloke et al., *Multi-agent Path Planning and Reinforcement Learning for wire harness handling*, Frontiers in Mechanical Engineering 12 (2026), DOI `10.3389/fmech.2026.1910151`; 2026-10-16 accept, 페이지에 최종 조판 전이라고 표시됨 | MuJoCo/Gymnasium에서 부분적으로 형성된 harness에 강하게 결합된 connector-housing fixture를 목표 배열로 이동시킨다. 단일 정책이 5개 planar mover의 10차원 속도 행동을 낸다. [원문](https://www.frontiersin.org/journals/mechanical-engineering/articles/10.3389/fmech.2026.1910151/abstract) | 5개의 **mover**가 모두 행동하지만 6D 로봇 팔·그리퍼·삽입 접촉·힘/토크 센서는 없다. 관측은 mover 간 거리와 목표 거리/각도다. [고정 SHA 코드](https://raw.githubusercontent.com/ludwigstr/WireHarness-MultiAgent-RL/0644f17b26669c0d3a749691cda81847ac0d1301/README.md) | 기하학적 prioritized planner는 최고 78%, end-to-end SAC는 평균 98%(95–99%, 120개 순열 중 약 114개), residual RL은 단일 전이 안정성에도 전체 순열 일반화에 실패했다고 보고한다. [원문](https://www.frontiersin.org/journals/mechanical-engineering/articles/10.3389/fmech.2026.1910151/abstract) | 케이블 때문에 여러 조작기가 결합된다는 점은 가장 가깝다. 그러나 이 결과로 두 팔의 접촉 삽입·6D residual PPO가 유효하다고 결론낼 수는 없다. 오히려 "residual이면 더 좋다"는 가설의 반례다. |
| Wu et al., *Robotic Assembly of Deformable Linear Objects via Curriculum Reinforcement Learning*, IEEE RA-L 10(5), 2025, pp. 4770–4777, DOI `10.1109/LRA.2025.3553676` | UR5 한 대가 2-finger gripper로 Type-C 케이블의 **케이블 부분**을 잡고 socket에 삽입한다. 초기 굽힘 난도를 curriculum으로 올리고, RLPD가 3D Cartesian 변위를 학습한다. [저자 공개 원고](https://openreview.net/attachment?id=KlQndUeGD8&name=pdf) | 양팔 제어는 없다. eye-in-hand RealSense D405, 고정 DFK 카메라, Robotiq FT-300의 3D 힘, 케이블 굽힘각을 관측에 넣고 PD 제어기가 행동을 실행한다. [저자 공개 원고](https://openreview.net/attachment?id=KlQndUeGD8&name=pdf) | 최고 난도는 약 40° 굽힘이며, 저자 방법은 98% (50회 시험), ACT는 0%, 비교 RL은 52%로 제시된다. socket 방향 알려짐, 로봇·그리퍼가 socket에 수직, 케이블은 x축으로만 굽는다는 가정이 있다. [저자 공개 원고](https://openreview.net/attachment?id=KlQndUeGD8&name=pdf) | 케이블 굽힘이 정렬·삽입력을 망치고 jam을 낸다는 물리적 동기는 강하다. 반면 fixed socket과 한 팔이라는 설정이므로, 양쪽을 잡는 편이 필요한지의 증거는 아니다. |
| NVIDIA, *The Machines that Make the Machines* (기술 보고, 2026-10 공개) | GB300 tray의 여러 케이블을 잡고 커넥터를 board socket에 삽입한다. 전용 손가락 형상, pose estimation, sim RL, 실제 환경 residual RL을 조합한다. [NVIDIA 원문](https://developer.nvidia.com/blog/the-machines-that-make-the-machines/) | 한 팔이 커넥터를 두 번째 팔이 잡을 자세로 옮기는 단계가 있으며, 삽입 residual은 카메라 대신 force-torque 입력을 쓴다. [NVIDIA 원문](https://developer.nvidia.com/blog/the-machines-that-make-the-machines/) | 완성 시스템은 4개 커넥터에서 90–95%를 보고하지만, 남은 실패의 주원인은 connector grasp 단계의 multi-arm coordination이라고 구분한다. [NVIDIA 원문](https://developer.nvidia.com/blog/the-machines-that-make-the-machines/) | 다중 팔·케이블·connector라는 산업적 유사성이 있다. 그러나 양팔이 양쪽 커넥터를 동시에 잡아 상대 운동을 제어한 삽입 실험은 아니다. |

- wire-harness 원문은 cable-coupled mover들의 결합을 명시하므로, **한쪽 행동이 다른 쪽의 상태/충돌 여유를 바꾼다**는 종류의 현상을 보여 준다. 다만 두 팔을 독립 PPO 두 개로 학습하면 그 결합을 관측·보상·행동 공간에서 제대로 표현하는지는 별도 문제다. [원문](https://www.frontiersin.org/journals/mechanical-engineering/articles/10.3389/fmech.2026.1910151/abstract)
- Wu et al.은 케이블 휨이 삽입력 전달을 약화하고 jam/damage 위험을 만든다고 관찰했다. 이는 한쪽 끝을 고정하지 않은 male/female 조합에서 양 팔의 자세·장력 조절을 측정할 물리적 근거가 되지만, 해당 논문은 환경 접촉으로 케이블 모양을 조절하는 단일팔 해법을 검증했다. [저자 공개 원고](https://openreview.net/attachment?id=KlQndUeGD8&name=pdf)
- 공개 코드 확인: `ludwigstr/WireHarness-MultiAgent-RL@0644f17b26669c0d3a749691cda81847ac0d1301:README.md:L1-L5`는 5 planar mover 단일 정책 환경이라고 설명하고, `:L56-L68`은 31차원 관측/10차원 행동을 명시한다. 저장소 API상 마지막 갱신은 2026-10-20이다. README `:L119-L121`에는 MIT라고 쓰였지만, 같은 SHA에서 `LICENSE` 파일을 찾지 못했고 GitHub API의 license 필드도 비어 있었다. 따라서 **코드 MIT 라이선스는 확인 불가**로 기록한다. [고정 SHA README](https://raw.githubusercontent.com/ludwigstr/WireHarness-MultiAgent-RL/0644f17b26669c0d3a749691cda81847ac0d1301/README.md)

### Inferences

- 프로젝트의 가장 설득력 있는 질문은 "두 팔이 있다"가 아니라, **케이블 장력·torsion·마찰이 커넥터 상대 자세/접촉력의 교란으로 실제 전달될 때, 양쪽을 동시에 조절하면 한쪽 고정 또는 한쪽만 잔차 제어보다 성공·힘·시간·재시도에서 우세한가**이다. 이 인과 경로가 없다면 양팔은 상태·행동 차원만 늘린다.
- v1의 한 학습 seed, 한 grasp, 총 11 완료 episode는 위 인과를 판정하기에 부족하다. 이는 결과가 나쁘다는 판정이 아니라, 현재 수치로는 seed·초기 자세·케이블 물성 변동과 방법 효과를 구분할 수 없다는 뜻이다.
- 최소한 다음 4개 조건을 같은 초기화 분포에서 비교해야 한다: (A) 양팔 고정 궤적/순응 제어, (B) 한쪽만 residual·반대쪽 고정, (C) 양팔 residual이나 독립 정책, (D) 하나의 중앙집중 joint policy. (B)와 (C)의 차이가 양팔 능동 권한의 효과이며, (A)와의 차이가 학습의 효과다.

### Gaps

- 이번 조사에서 **양쪽 케이블이 달린 male/female 커넥터를 양팔이 각자 잡아 동시 삽입**하고, force/torque·마찰·torsion randomization과 비학습 양팔 순응 제어를 함께 비교한 2023–2026 공개 원문은 찾지 못했다. 따라서 이 과제의 새로움은 가능하지만, 기존 SOTA 재현 또는 일반화 성능 주장은 근거가 없다.
- 현재 asset의 케이블 모델이 장력·torsion·테이블 접촉을 동역학적으로 재현하는지와 policy 관측이 이를 식별할 수 있는지는 문헌만으로 판정할 수 없다. 이 항목은 asset/시뮬레이터 검증으로 넘어가야 한다.

## 2. Classical compliant/force control 대비 RL 이득을 검증할 수 있는가? 케이블 두 개는 복잡도만 늘리는가?

### Takeaway

검증 가능한 RL 이득 질문은 있다. 단, 비교 대상은 "zero residual"만이 아니라, 동일한 상대 pose·force 제한·접촉 탐색을 갖춘 양팔 compliant/force baseline이어야 한다. 현 문헌은 학습이 넓은 초기 자세나 분포 이동에서 유용할 수 있음을 보이지만, 단일팔 cable insertion의 고전 force-control 대조나 양팔 공정의 공정 시간 대조를 제공하지 않는다. 따라서 RL의 이득은 아직 가설이다.

### Cited Findings

| 핵심 자료·상태 | 실제 과제 / 학습 대상 | 센서·저수준 제어 | 비학습 baseline 및 효과 근거 | 이 프로젝트에 주는 제한 |
|---|---|---|---|---|
| Huang et al., *Force-Aware Residual DAgger via Trajectory Editing for Precision Insertion with Impedance Control*, arXiv:2603.04038, v3 2026-07-23; IROS 2026 accepted(공식 proceedings 출판 여부는 별도) | USB·2/3-pin plug precision insertion에서 base policy의 rollout과 사람 교정을 trajectory editing으로 섞어 residual policy를 학습한다. [arXiv 원문](https://arxiv.org/abs/2603.04038) | base policy는 이미지·EE pose·force로 미래 pose/force를 예측하고, 50 Hz 오류 검출기는 예측 힘과 측정 힘의 차이를 쓴다. 실행은 Cartesian impedance control이다. [arXiv 원문](https://arxiv.org/abs/2603.04038) | ACT, FILIC, HG-DAgger, retrain, finetune과 비교한다. 저자는 전체 평균 77.2%가 최고 baseline finetune 40.0%보다 37.2 percentage points 높다고 보고한다. 다만 고전 impedance-only/force-search baseline은 없다. [arXiv 원문](https://arxiv.org/abs/2603.04038) | 힘을 관측에 넣는 것만으로 평균 27.6→32.4%였고, residual data 구성까지 포함한 방법이 핵심이었다. 그러므로 "6D residual PPO"라는 형태만으로 결과를 귀속하면 안 된다. |
| Wu et al. (RA-L 2025) | Type-C 케이블을 한 팔로 socket에 삽입하며, cable bend 난도를 curriculum으로 바꾼다. [저자 공개 원고](https://openreview.net/attachment?id=KlQndUeGD8&name=pdf) | 두 카메라, 3D EE force, connector/cable angle; RL 행동은 PD controller가 실행한다. [저자 공개 원고](https://openreview.net/attachment?id=KlQndUeGD8&name=pdf) | 비교는 ACT 및 다른 RL이며, 논문 표에는 conventional compliant force/position controller가 없다. [저자 공개 원고](https://openreview.net/attachment?id=KlQndUeGD8&name=pdf) | 케이블 변형을 학습의 난도 축으로 쓸 수 있다는 근거는 제공하지만, RL이 force control을 이긴다는 근거는 제공하지 않는다. |
| Yamazaki et al., *Wiring connector-terminated cables based on manipulation planning with collision-free EMD net*, Robotics and Autonomous Systems 175, 104673 (2024), DOI `10.1016/j.robot.2024.104673` | 한 산업용 팔이 cable 끝 connector를 잡아 board socket으로 운반·삽입한다. CF-EMD net은 케이블 shape 변화와 connector trajectory를 학습해 cable-obstacle collision을 피한다. [출판 원고](https://repositori.mypolycc.edu.my/jspui/bitstream/123456789/6803/1/Wiring%20connector-terminated%20cables%20based%20on%20manipulation%20planning%20with.pdf) | range image sensor로 실제 cable shape를 재고, simulation에서 학습한 model의 출력을 실측 shape로 보정한다. [논문 설명](https://www.sciencedirect.com/science/article/am/pii/S0921889024000563) | real cable과 여러 길이에서 안정적 manipulation을 보고하지만, RL 대 classical force control의 성공률 표나 양팔 ablation은 제시하지 않는다. [출판 원고](https://repositori.mypolycc.edu.my/jspui/bitstream/123456789/6803/1/Wiring%20connector-terminated%20cables%20based%20on%20manipulation%20planning%20with.pdf) | 케이블이 하나여도 cable-obstacle collision·shape estimation이 핵심 문제가 될 수 있다는 반례다. 두 번째 케이블은 이 문제가 실제로 상대 pose를 교란할 때만 정보량을 더한다. |
| NVIDIA 기술 보고 (2026-10, 보조 근거; peer-reviewed 논문 아님) | multi-connector cable assembly에서 pose estimator, 전용 finger, sim RL base, 실제 force-torque residual을 결합한다. [NVIDIA 원문](https://developer.nvidia.com/blog/the-machines-that-make-the-machines/) | 삽입 중 visual occlusion 때문에 image 대신 force-torque residual을 쓴다. [NVIDIA 원문](https://developer.nvidia.com/blog/the-machines-that-make-the-machines/) | large connector의 sim-to-real 성능은 높았지만 small connector에는 덜 효과적이고, 산업 기준 신뢰도에는 미치지 못했다고 기술한다. [NVIDIA 원문](https://developer.nvidia.com/blog/the-machines-that-make-the-machines/) | RL은 sensor/geometry mismatch를 없애지 않는다. 기계적 guide와 pose repeatability를 먼저 만들 수 있는지 또한 경쟁 가설로 남는다. |

- Frontiers의 wire-harness 연구는 residual RL이 최종 순열 일반화에서 실패했고 plain SAC가 더 좋았다고 보고한다. 따라서 residual PPO를 기준법으로 전제하거나, 두 케이블의 복잡성을 residual이 자동 흡수한다고 가정하면 안 된다. [원문](https://www.frontiersin.org/journals/mechanical-engineering/articles/10.3389/fmech.2026.1910151/abstract)
- NVIDIA의 기록은 cable shape/stiffness가 샘플별로 달라 behavior cloning 데이터 규모 확보가 막혔고, 전용 finger geometry로 grasp pose repeatability를 높였다고 설명한다. 이는 케이블을 더 넣기 전에 **교란을 물리적으로 줄이는 설계**도 경쟁 기준에 포함해야 한다는 보조 근거다. [NVIDIA 원문](https://developer.nvidia.com/blog/the-machines-that-make-the-machines/)
- 공개 코드/라이선스: 위 wire-harness 환경만 저자 저장소를 확인했다. Wu et al., Huang et al., Yamazaki et al.에 대해서는 이번 제한 탐색에서 대응하는 공개 저자 코드를 찾지 못했다. NVIDIA 글은 제품/연구 기술 보고이며, 이 특정 조립 파이프라인의 공개 코드 라이선스는 확인하지 못했다.

### Inferences

- **RL의 검증 가능한 이득 가설:** 동일한 force limit·접촉 검출·상대 pose 목표·action rate를 맞춘 cooperative impedance/admittance controller보다, random cable stiffness/길이, initial relative-pose bias, table friction, cable twist가 커질수록 (i) 성공률, (ii) 최대 접촉력 초과율, (iii) 삽입 시간, (iv) 재접촉/재시도 횟수에서 유리하다. 이 네 지표 중 하나만 좋아도 되지만, 사전에 primary metric과 safety threshold를 정해야 한다.
- **케이블 두 개가 복잡도만 늘리는 경우:** 두 케이블을 고정 길이·균일 stiffness·고정 grasp·마찰 없음으로 두고, 한 커넥터의 상대 pose를 다른 팔이 기하학적으로 보정할 수 있다면 케이블은 관측하기 어려운 자유도와 탐색 공간만 보탠다. 이 경우 rigid male/female 또는 한쪽 fixture 고정 task가 더 짧고 강한 baseline이다.
- **의미 있는 경우:** 같은 상대 pose bias에서 (1) 케이블 force/twist/table-contact randomization을 켰을 때만 fixed-arm 또는 one-active-arm이 무너지고, (2) 두 팔의 동시 정책이 force limit을 지키며 회복한다면, 두 케이블은 실제 협동 적응의 원인 변수다. 이 조건이 확인되기 전에는 "케이블 복잡성에 강하다"라고 쓰지 않는다.

### Gaps

- 2023–2026의 가까운 공개 자료에서 "양팔 compliant/force controller 대 양팔 RL"을 같은 cable connector insertion 조건으로 정량 대조한 결과를 찾지 못했다. 그러므로 portfolio 결과는 기존 문헌과의 성능 경쟁이 아니라, 명시한 자체 ablation의 공정 비교로 제시해야 한다.
- classical baseline의 gain/stiffness, force threshold, search pattern은 connector의 clearance·재질·케이블 강성·sim 접촉 모델에 따라 달라진다. 문헌만으로 공정한 수치를 정할 수 없으며, asset 측정과 사전 고정이 필요하다.

## 3. 후보 주제 비교와 진행 추천은 무엇인가?

### Takeaway

현재 v1을 즉시 "양팔 협동 RL이 케이블 조립을 개선한다"는 portfolio 주장으로 확장하기보다, **두 팔 능동 권한의 필요성**과 **RL 대 compliant baseline의 조건부 이득**을 분리한 작은 benchmark로 좁히는 편이 문헌과 현재 증거에 맞는다. 문헌은 과제의 산업적 관련성과 물리적 난도를 뒷받침하지만, 현재 한 seed·한 grasp·11 completed episode는 어느 후보의 우위를 정하기에 충분하지 않다.

### Cited Findings

| 후보 | 검증하는 주장 | 문헌 적합도 / 결정적 반례 | 우선도 |
|---|---|---|---|
| A. 현 제안 유지: 양팔이 각자 cable-mounted male/female을 잡고 결합 | cable-induced coupling에서 양팔 동시 적응이 필요하며 RL이 회복한다 | harness 결합은 Streloke et al.이, cable bending insertion 난도는 Wu et al.이 뒷받침한다. 그러나 양팔 connector-to-connector insertion의 선행 정량 증거는 찾지 못했다. [Frontiers](https://www.frontiersin.org/journals/mechanical-engineering/articles/10.3389/fmech.2026.1910151/abstract) [Wu 공개 원고](https://openreview.net/attachment?id=KlQndUeGD8&name=pdf) | **조건부 1순위.** 아래 gate가 먼저 통과할 때만 학습 확장 가치가 있다. |
| B. 한쪽 connector를 fixture/socket에 고정하고 다른 쪽 cable connector 삽입 | 케이블 bending/접촉 교란에서 force-aware residual 또는 curriculum이 고전 기준을 넘는다 | Wu et al.과 Yamazaki et al.의 실제 과제에 더 가깝고, 관측·action·baseline을 작게 유지한다. 단, 양팔 협동 주장은 할 수 없다. [Wu 공개 원고](https://openreview.net/attachment?id=KlQndUeGD8&name=pdf) [Yamazaki et al.](https://www.sciencedirect.com/science/article/am/pii/S0921889024000563) | **1순위 대안.** A의 two-active-arm gate가 실패하면 더 강한 cable-insertion portfolio가 된다. |
| C. connector를 fixture로 hold하고 양팔은 cable routing/shape control만 수행 | multi-agent cable coupling과 routing 일반화 | Streloke et al.의 공개 환경과 가장 닮았으나 insertion force/contact를 검증하지 못한다. end-to-end SAC가 residual RL보다 좋았다는 반례도 있다. [Frontiers](https://www.frontiersin.org/journals/mechanical-engineering/articles/10.3389/fmech.2026.1910151/abstract) | **2순위.** 삽입보다 harness routing 자체가 목표일 때만 선택한다. |
| D. 전용 guide/finger/fixture + 한팔 또는 순응 제어 | 학습보다 mechanical uncertainty reduction이 비용·신뢰도에 유리하다 | NVIDIA가 전용 finger로 repeatable grasp pose를 만든 뒤에도 small connector의 sim-to-real 한계를 기록했다. 이는 A의 강한 현실성 반례다. [NVIDIA 원문](https://developer.nvidia.com/blog/the-machines-that-make-the-machines/) | **필수 반례 baseline.** 결과가 좋으면 RL보다 이 설계가 더 좋은 해법일 수 있다. |

- A를 계속할 최소 gate는 다음과 같다. (i) cable/twist/friction randomization이 있을 때 one-active-arm과 fixed-arm의 성능이 유의하게 떨어지고, (ii) two-active-arm이 같은 safety force limit에서 그 차이를 회복하며, (iii) cooperative compliant baseline도 이기거나 적어도 이기지 못한 조건을 정직하게 보고한다. 이 중 (i)가 없으면 A의 양팔성은 검증되지 않는다.
- baseline 순서는 `fixed/fixed → one-active/fixed → two-active compliant → two-active learned`가 적절하다. 이 순서가 케이블의 물리적 난도, 양팔 능동 제어, 학습 이득을 각각 분리한다. Streloke et al.의 planner·end-to-end RL·residual RL 비교와 Huang et al.의 force-aware residual 대 학습 baseline 비교는 이런 분리 필요성을 뒷받침하지만, 해당 순서 자체를 검증한 것은 아니다. [Frontiers](https://www.frontiersin.org/journals/mechanical-engineering/articles/10.3389/fmech.2026.1910151/abstract) [Huang et al.](https://arxiv.org/abs/2603.04038)
- 보고 시에는 하나의 aggregate success만 쓰지 말고 perturbation 축별 성공률과 force/time/retry를 함께 제시해야 한다. NVIDIA는 large/small connector에서 효과가 달랐고, Wu et al.은 난도별 50회 시험을 사용했다. 이는 조건을 뭉개면 어떤 종류의 일반화가 얻어졌는지 알 수 없다는 근거다. [NVIDIA 원문](https://developer.nvidia.com/blog/the-machines-that-make-the-machines/) [Wu 공개 원고](https://openreview.net/attachment?id=KlQndUeGD8&name=pdf)

### Inferences

- **추천:** A를 "물리 교란을 통제한 양팔 협동 benchmark"로 한정해 먼저 검증하고, gate (i)가 불통과면 B로 전환한다. 이 선택은 목표를 축소하는 것이 아니라, 양팔성과 RL 이득의 원인을 보여 줄 수 있게 만든다.
- **portfolio 문장 초안의 안전한 범위:** "케이블 장력·비틀림·테이블 접촉을 요인으로 분리하고, 고정/한쪽 능동/양쪽 능동의 force-limited 삽입을 비교하는 시뮬레이션 benchmark를 구축했다." 학습 결과가 축적된 뒤에도 "개선했다"는 말은 사전 정의한 baseline·seed·randomization 평가에서 확인된 지표에만 연결한다.
- **현재 v1의 해석:** freshbias8에서 `single 2 / multi 3 / zero 2`, 사전+2 gate 미달이라는 기록은 탐색 신호일 수 있으나, 표본이 작고 one seed/one grasp이므로 A·B·C·D의 선택 근거로 쓰기 어렵다. 이는 문헌 사실이 아니라 주어진 실행 기록에 대한 방법론적 해석이다.

### Gaps

- 위 후보의 최종 선택에는 asset의 실제 접촉 모델, connector clearance, cable stiffness/torsion 범위, force sensor/추정 신뢰도, controller frequency가 필요하다. 이번 문헌 조사 범위 밖이므로 판단을 보류한다.
- 저자 공개 코드를 확인한 자료는 wire-harness 환경뿐이며, 해당 저장소도 LICENSE 파일과 API metadata가 없어 MIT 재사용 가능 여부는 보류한다. 논문 자체의 CC BY 표기는 Frontiers 논문 텍스트에만 적용된다. [Frontiers 원문](https://www.frontiersin.org/journals/mechanical-engineering/articles/10.3389/fmech.2026.1910151/abstract)
