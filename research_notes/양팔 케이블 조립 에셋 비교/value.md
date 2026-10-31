# Critic 판정 수신 요약

이 문서는 2026-10-31에 `/root/cable_project_value` Critic 에이전트가 읽기 전용 검토로 전달한 판정을 보고서 작성자가 요약한 기록이다. Critic이 생성한 기존 연구 노트가 아니며, 원 판정은 별도 파일로 저장되지 않았다.

## 최종 판정

현 제안은 **실행 준비 미달(REJECT), 연구 주제로는 조건부 가치 있음**으로 판정됐다. 최근 SimReady 에셋과 양쪽 free tail을 추가하는 것만으로 양팔 협동이나 강화학습의 필요성이 생기지 않는다. 기존 v1도 오른팔 6D residual만 학습하고 왼팔·진행·중단·복구는 스크립트가 맡았으며, 한 seed·한 파지·각 학습 8,192 transitions·완료 11 episodes에서 미사용 조건 성공이 `single64 2/8`, `multi64 3/8`, `zero 2/8`로 사전 개선 기준을 통과하지 못했다.

## 의미가 생기는 조건

두 케이블의 anchor, 장력 방향, 초기 곡률, 테이블·장애물 접촉이 커넥터의 상대 자세와 삽입력에 힘을 전달하고, 왼팔 움직임이 오른팔의 jam·접촉 하중을 바꿔야 한다. 이 물리 결합이 없다면 두 번째 케이블은 관측과 행동 차원만 늘린다. 학습 전 물리 pilot은 왼팔 변위가 오른팔 삽입력·장력·걸림에 미치는 영향과 `양팔 classical active` 대 `왼팔 고정` 차이만 확인해야 한다. 실행 전부터 learned-left 우위를 요구하면 순환 판정이 되므로 사용하지 않는다.

물리 gate를 통과한 뒤에는 동일 센서, action limits, supervisor, 초기 분포, force limit, 데이터·튜닝 조건에서 `강한 양팔 impedance/admittance baseline`, `오른팔 RL`, `양팔 RL`을 비교한다. baseline을 일부러 약하게 만들거나 결과를 본 뒤 조건을 고르면 판정은 무효다. 양팔 RL의 이득이 없으면 협동 강화학습 주장을 중단한다.

## 권고 우선순위

1순위는 큰 keyed connector 또는 compliant fixture를 사용한 force-coupled rigid 양팔 삽입이다. 2순위는 장력·anchor·장애물 접촉을 명시한 양단 케이블 inline connector다. 3순위는 USB 케이블이 달린 LAN 동글과 Ethernet RJ45 케이블 조합이다. 세 번째 장면은 자연스럽지만 왼쪽 케이블이 수동 외란에 머물 가능성이 커 양팔 RL 대표 과제로는 약하다. 큰 산업용 커넥터가 RB-Y1 기본 그리퍼에 더 적합하다는 평가는 설계 판단이며, 검증 결과가 아니다.

## 물리·센서·평가 제한

SimReady 표시는 유연 케이블 물성을 보장하지 않는다. sim-only 비교에는 가정 물성을 쓸 수 있지만 파라미터 provenance, stiffness·friction 민감도, timestep·분절 수렴, 힘 전달, connector–strain-relief 결합, 파지 유지를 검증해야 한다. actor가 사용할 센서 입력과 training-only 정답 상태를 분리하고, reward에서 정답 자세·힘을 쓰는 것과 actor에 정답이 누출되는 것을 구분해야 한다. 케이블 장력은 이력이 있는 상태이므로 짧은 센서 time history 또는 recurrent policy의 필요성은 검증할 가설로 남긴다. 50 µm는 제조사 공차가 아닌 프로젝트 기준이므로 보조 지표로 유지하고, 주 성공 기준은 geometry 기반 삽입 깊이·dwell·접촉 하중·jam·손상 금지 조건으로 사전 등록한다.

## 중단 기준

케이블이 실제 힘을 전달하지 않거나, 왼팔 고정과 양팔 active의 차이가 없거나, 합리적인 물성·solver 범위에서 방법 순위가 뒤집히면 케이블 양팔 RL 주장을 중단한다. 동일 조건의 강한 classical baseline이 해결하면 난도를 사후에 올리지 않고 과제 우선순위를 다시 평가한다. v1 성공은 v2의 절대 전제가 아니지만, v2는 독립 버전과 독립 판정 계약으로 운영하고 v1의 미입증 결과를 그대로 보존해야 한다.

## 종합 검수 보완

아래 내용은 위 Critic 수신 요약을 고쳐 쓴 원문이 아니라, 보고서 독립 검수에서 추가된 실험 설계 보완이다.

`왼팔 이동 → 오른팔 force 또는 jam 변화`만으로는 케이블 고유 결합을 입증할 수 없다. 왼팔 이동이 커넥터끼리의 상대 정렬, 삽입 깊이, 속도를 바꾼 당연한 결과일 수 있기 때문이다. 비학습 pilot은 커넥터 상대 자세·삽입 깊이·속도 profile을 같게 유지하고, 두 팔과 두 물체를 함께 움직이는 common-mode 조건 또는 tail anchor·초기 곡률·장력만 바꾸는 조건을 사용한다. `cable off / rigid tail / slack / tensioned / obstacle-contact`를 같은 초기화로 짝지어 비교하고, 반대 손 wrench와 커넥터 상대 자세를 함께 기록해 geometry confound를 제거한다.

rigid force-coupled 과제에서도 왼팔을 움직였다는 사실 자체는 양팔 필요성의 gate가 아니다. reach·관절 토크·접촉력 제한 때문에 한 팔만으로 만족하기 어려운 조건에서 양팔 task-space 권한이 효용을 갖도록 설계한다는 가정이며, 아직 측정된 결과가 아니다. `FET_100_ISAAC=false`는 optional Isaac composition 요구를 충족하지 못한 비차단 표기로 확인됐고 실행 실패를 뜻하지 않는다. 현재 고정 런타임에서 실제 실행하지 않았으므로 호환 미확인 판정은 유지한다.
