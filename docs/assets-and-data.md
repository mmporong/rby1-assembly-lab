# 자산과 데이터 사용 원칙

이 프로젝트는 RB-Y1 공식 시뮬레이터 자산을 고정된 Git revision에서 내려받고, Isaac Sim 자산과 AutoMate 데이터는 실제 장면·참고 데이터·준비된 선택 항목으로 구분한다. 큰 바이너리와 라이선스가 불분명한 USD를 저장소에 복제하지 않는다.

## 자산 상태

| 자산 | 현재 상태 | 획득 방식 | 검증 |
|---|---|---|---|
| RB-Y1 Model M v1.2 USD | 실제 생성 장면의 로봇 articulation | Rainbow Robotics 저장소의 commit `e2415548e5d6db712ecbff2074741b70261d0cbf`에서 런타임 다운로드 | 파일 크기와 SHA-256, USD 관절·충돌·articulation 감사 |
| RB gripper 좌·우 USD와 mount JSON | 실제 생성 장면의 양손과 손목 부착 | 같은 commit에서 런타임 다운로드 | 원본 충돌 누락과 생성 장면의 collider 6개, mount frame 감사 |
| 자체 생성 plug/socket/table | 현재 실제 조립 장면 | `configs/task.json`의 명시적 치수로 생성 | 자유 rigid object 2개, 소켓 내부 공간과 벽 collider 4개 감사 |
| AutoMate `00015` peg/socket | 형상·SDF 설계의 비교 참고 자료 | Isaac 5.1 asset service의 `/Isaac/IsaacLab/AutoMate/00015/` | 최상위 USD와 6개 구성 layer, OBJ의 크기·SHA-256 |
| AutoMate grasp·거리·분해 궤적 | 초기분포·복구 curriculum 참고 자료. 아직 정책 학습에는 미사용 | 같은 asset service | JSON parsing, asset ID와 episode shape 검사 |
| Isaac Simple Room | 선택형 시각 배경으로 준비됨. 현재 scene에는 미연결 | Isaac Sim asset root의 `/Isaac/Environments/Simple_Room/simple_room.usd` | versioned URL 응답과 최상위 파일 hash 확인 |
| Seattle Lab Table | 선택형 공식 작업대로 준비됨. 현재 scene은 자체 생성 table 사용 | Isaac Sim asset root의 `/Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd` | versioned URL 응답과 최상위 파일 hash 확인 |

`configs/assets.json`에는 다운로드 파일의 URL·크기·SHA-256과 외부 asset-root 경로를 기록했다. `scripts/fetch_assets.py`는 manifest에 등록된 ID만 받는다. 임의 URL 입력은 받지 않으며 HTTPS 호스트 allowlist, 응답 크기 제한, timeout, 임시 파일 다운로드, SHA-256 확인을 통과한 뒤 원자적으로 cache에 옮긴다.

```bash
cd "$HOME/rby1-assembly-lab"
.venv/bin/python scripts/fetch_assets.py
.venv/bin/python scripts/audit_assets.py
.venv/bin/python scripts/audit_assets.py --generated --output artifacts/asset-audit.json
```

cache 기본 경로는 `$HOME/rby1-assembly-lab/assets/cache/`이다. 이 디렉터리는 Git에서 제외해야 하며 배포물에도 포함하지 않는다.

Simple Room과 Seattle Lab Table은 여러 하위 자산을 asset resolver로 읽으므로 최상위 USD만 cache에 복사하지 않는다. Isaac Sim 5.1의 versioned HTTPS URL이 응답하는지 확인하고 최상위 파일의 크기·SHA-256·확인일을 manifest에 기록했으며, 실행 때는 `asset_root_path`로 전체 의존성을 해석한다.

## SimReady 표기와 물리 감사의 차이

NVIDIA 문서가 제공하는 Isaac Sim asset-root 자산은 시뮬레이션용으로 제공되지만, 이 프로젝트의 task에 필요한 물리 조건을 자동으로 보장하지는 않는다. 따라서 “SimReady 자산을 불러왔다”와 “조립 접촉에 쓸 수 있다고 확인했다”를 구분한다.

감사기는 다음 항목을 기록한다.

- stage의 default prim, up axis, meters-per-unit
- articulation root와 joint 종류·개수, revolute/prismatic limit
- collision, rigid body, authored mass 또는 density 수
- Isaac Sim 밖에서 읽을 때 해석되지 않는 USD·MDL 의존성
- PhysX Python plugin 없이도 읽을 수 있는 applied schema 이름

시각 배경은 collision이 없어도 경고만 남긴다. 작업대처럼 접촉에 쓰는 정적 자산은 collision이 없으면 실패한다. 움직이는 부품은 collision과 mass/density가 모두 필요하다. RB-Y1 USD에서 `OmniPBR.mdl` 같은 의존성이 standalone `usd-core`에서 해석되지 않는 것은 기록하되, Isaac Sim runtime의 resolver가 제공하는 항목과 구분한다.

원본 RB gripper USD의 `/collisions` Xform은 비어 있어 원본 감사가 의도적으로 실패한다. 생성 단계는 좌우 손가락 네 개와 palm 두 개의 원본 CAD visual mesh instance를 풀고 `convexHull` collision을 적용한다. `--generated` 감사는 이 6개 collider, 원본 4개 파일의 SHA-256, 손목 링크와 end-effector 사이의 `-0.1261 m`·identity quaternion mount frame, 두 개의 자유 rigid object, 네 벽으로 둘러싸인 소켓 내부 공간을 확인한다. 보고서는 원본 결함과 생성된 보완 결과를 분리해 보존한다.

## 라이선스와 provenance

RB-Y1 upstream README는 NVIDIA SPDX header가 있는 소스 파일을 Apache-2.0으로 설명하지만 USD 자산의 재배포 조건은 명확히 적지 않는다. 그래서 USD와 gripper JSON은 저장소에 넣지 않고 출처 commit과 hash만 고정한다. NVIDIA 환경 자산도 제품의 asset service 또는 설치된 asset pack에서 불러온다. 자산을 외부에 배포하기 전에는 각 upstream 약관을 다시 확인해야 한다.

## AutoMate `00015` 사용 범위

Isaac Lab v2.3.0의 AutoMate 설정이 가리키는 asset root에서 `00015` 8 mm peg/socket 한 쌍을 골랐다. 최상위 USD, 구성 layer, SDF 계산용 OBJ, 전체 grasp·분해거리 표와 `00015` 분해 궤적을 내려받아 프로젝트 manifest에서 새 SHA-256으로 고정했다. 서비스가 checksum을 게시한 것이 아니라 이 프로젝트가 받은 정확한 바이트를 기록한 값이다.

확인된 데이터는 grasp 100개 ID 중 `asset_00015`의 7차원 grasp pose, 분해거리 `0.027 m`, 분해 episode 100개이며 각 episode는 128 step이다. episode에는 fingertip pose, arm joint position, plug grasp·초기 pose와 시간별 plug pose가 들어 있다. 이 값은 Franka로 생성된 AutoMate 시연이므로 RB-Y1 action을 그대로 지도학습하지 않는다. 조립 순서와 residual recovery 초기 분포를 설계하는 참고 데이터로 쓰고, RB-Y1 양팔 trajectory는 새 환경에서 다시 생성한다.

`plug.usd`와 `socket.usd`는 같은 디렉터리의 `configuration/*_base.usd`, `*_physics.usd`, `*_sensor.usd`를 상대 경로로 읽는다. fetch script는 이 의존 layer까지 같은 디렉터리 구조로 보존한다. standalone `usd-core` 감사에서 남는 `OmniPBR.mdl`은 Isaac Sim runtime이 제공하는 재질 의존성이다.

plug USD 자체에는 mass가 authoring되어 있지 않고 Isaac Lab 설정이 `0.019 kg`을 spawn 시 적용한다. manifest의 `runtime_mass_kg`에 이 값을 명시해, 감사 결과가 “USD에 내장된 질량”과 “환경 설정이 주입할 질량”을 구분하도록 했다.

또한 NVIDIA의 공개 단일팔 조작 시연이나 Franka용 grasp 데이터는 RB-Y1 양팔 데이터가 아니다. 참고·사전 실험에 사용할 수는 있어도 RB-Y1 정책의 학습 데이터나 실물 검증 결과로 이름을 바꾸지 않는다. 이 프로젝트의 학습 데이터는 RB-Y1 환경에서 생성한 양팔 관측·행동·접촉·성공 판정과 출처를 별도 기록한다.
