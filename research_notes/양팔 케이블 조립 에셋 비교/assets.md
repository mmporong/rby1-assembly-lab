# 양팔 USB–LAN/RJ45 케이블 조립 에셋의 공개 근거와 물리 범위

조사 기준일은 2026-10-31이다. 범위는 NVIDIA SimReady Central의 공개 등록, 해당 Hugging Face 데이터셋의 고정 커밋 메타데이터·USD 레이어, NVIDIA Isaac Sim 5.1.0 공식 문서, 제조사 공개 제품 문서로 제한했다. 대용량 에셋 수집·Isaac Sim 실행·GPU/Brev 사용은 수행하지 않았다. 아래의 Hugging Face `lastModified`와 USD `usd_date_generated`는 각각 데이터셋 커밋 수정 시각과 생성 필드이며, 자산의 공식 출시일로 간주하지 않는다.

## 1. 최근 공개 SimReady 에셋에 `휴대형 USB–LAN 어댑터 + RJ45 케이블` 조합이 있는가?

### Takeaway
**정확히 그 조합인 공개 SimReady 에셋은 확인하지 못했다.** 가장 가까운 공개·등록 에셋은 ForgeCAD의 `data_connector_pack`이며 USB-A, USB-C, RJ45 플러그와 포트 패널을 한 패키지에 제공한다. 그러나 USB–LAN 어댑터 본체에 연결된 짧은 USB 케이블과, 반대 손이 잡을 별도 Ethernet 케이블/RJ45 플러그의 조합은 패키지 구성·물리 레이어에서 확인되지 않았다. 따라서 이를 "이미 준비된 케이블 조립 에셋"으로 취급하면 안 된다.

### Cited Findings
- NVIDIA SimReady Central은 공개 Hugging Face 데이터셋의 **특정 커밋**을 재현 가능한 자산 버전으로 사용하며, 소비자는 target profile, validation evidence, license를 검토하도록 안내한다. [NVIDIA SimReady Central Library](https://docs.omniverse.nvidia.com/simready/latest/simready-central.html)
- Central 등록 파일은 ForgeCAD의 `data_connector_pack`을 고정 커밋 `0c48fab70fb48d580d58b50319e395b85605512e`의 패키지 정의로 가리킨다. [SimReady Central 등록 원본](https://raw.githubusercontent.com/NVIDIA-Omniverse/simready-central/bd491523d15700220fa8293112056860f75a0880/submissions/ForgeCAD/simready-assets)
- 그 등록 커밋의 dataset metadata는 `lastModified=2026-10-20`과 `CC-BY-4.0` 라이선스를 기록한다. 이 수정 시각은 공식 출시일 근거가 아니다. [고정 커밋 metadata](https://huggingface.co/api/datasets/ForgeCAD/simready-assets/revision/0c48fab70fb48d580d58b50319e395b85605512e)
- 고정 USD 루트는 `metersPerUnit = 1`, `upAxis = "Z"`, 생성일 `2026-10-18`을 명시하고 `Plug_USB_A`, `Plug_USB_C`, `Plug_RJ45` prim을 참조한다. 케이블 prim은 이 루트 레이어에서 확인되지 않는다. [고정 USD 루트](https://huggingface.co/datasets/ForgeCAD/simready-assets/resolve/0c48fab70fb48d580d58b50319e395b85605512e/data_connector_pack/simready_usd/data_connector_pack.usda)
- ForgeCAD 패키지 BOM은 USD, geometry, physics, material, texture 파일을 열거하지만 케이블 전용 파일을 열거하지 않는다. [고정 BOM](https://huggingface.co/datasets/ForgeCAD/simready-assets/resolve/0c48fab70fb48d580d58b50319e395b85605512e/data_connector_pack/.metadata/com.nvidia.simready.packaging.bom.json)
- 실제 제품 형태의 기준은 별도로 존재한다. StarTech의 USB31000S는 USB 3.0 male 쪽과 RJ45 포트를 가진 USB–Gigabit Ethernet 어댑터이며, 공식 공개 제품 페이지에서는 USD/CAD 또는 물리 파라미터 패키지를 제공한다는 근거를 찾지 못했다. [StarTech USB31000S 공식 제품 페이지](https://www.startech.com/en-us/networking-io/usb31000s)

### Inferences
- ForgeCAD 패키지는 **말단 커넥터와 수동 포트 접촉 형상**의 후보이며, 과제의 두 손-두 케이블 외란 시나리오를 완성하는 후보는 아니다.
- StarTech 제품은 장면 의미와 외형 기준에는 유용하지만, 이번 조사 범위에서는 재사용 가능한 공개 CAD/SimReady 소스로 검증되지 않았다.
- 이 결론은 공개 Central 등록과 가장 가까운 공개 패키지의 파일·prim 목록에 대한 한정된 음성 결과다. NVIDIA/Hugging Face 전체에 비공개·미등록·이름이 다른 에셋이 없다는 전역 부재 증명은 아니다.

### Gaps
- `data_connector_pack`의 시각 썸네일과 11.4 MB geometry 레이어를 렌더링하여 실제 어댑터 본체의 형태·스케일을 대조하지 않았다. 이 작업은 현 단계의 대용량 자산 수집 금지 범위에 포함된다.
- 제조사별 USB–LAN 어댑터의 공개 CAD는 일부 포털에서 로그인·이용약관 동의를 요구할 수 있다. 자격증명 탐색이나 로그인은 수행하지 않았으므로, 사용 가능한 라이선스가 있는 공급사 CAD 후보는 확정하지 못했다.

## 2. 후보별로 외형·접촉·질량/재료·유연 케이블 동역학·현재 런타임 호환을 어떻게 판정해야 하는가?

### Takeaway
현재 공개 근거상 **ForgeCAD `data_connector_pack`만 재사용 가능한 SimReady 후보**다. 다만 이는 rigid PhysX prop 검증의 일부를 통과한 커넥터 팩이며, physics-calibrated flexible cable은 아니다. 고정 validation에는 PhysX collision/rigid-body 관련 feature가 통과로 기록되지만 `FET_100_ISAAC`, Newton, MuJoCo 관련 feature는 실패로 기록되어 Isaac Sim 5.1.0/Isaac Lab 2.3.0에서의 완성 장면 호환성은 실행 전 검증되지 않았다.

### Cited Findings

| 후보 | 실제 위치·게시/버전·라이선스 | 제공 근거 | 케이블 동역학 | 고정 런타임 판정 |
| --- | --- | --- | --- | --- |
| ForgeCAD `data_connector_pack` | [Central 등록](https://raw.githubusercontent.com/NVIDIA-Omniverse/simready-central/bd491523d15700220fa8293112056860f75a0880/submissions/ForgeCAD/simready-assets), 고정 [HF commit](https://huggingface.co/api/datasets/ForgeCAD/simready-assets/revision/0c48fab70fb48d580d58b50319e395b85605512e), CC-BY-4.0. `lastModified=2026-10-20`은 release date가 아닌 커밋 수정 시각이다. | root USD에 USB-A/C, RJ45와 meter/Z-up; physics 레이어에 rigid bodies, convex-hull colliders, RJ45 latch revolute joint, 질량·밀도·마찰·반발 값이 있다. [USD root](https://huggingface.co/datasets/ForgeCAD/simready-assets/resolve/0c48fab70fb48d580d58b50319e395b85605512e/data_connector_pack/simready_usd/data_connector_pack.usda), [physics USD](https://huggingface.co/datasets/ForgeCAD/simready-assets/resolve/0c48fab70fb48d580d58b50319e395b85605512e/data_connector_pack/simready_usd/forgecad/physics.usda) | 케이블 prim·bending/torsion stiffness·damping·self/contact calibration 없음. RJ45의 짧은 `stub`은 존재해도 물리적 유연 케이블의 근거가 아니다. [physics USD](https://huggingface.co/datasets/ForgeCAD/simready-assets/resolve/0c48fab70fb48d580d58b50319e395b85605512e/data_connector_pack/simready_usd/forgecad/physics.usda) | `Robotics-Prop 4.0.0`, PhysX rigid/collision feature 일부 pass, `FET_100_ISAAC=false`, Newton/MuJoCo feature false. Isaac Sim 5.1.0 실행 호환은 **미검증**. [validation](https://huggingface.co/datasets/ForgeCAD/simready-assets/resolve/0c48fab70fb48d580d58b50319e395b85605512e/data_connector_pack/.metadata/data_connector_pack_validation.json) |
| StarTech USB31000S 제품 | [제조사 공식 제품 페이지](https://www.startech.com/en-us/networking-io/usb31000s) | USB 3.0–RJ45 어댑터라는 과제 의미에 맞는 실제품 기준만 확인됨. 공개 OpenUSD/SimReady, collision mesh, 질량/관성, 재료·마찰, 라이선스된 CAD 배포 근거는 이번 조사에서 없음. | 제공 근거 없음. | Isaac Sim/PhysX 사용 자산으로는 **후보 아님**; 외형·치수의 추가 확인 대상일 뿐. |
| NVIDIA SimReady Warehouse 02 | [NVIDIA 공식 pack 안내](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html) | 178 USD 모델, 20.5 GB pack으로 공시되나 문서의 예시는 pallet/rack/ramp다. USB–LAN/RJ45 조합이나 케이블 물리 정보는 안내 페이지에서 확인되지 않는다. | 제공 근거 없음. | 대용량 다운로드 금지 때문에 파일명·USD metadata 검증을 하지 않았으며, 현재 과제 자산 후보로 채택할 근거가 없다. |

- NVIDIA는 SimReady를 시각 3D 모델 그 이상으로 설명하지만, 자산은 목표 profile에 필요한 metadata와 validation evidence를 보유해야 한다. [SimReady specification](https://docs.omniverse.nvidia.com/simready/latest/overview/simready-spec.html)
- NVIDIA의 SimReady FAQ는 cable을 전력·구조·광·제어·유압 등 서로 다른 속성을 지닌 도메인으로 구분하고 stiffness, conductivity, bend radius, fatigue life 등을 예로 든다. 현재 표준화된 robotics profile은 rigid body, collider, grasp, articulation 중심이다. [SimReady FAQ](https://docs.omniverse.nvidia.com/simready/latest/simready-faq.html)
- Isaac Sim 5.1.0은 PhysX 기반 rigid body, deformable body, particle, articulation을 지원하며, imported asset에는 적절한 collision 및 joints를 추가해야 한다고 안내한다. [Isaac Sim 5.1 reference architecture](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/introduction/reference_architecture.html)
- Isaac Sim 5.1.0 API의 D6 joint는 두 rigid body를 연결하고 axis, limit, stiffness, damping, force/torque limit을 설정할 수 있다. [D6 API](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/api/structomni_1_1isaac_1_1dynamic__control_1_1_dc_d6_joint_properties.html)
- GPU dynamics에서는 CCD 요청이 무시되고, GPU deformable에는 GPU dynamics가 필요하다. 따라서 얇은 케이블과 삽입 접촉의 통과·관통은 문서상 사전 보장되지 않는다. [Isaac Sim 5.1 PhysicsContext API](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.api/docs/index.html)

### Inferences
- **외형/조립 접촉:** ForgeCAD의 USB/RJ45 terminal geometry와 convex-hull collision은 출발점이 될 수 있다. USB–LAN 어댑터 본체 형상, 두 케이블의 길이·외피, RJ45 상대 port/socket, 삽입 공차와 latch의 체결/분리 기준은 별도 범위다.
- **유연부 표현:** 현 고정 PhysX 런타임에서는 작은 rigid segment와 D6 constraints의 체인이 가능한 경로다. 이 경로는 제품 수준 material calibration이 아니므로 세그먼트 길이/질량/관성, 축별 limit, bending·torsion stiffness/damping, end-effector와의 부착, self-collision, 커넥터/포트 contact를 정하고 시험으로 확인해야 한다. D6 지원 사실은 가능 경로의 근거일 뿐, 케이블 정확도의 근거는 아니다.
- **물리 재료:** ForgeCAD 레이어의 density 1150/1400 kg/m³ 및 dynamic/static friction 0.3, restitution 0.1은 port-panel/connector physics material에 적힌 값이다. 케이블 jacket·도체의 굽힘·비틀림 특성값으로 전용하면 안 된다. [physics USD](https://huggingface.co/datasets/ForgeCAD/simready-assets/resolve/0c48fab70fb48d580d58b50319e395b85605512e/data_connector_pack/simready_usd/forgecad/physics.usda)
- **버전 경계:** 여기의 Isaac Sim은 고정된 5.1.0 문서이고, SimReady validation은 `2026.8.0` 도구 결과다. `FET_100_ISAAC=false`인 등록 자산을 Isaac Sim 5.1.0에서 검증 완료라고 읽을 수 없다. Isaac Sim 5.1.0 버전명과 PhysX SDK의 별도 버전 표기는 같은 번호라는 뜻이 아니다.
- **프로젝트 의미:** 최근 공개 자산을 사용했다는 사실은 양팔이 케이블 외란 하에서 USB–LAN/RJ45 삽입을 수행한다는 연구 가치를 만들지 않는다. 가치는 (a) 두 terminal의 grasp와 상대 pose, (b) 유연부 외란, (c) 삽입 성공/접촉 안정성/실패 회복을 분리 측정하고, 실물 또는 근거 있는 파라미터와 대조할 때 생긴다.

### Gaps
- D6 chain의 bending/torsion/contact 안정성, GPU PhysX 설정별 관통, RB-Y1 grasp 중 외란 재현성은 실행·파라미터 sweep 없이는 판정할 수 없다. 이번 지시의 실험 금지로 검증하지 않았다.
- Newton의 최신 cable/object 실험 기능은 이 과제의 Isaac Sim 5.1.0/PhysX GPU 고정 환경 지원 근거로 사용하지 않았다. 최신 Isaac Sim release note의 Newton 항목은 다른 최신 release stream의 내용이므로 호환 추정 근거가 아니다. [최신 Isaac Sim release notes](https://docs.isaacsim.omniverse.nvidia.com/latest/overview/release_notes.html)

## 3. 구현 전 프로젝트 실현 가능성의 조건부 결론은 무엇인가?

### Takeaway
**조건부로 실현 가능하나, 현 공개 에셋만으로는 시작 조건이 충족되지 않는다.** ForgeCAD 팩을 terminal/port 후보로 한정하고 새 유연 케이블과 USB–LAN 어댑터 본체·상대 port를 범위에 포함시킬 때에만 물리 과제로 성립한다. 그 뒤의 첫 검증은 자산 로딩이 아니라 정적 삽입 공차와 케이블 체인 contact 안정성의 최소 장면 검증이어야 한다.

### Cited Findings
- SimReady validator는 개별 asset/feature의 구조와 선언 capability를 확인하며, fully composed scene의 문제는 target simulator runtime에서 드러난다고 NVIDIA가 명시한다. [SimReady FAQ의 scene-level validation 제한](https://docs.omniverse.nvidia.com/simready/latest/simready-faq.html)
- Isaac Sim 5.1.0 문서는 collision 형상으로 convex hull, convex decomposition, SDF tri-mesh 등을 설명하고, 각 근사는 사용 사례에 따라 달라진다고 안내한다. [Isaac Sim 5.1 reference architecture](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/introduction/reference_architecture.html)

### Inferences
- **Go 조건:** USB/RJ45 terminal의 어느 prim을 손에 쥘지, USB–LAN 어댑터와 RJ45 port의 nominal frame·clearance, 양 cable의 길이/질량/외피 반경, cable-to-hand/end-frame 부착, 성공·실패 metric을 문서화할 수 있어야 한다.
- **No-go 조건:** `data_connector_pack`을 두 케이블·어댑터의 물리 교본으로 간주하거나, rigid collision metadata만으로 cable 외란을 주장하는 계획은 중단해야 한다.
- 후속 자산 제작 범위는 terminal geometry 재사용 여부와 무관하게 `adapter rigid body + USB pigtail + Ethernet cable + mating port/socket + collision proxies + cable chain parameters + validation scenes`다. 각 항목은 asset import가 아닌 과제의 물리 가설을 위해 필요하다.

### Gaps
- 실제 사용할 USB–LAN 어댑터와 Ethernet plug/port의 제조사·품번이 아직 정해지지 않아 치수, 질량, 재질, 삽입력, latch force, cable bend radius의 근거값을 확정할 수 없다.
- RB-Y1 gripper 형상과 가능한 grasp pose가 이 문서 범위에는 없으므로, ForgeCAD terminal의 크기·collision proxy가 양손 grasp에 적합한지는 판단하지 않았다.
