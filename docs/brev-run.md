# Brev 실행 절차

## 실행 범위

기존 작업과 겹치지 않는 `$HOME/rby1-assembly-lab` workspace와 별도 Python 환경을 사용한다. 이 프로젝트의 Brev 컴퓨팅 예산은 2026-10-25부터 2026-11-25 이전까지 최대 700 USD로 승인됐다. 실행 전 다른 GPU 프로세스를 조회하며 다른 프로젝트의 프로세스·데이터·가상환경을 변경하지 않는다.

## 비용 한도와 실행 단계

비용 계약의 정본은 `configs/compute-budget.json`이다. 이 파일은 이 프로젝트의 컴퓨팅 비용을 계획하고 추정하기 위한 것이며 Brev 계정 전체 잔액이나 다른 프로젝트의 사용액을 확인한 기록이 아니다.

초기 인스턴스 `rby1-assembly-20261025`는 Crusoe의 L40S 48 GB 1장 구성으로, 생성 시점 카탈로그 가격은 시간당 1.74 USD다. 가격이 유지되고 추가 비용이 전혀 없다고 단순 계산하면 700 USD는 약 402.3 GPU 시간에 해당한다. 스토리지·세금·가격 변경·다른 프로젝트 사용액은 이 계산에 포함되지 않으므로 실제 사용 가능 시간으로 보장하지 않는다.

비용은 다음 순서로 사용한다.

1. 단일 환경에서 런타임·PhysX 장면 로딩과 물리 step을 확인한다.
2. 실제 마찰 파지와 기준 제어의 조립 완료를 검증하고 성공·실패 rollout을 기록한다.
3. 스모크와 기준선이 통과한 뒤에만 환경 수와 학습 반복을 늘린다.
4. 각 실행은 시작·종료 시각, 인스턴스 유형, 카탈로그 단가, 추정 비용과 산출물 경로를 실행 기록에 남긴다.

승인 한도는 필요한 검증과 학습에 충분히 쓸 수 있는 상한이다. 예산 소진만을 목적으로 무한 실행하지 않으며, 진전이 없는 반복 실행은 원인을 확인한 뒤 재개한다. 이 프로젝트에서 시작한 서버와 장기 실행 프로세스는 결과를 회수한 뒤 종료하고 상태를 확인한다.

실행 장부에는 인스턴스 ID·이름, 시작 시각의 근거, 단가 조회 시각, 누적 추정액, 예정 종료 시각을 저장한다. 새 실행 전에 기존 추정액·예정 실행액·부대비용 여유분을 합해 한도와 대조한다. 2026-10-25 준비 실행의 장부는 `artifacts/cloud/run-20261025.json`이며, 부대비용 여유분 70 USD를 별도로 두었다. 이 값은 실제 부대비용의 측정치가 아니다.

현재 실행에는 로컬 제어 PC의 systemd 타이머와 `scripts/stop_brev_instance.py`를 사용한다. helper는 명시한 조직에서 ID와 이름이 모두 일치하는 서버만 중지하고 `STOPPED`를 재확인한다. 중지 명령이 실패하거나 시간 초과되어도 실제 상태를 확인하며, `STOPPING`이면 상태 조회만 수행한다. 타이머의 서비스 실패는 30초 후 재시도하되 600초 내 최대 3회로 제한한다. 로컬 PC·네트워크·Brev 인증이 유지되어야 하므로 공급자 수준의 과금 한도 보장은 아니다. 정상 완료나 중단 시에는 시한을 기다리지 않고 서버를 중지한 뒤 타이머를 해제한다. 다른 프로젝트의 실행 중인 서버는 중지 대상이 아니다.

서버가 켜진 뒤 GPU 모델·드라이버·디스크·설치 환경을 먼저 확인한다. 현재 설치 환경을 읽기 전용으로 기록하는 명령은 다음과 같다.

```bash
cd "$HOME/rby1-assembly-lab"
python3 scripts/preflight.py --output artifacts/preflight-before.json
```

미설치 환경에서는 exit code 2가 정상적인 진단 결과다. 이 검사는 Isaac Sim을 시작하지 않는다. GPU 조회와 패키지 확인 통과만으로 PhysX 실행을 인증하지 않는다.

## 별도 환경 설치

아래는 Linux x86_64용 고정 버전 설치 절차다. 원격에 호환 환경이 있으면 먼저 검증하고 사용하며, 기존 환경에 패키지를 덮어쓰지 않는다. 다운로드 전에 남은 디스크 용량을 확인한다. Ubuntu 22.04의 기본 Python 3.10 대신 Python 3.11 환경이 필요하다.

```bash
cd "$HOME/rby1-assembly-lab"
python3.11 -m venv .venv-isaac
source .venv-isaac/bin/activate
sudo apt-get update
sudo apt-get install -y libglu1-mesa
python -m pip install --upgrade pip
python -m pip install -c configs/isaac-compat-constraints.txt setuptools wheel
python -m pip install 'torch==2.7.0' 'torchvision==0.22.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c configs/isaac-compat-constraints.txt 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
python -m pip install --no-build-isolation 'flatdict==4.0.1'
mkdir -p .cache/upstream
git clone --depth 1 --branch v2.3.0 https://github.com/isaac-sim/IsaacLab.git .cache/upstream/IsaacLab
test "$(git -C .cache/upstream/IsaacLab rev-parse HEAD)" = '3c6e67bb5c7ada942a6d1884ab69338f57596f77'
python -m pip install -c configs/isaac-compat-constraints.txt \
  -e .cache/upstream/IsaacLab/source/isaaclab \
  -e .cache/upstream/IsaacLab/source/isaaclab_assets \
  -e .cache/upstream/IsaacLab/source/isaaclab_tasks \
  -e '.cache/upstream/IsaacLab/source/isaaclab_rl[rsl_rl]' \
  -e . 'rsl-rl-lib==3.0.1' 'torch==2.7.0' 'torchvision==0.22.0'
python -m pip check
```

`isaaclab==2.3.0`은 설치 패키지 버전이 아니다. 릴리스 태그 v2.3.0의 `isaaclab` 배포 버전은 0.47.2이며, Isaac Sim 5.1.0의 배포 버전은 5.1.0.0이다. 사전 검사는 이 구분을 반영한다. NumPy는 `isaacsim-kernel`이 요구하는 1.26.0으로 고정했다. `flatdict` 빌드에는 `pkg_resources`를 제공하는 setuptools가 필요하고, 최신 Hugging Face 의존성은 Sim의 click·idna 고정값과 충돌해 호환 제약 파일을 적용한다.

Crusoe 기본 이미지에서 확인한 565.57.01 드라이버는 프로젝트 전용 VM에서 R580.196.07로 갱신하고 재부팅했다. 드라이버 패키지 hold와 남은 `libnvidia-extra-565` 파일 충돌을 해결한 뒤 GPU 조회를 확인했다. 공유 서버에 이 변경을 적용하지 않는다. R580 갱신과 CUDA 연산 통과만으로 Isaac Sim의 RTX·PhysX 호환성이 입증되는 것은 아니다. 근거: [Isaac Sim 요구사항](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html), [Crusoe 드라이버 안내](https://docs.crusoecloud.com/resources/troubleshooting/index.html).

Crusoe 기본 이미지에는 Isaac Sim이 요구하는 `libGLU.so.1`이 없어 `libglu1-mesa`를 설치했다. 사용자가 NVIDIA Omniverse EULA를 승인한 뒤 각 런타임 명령에 `OMNI_KIT_ACCEPT_EULA=YES`를 명시한다. 프로젝트 스크립트는 이 값을 자동으로 설정하지 않는다. Isaac Sim 런타임에 포함된 USD를 사용하며 이 환경에 standalone `usd-core`를 추가로 설치하지 않는다.

근거: [Isaac Sim Python 설치](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html), [Isaac Lab 2.3 설치](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html). Lab의 RSL-RL 버전은 고정 source revision의 `source/isaaclab_rl/setup.py`를 기준으로 한다.

## 확인한 실행 범위

2026-10-25 전용 L40S에서 패키지 사전 검사, `pip check`, CUDA 텐서 연산, Isaac Sim·PhysX 장면 로딩과 RTX 렌더링을 확인했다. 현재 로컬 Python 3.11 + USD 검사 환경의 일반 테스트 76개와 Torch 기반 제어·접촉 계산 테스트 25개가 통과했다.

기준 성공 실행 22의 단일 환경 zero-residual 제어기는 고정 조인트나 물체 자세 덮어쓰기 없이 네 손가락의 마찰 접촉을 유지했다. 두 부품을 각각 초기 높이보다 15 mm 이상 들어 올린 뒤 실제 삽입 깊이 30.66 mm, 횡오차 1.08 mm, 유지 시간 0.4초로 44.07초에 독립 성공 판정을 통과했다. 실행 seed는 `20261025`이며 근거 묶음은 `docs/evidence/nominal-baseline.json`에 있다.

이 성공은 삽입 합력과 개별 법선 하중의 합이 모두 0 N인 여유 끼워맞춤 nominal 조건 한 건이다. 따라서 잼 회복, 잔차 강화학습 성능, 카메라 기반 제어, 실물 검증의 근거로 사용하지 않는다. 기준 제어와 단계 전환은 시뮬레이터 정답 상태를 사용하고, 정책 관측은 정답 상태에 잡음을 더하는 별도 계약이다. 로봇은 고정 베이스이며 자체 충돌은 비활성화되어 있다.

근접 촬영 실행 24와 전체 구도 실행 25는 같은 성공 동작을 다시 기록했다. 공식 Isaac Simple Room을 충돌 없는 시각 배경으로 사용하며, 근접 카메라의 원본은 1280×960·120 fps다. 공유용 4배속 근접 영상과 정상 속도 전체 구도 영상은 H.264·60 fps이며 각각 약 13초와 44초다. 이전 30 fps 4배속 편집보다 표시 프레임 사이의 시뮬레이션 시간 간격이 절반으로 줄었다. 이 카메라는 관찰용 viewer이며 제어 입력이 아니다. 프레임 수·타임스탬프·소스 해시·기존 성공 trace 대조 결과는 [영상 검사 근거](evidence/video-cadence.json)에 있다.

기준선 기록 `baseline-rollouts-final.h5`는 서로 다른 3개 seed가 모두 독립 성공·유효 판정을 통과했다. 결정적 분할은 train 2개·eval 1개이며 학습 가능 episode는 train 2개다. 각 episode의 source hash와 함께 보관한 원본 source archive도 검증했다. 정확한 파일 경로·hash·episode 수치의 정본은 [기준선 데이터 근거](evidence/baseline-dataset.json)다. 기록 당시의 소스·설정·에셋 해시와 원본 소스 묶음을 보관한다. 평가기 불일치를 발견했던 초기 HDF5는 진단용으로 보존한다.

PPO v2 runtime smoke는 `artifacts/cloud/ppo-smoke-v2/2026-10-25_08-02-57`에서 4환경·1 iteration·환경당 24 step, 총 96 transition을 실행했다. 프로세스는 exit 0으로 끝났고 체크포인트 tensor는 모두 유한했으며 optimizer 상태와 parameter update를 확인했다. 초기 episode 일부만 지난 실행이므로 조립 성공률이나 잔차 정책 성능의 근거가 아니다.

첫 PPO 실행에서 RB-Y1 원본의 wheel collision group 4개가 환경마다 복제되며 PhysX fast replication과 충돌했다. 수정된 장면은 `replicate_physics=False`인 일반 USD clone과 GPU 병렬 실행을 사용한다. 원본 USD는 바꾸지 않고, 자체 충돌이 꺼진 경우에만 환경별 wheel group의 runtime 복제본을 비활성화하며 전역 지면과 환경 간 필터는 유지한다. 4환경 검사에서는 env 0 plug를 env 1 palm에 겹친 음성 대조의 필터 기반 금지 접촉 하중과 미필터 palm net force가 모두 0 N이고 종료도 없었다. reset 뒤 env 0 내부 금지 접촉은 env 0만 abort시키고 나머지 세 환경은 계속 진행했다. 모든 검사 항목의 정본은 [병렬 장면 검사](evidence/parallel-scene-probe.json)다.

원격 Isaac 가상환경에서 앱을 시작하지 않고 전체 단위 테스트를 실행하면 `pxr` 경로가 초기화되지 않아 USD 검사에서 실패한다. standalone USD를 런타임에 섞어 해결하지 않는다. USD 정적 검사는 README의 로컬 검사 환경에서 수행하고, Isaac의 USD·물리는 약관 동의 후 AppLauncher로 검증한다. `artifacts/cloud/remote-validation.log`에 해당 실패를 보존했다.

준비 실행에서 관찰한 시간과 카탈로그 단가로 계산한 컴퓨팅 추정액은 약 0.44 USD였다. 후속 물리 실행의 시작·종료 상태와 추정 비용은 `artifacts/cloud/run-physics-20261025.json`에 이어서 기록한다. 공급자의 청구 시작 시각·스토리지·타 프로젝트 비용은 확인하지 않았으므로 장부의 추정액은 실제 청구액이나 계정 잔액을 뜻하지 않는다.

## 검증 순서

로컬에서 준비한 소스와 `assets/cache`, `assets/generated`를 같은 상대 경로로 옮긴다. 별도 머신에서 새로 받는 경우 자산 다운로드·생성·감사를 먼저 실행한다. 실행 전 `configs/runtime.json`과 에셋 SHA를 대조한다.

```bash
cd "$HOME/rby1-assembly-lab"
source .venv-isaac/bin/activate
python scripts/preflight.py
OMNI_KIT_ACCEPT_EULA=YES python scripts/run_sim.py \
  --headless --num-envs 1 \
  --trace artifacts/physics-trace.jsonl \
  --report artifacts/physics-smoke.json
OMNI_KIT_ACCEPT_EULA=YES python scripts/run_sim.py \
  --headless --enable_cameras --num-envs 1 --require-completion \
  --background --camera-view assembly --video-fps 120 \
  --trace artifacts/task-trace.jsonl \
  --video artifacts/task-smoke.mp4 \
  --report artifacts/task-smoke.json
```

`--steps`를 생략하면 `configs/task.json`의 60초 제한을 제어 주기 1/30초로 나눈 1,800단계를 사용한다. 첫 실행은 장면 로딩·물리 step과 trace 기록을 확인한다. 두 번째 실행은 조립 성공을 요구하고 실제 렌더 프레임을 MP4로 저장한다. 영상은 단일 환경과 `--enable_cameras`가 필요하다. `--camera-view overview`는 960×720 전체 구도이고, `--camera-view assembly`는 1280×960 근접 구도다. `--background`의 Simple Room과 카메라는 시각화 전용이며 물리 판정이나 제어 관측을 바꾸지 않는다. 실패하면 기준 제어와 접촉 설정을 고치고 재검증한다. 스모크 결과의 `completed=false`를 성공으로 보고하지 않으며 실패 trace와 영상도 보존한다.

`--video-fps`는 30·60·120 중에서 선택하며 기본값은 120이다. 영상 실행에서만 렌더 간격을 물리 4·2·1단계로 바꾼다. 물리 시간 간격 1/120초와 제어 decimation 4는 유지한다. viewer RGB render product를 만든 뒤 `/isaaclab/render/rtx_sensors` 설정으로 물리 루프의 렌더 필요 조건을 등록하고 종료 시 이전 값을 복원한다. 이는 viewer 촬영 예약이며 정책용 카메라 센서를 추가하지 않는다.

준비 렌더링 16회는 물리를 진행하지 않는다. 기록은 첫 렌더 간격 뒤부터 시작하고, 종료 판정 전 물리 단계까지 포함한다. 같은 물리 counter에서 일어나는 자동 reset 렌더는 기록하지 않는다. 각 제어 단계의 예상 프레임 수나 물리 counter 간격이 맞지 않으면 실행을 실패 처리한다. 60 fps 공유본은 실제 렌더 프레임을 일정한 시간 간격으로 선택하며, 마지막 정지 화면 이외의 프레임 복제나 동작 보간은 사용하지 않는다.

삽입 접촉은 두 값으로 나눠 기록한다. `insertion_contact_force_n`은 플러그와 소켓 사이 월드 좌표계 법선 합력 벡터의 크기이고, `insertion_contact_load_n`은 같은 접촉의 개별 법선력 절댓값을 더한 하중이다. 잼·과부하 판정에는 후자를 사용한다. 금지 접촉 하중은 로봇 29개 부위별로 허용되지 않은 대상과의 접촉 쌍에서 개별 법선력 절댓값을 합산한다. 접촉 버퍼 포화, 음수 개수, 잘못된 범위, 활성 힘의 NaN은 유효한 0 N으로 처리하지 않고 실패로 종료한다. 기존 net force 기반 값은 접촉 위치 진단용 trace일 뿐 안전 판정값이 아니다.

실행 데이터를 기록할 때는 하나의 환경에서 episode마다 seed를 바꿔 성공·실패 모두 보존한다. 기록기는 자동 reset 전 상태를 저장하고, 원시 물체 자세로 삽입 오차를 다시 계산해 판정을 대조한다.

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/record_baseline.py \
  --headless --episodes 3 --output artifacts/baseline_rollouts.h5
python scripts/inspect_dataset.py artifacts/baseline_rollouts.h5
```

기록기도 `--steps`를 생략하면 과제 제한 시간과 제어 주기에서 에피소드당 최대 단계를 계산한다. 현재 설정에서는 1,800단계다. `inspect_dataset.py`가 계약·shape·속성·source hash·독립 평가 종단값·결정적 분할을 모두 통과한 train 성공 episode만 학습 대상으로 고른다.

학습용 성공 데이터는 실제 시뮬레이터 실행 출처·평가 유효성·성공 판정·분할 검사를 통과해야 한다. 공개 AutoMate 자료와 테스트용 합성 episode는 이 조건을 만족하지 않는다.

과제 스모크가 통과한 뒤 학습을 시작한다. 먼저 작은 배치로 한 번의 업데이트를 검증한 다음 규모를 늘린다.

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/train.py --headless --num-envs 4 --max-iterations 1
OMNI_KIT_ACCEPT_EULA=YES python scripts/train.py --headless --num-envs 64 --max-iterations 500
```

체크포인트 선택과 평가 조건은 실험 전에 고정한다. `scripts/evaluate_policy.py`는 체크포인트 경로와 별도 평가 seed를 받는다. 현재 확인한 1-iteration smoke를 학습 완료나 성공률로 기록하지 않는다. 잼 난이도를 고정한 뒤 여러 학습 seed로 규모를 늘리고, 학습에 쓰지 않은 조건에서 zero-residual 기준 제어와 비교해야 한다.

## 명령 오류로 실패 재현

실패 촬영은 학습 정책 평가와 분리한 단일 환경 음성 시험이다. `configs/failure-scenarios.json`에 조건을 고정하고 `--command-fault`로 선택한다. 파지한 오른손을 여는 `right_gripper_open`과, 공중에서 양손에 같은 목표를 주는 `hand_collision`이 공개 영상의 두 사례다. `insertion_lateral_disturbance`는 삽입 정체를 조사한 진단 옵션이며 대표 실패 영상에는 쓰지 않는다.

```bash
cd "$HOME/rby1-assembly-lab"
OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python scripts/run_sim.py \
  --headless --enable_cameras --num-envs 1 --background \
  --camera-view assembly --video-fps 120 --command-fault hand_collision \
  --trace artifacts/failure-hands.jsonl --video artifacts/failure-hands.mp4 \
  --report artifacts/failure-hands.json
python3 scripts/export_failure_media.py \
  --report artifacts/failure-hands.json --trace artifacts/failure-hands.jsonl \
  --video artifacts/failure-hands.mp4 \
  --output artifacts/failure-hands-60fps.mp4 \
  --metadata artifacts/failure-hands-60fps.json
```

원격 촬영은 Isaac 환경을 사용하고, 마지막 편집 명령은 FFmpeg·OpenCV·Pillow와 Noto CJK 글꼴이 있는 로컬 편집 환경에서 실행할 수 있다. 원본·보고서·trace를 같은 경로로 회수해 해시를 확인한다.

명령 오류 시험의 trace는 제어 단계마다 기록한다. `--trace-every`는 일반 실행의 기록 간격을 바꾸며, 명령 오류 시험에서는 1만 허용한다. 촬영은 물리 120 Hz의 실제 프레임을 사용하고, 편집본은 0.5배속·60 fps로 재생한다. 마지막 정지 화면 외에 동작 프레임을 복제하거나 보간하지 않는다. 측정값은 표시 프레임 시각보다 늦지 않은 최신 관측값을 유지한다.

양손 충돌 전용 환경은 runtime 복제본에서 자체 충돌을 켜고 66개 강체 중 좌우 그리퍼 9쌍만 충돌을 허용한다. 나머지 내부 2,136쌍은 필터링하며 외부 접촉 관계와 원본 USD는 유지한다. 왼쪽 손바닥·두 손가락에서 오른손 3개 강체로의 접촉 법선 하중을 합산하고, 외부 금지 접촉과 구분한다. 기존 1 N 금지 접촉 모니터에는 두 하중의 최댓값을 제공한다. 명령 전 접촉 0 N, 양손 간격 감소, 실제 양손 하중 1 N 초과, 외부 금지 접촉 1 N 이하와 `forbidden_contact` 종료를 채택 조건으로 삼는다. 관절 속도의 비유한 값과 잘못된 접촉 버퍼는 실행을 실패 처리한다.

파지 해제 시험은 관절 목표만 OPEN으로 바꾸며 물체·센서·판정값을 변경하지 않는다. 관측된 낙하량과 연속 파지 소실 시간, 기존 모니터의 `grasp_lost` 판정을 함께 기록한다. 각 영상의 설정·실측·소스 해시 정본은 [실패 근거](evidence/failure-cases.json)와 편집본 옆 JSON이다.

## 회수와 종료

결과에는 소스·설정·에셋·체크포인트 해시, GPU와 패키지 버전, seed, 물리 시간 간격을 포함한다. 로그·영상·데이터를 로컬로 회수해 해시를 대조한다. 이 작업에서 시작한 실행과 서버는 승인된 비용 한도·종료 조건에 맞춰 종료하고 `brev ls --json`으로 중지 상태를 확인한다. 이미 실행 중인 다른 프로젝트와 공유하는 서버는 별도 조율 없이 중지하지 않는다.
