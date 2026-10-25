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

Isaac Sim의 라이선스 동의는 설치 환경의 유효한 동의 상태를 따른다. 스크립트가 동의 값을 자동으로 설정하지 않는다. Isaac Sim 런타임에 포함된 USD를 사용하며 이 환경에 standalone `usd-core`를 추가로 설치하지 않는다.

근거: [Isaac Sim Python 설치](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html), [Isaac Lab 2.3 설치](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html). Lab의 RSL-RL 버전은 고정 source revision의 `source/isaaclab_rl/setup.py`를 기준으로 한다.

## 확인한 실행 범위

2026-10-25 전용 L40S에서 패키지 사전 검사, `pip check`, CUDA 텐서 연산, 제어 메서드의 CPU 텐서 검사 8개가 통과했다. 로컬 Python 3.11 + USD 검사 환경에서는 테스트 51개가 통과했다. 새 서버의 Isaac Sim import는 EULA 동의 질문에서 종료했으며, 장면 로딩·PhysX step·파지·삽입·학습은 아직 실행하지 않았다.

원격 Isaac 가상환경에서 앱을 시작하지 않고 전체 단위 테스트를 실행하면 `pxr` 경로가 초기화되지 않아 USD 검사에서 실패한다. standalone USD를 런타임에 섞어 해결하지 않는다. USD 정적 검사는 README의 로컬 검사 환경에서 수행하고, Isaac의 USD·물리는 약관 동의 후 AppLauncher로 검증한다. `artifacts/cloud/remote-validation.log`에 해당 실패를 보존했다.

준비 실행 종료 후 전용 인스턴스 `d1gwdve58`의 `STOPPED`를 확인하고 로컬 중지 타이머도 해제했다. 관찰한 준비 시간과 카탈로그 단가로 계산한 컴퓨팅 추정액은 약 0.44 USD다. 공급자의 청구 시작 시각·스토리지·타 프로젝트 비용은 확인하지 않았으므로 실제 청구액이나 계정 잔액을 뜻하지 않는다. 재개할 때는 중지된 서버의 소스 스냅샷과 최신 저장소를 먼저 대조하고, 이 장부를 이어서 비용을 기록한다.

## 검증 순서

로컬에서 준비한 소스와 `assets/cache`, `assets/generated`를 같은 상대 경로로 옮긴다. 별도 머신에서 새로 받는 경우 자산 다운로드·생성·감사를 먼저 실행한다. 실행 전 `configs/runtime.json`과 에셋 SHA를 대조한다.

```bash
cd "$HOME/rby1-assembly-lab"
source .venv-isaac/bin/activate
python scripts/preflight.py
python scripts/run_sim.py --headless --num-envs 1 --steps 900 --report artifacts/physics-smoke.json
python scripts/run_sim.py --headless --num-envs 1 --steps 900 --require-completion --report artifacts/task-smoke.json
```

첫 실행은 장면 로딩·물리 step 확인이다. 두 번째 실행은 조립 성공을 요구한다. 실패하면 기준 제어와 접촉 설정을 고치고 재검증한다. 스모크 결과의 `completed=false`를 성공으로 보고하지 않는다. `artifacts`에는 실패 기록도 남긴다.

실행 데이터를 기록할 때는 하나의 환경에서 episode마다 seed를 바꿔 성공·실패 모두 보존한다. 기록기는 자동 reset 전 상태를 저장하고, 원시 물체 자세로 삽입 오차를 다시 계산해 판정을 대조한다.

```bash
python scripts/record_baseline.py --headless --episodes 3 --steps 900 --output artifacts/baseline_rollouts.h5
python scripts/inspect_dataset.py artifacts/baseline_rollouts.h5
```

학습용 성공 데이터는 실제 시뮬레이터 실행 출처·평가 유효성·성공 판정·분할 검사를 통과해야 한다. 공개 AutoMate 자료와 테스트용 합성 episode는 이 조건을 만족하지 않는다.

과제 스모크가 통과한 뒤 학습을 시작한다. 먼저 작은 배치로 한 번의 업데이트를 검증한 다음 규모를 늘린다.

```bash
python scripts/train.py --headless --num-envs 4 --max-iterations 1
python scripts/train.py --headless --num-envs 64 --max-iterations 500
```

체크포인트 선택과 평가 조건은 실험 전에 고정한다. `scripts/evaluate_policy.py`는 체크포인트 경로와 별도 평가 seed를 받는다. 실제 실행 결과가 생기기 전에는 학습 완료나 성공률을 기록하지 않는다.

## 회수와 종료

결과에는 소스·설정·에셋·체크포인트 해시, GPU와 패키지 버전, seed, 물리 시간 간격을 포함한다. 로그·영상·데이터를 로컬로 회수해 해시를 대조한다. 이 작업에서 시작한 실행과 서버는 승인된 비용 한도·종료 조건에 맞춰 종료하고 `brev ls --json`으로 중지 상태를 확인한다. 이미 실행 중인 다른 프로젝트와 공유하는 서버는 별도 조율 없이 중지하지 않는다.
