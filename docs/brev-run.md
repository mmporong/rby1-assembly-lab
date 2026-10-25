# Brev 실행 절차

## 실행 범위

기존 작업과 겹치지 않는 `$HOME/rby1-assembly-lab` workspace와 별도 Python 환경을 사용한다. GPU 비용 한도가 정해진 뒤에만 인스턴스를 켠다. 실행 전 다른 GPU 프로세스를 조회하며 다른 프로젝트의 프로세스·데이터·가상환경을 변경하지 않는다.

서버가 켜진 뒤 GPU 모델·드라이버·디스크·설치 환경을 먼저 확인한다. 현재 설치 환경을 읽기 전용으로 기록하는 명령은 다음과 같다.

```bash
cd "$HOME/rby1-assembly-lab"
python3 scripts/preflight.py --output artifacts/preflight-before.json
```

미설치 환경에서는 exit code 2가 정상적인 진단 결과다. 이 검사는 Isaac Sim을 시작하지 않는다. GPU 조회와 패키지 확인 통과만으로 PhysX 실행을 인증하지 않는다.

## 별도 환경 설치

아래는 Linux x86_64용 고정 버전 설치 절차다. 원격에 호환 환경이 있으면 먼저 검증하고 사용하며, 기존 환경에 패키지를 덮어쓰지 않는다. 다운로드 전에 남은 디스크 용량을 확인한다.

```bash
cd "$HOME/rby1-assembly-lab"
python3.11 -m venv .venv-isaac
source .venv-isaac/bin/activate
python -m pip install --upgrade pip
python -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
python -m pip install 'isaaclab[all]==2.3.0' --extra-index-url https://pypi.nvidia.com
python -m pip install 'torch==2.7.0' 'torchvision==0.22.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'rsl-rl-lib==3.0.1' -e .
python -m pip check
```

Isaac Sim의 라이선스 동의는 설치 환경의 유효한 동의 상태를 따른다. 스크립트가 동의 값을 자동으로 설정하지 않는다. Isaac Sim 런타임에 포함된 USD를 사용하며 이 환경에 standalone `usd-core`를 추가로 설치하지 않는다.

근거: [Isaac Sim Python 설치](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html), [Isaac Lab 2.3 설치](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/isaaclab_pip_installation.html). Lab의 RSL-RL 버전은 고정 source revision의 `source/isaaclab_rl/setup.py`를 기준으로 한다.

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
