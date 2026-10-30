# G2 학습률 진단

고정한 한 개 train 궤적에서 기존 학습률 `1e-3`(E0)과 10배 낮춘 `1e-4`(E1)를 비교했다. 두 실행은 Python 3.11.15, PyTorch 2.7.0+cpu, NumPy 1.26.0에서 동일한 입력 395행, seed `20261030`, 2,000 full-batch Adam step을 사용했다. 초기 actor state SHA-256도 두 실행에서 `cef1e9bf062d4d63516cdc6a391dce6de37f5bd9e1a2125e22b3ae1611be3c5f`로 같았다.

| 실행 | 학습률 | 최종 정규화 행동 MSE | 상수 평균 대비 | 마지막 5회 단조 감소 |
| --- | ---: | ---: | ---: | --- |
| E0 | `1e-3` | `2.320454e-6` | `98.56%` | 아니요 |
| E1 | `1e-4` | `2.336099e-6` | `99.23%` | 예 |

사전에 고정한 진단 기준은 E1이 상수 평균 MSE의 90% 이하이고, E0보다 낮으며, 마지막 다섯 기록이 계속 감소하는 것이었다. E1은 마지막 조건만 만족했다. 낮은 학습률은 말미의 진동을 줄였지만 2,000 step 안에서 E0보다 낮은 오차를 만들지 못했고, 상수 평균 예측기와도 충분히 벌어지지 않았다. 따라서 이번 진단은 `FAIL`이다.

이 결과는 같은 궤적의 오프라인 회귀 진단이다. 상태에 따른 수정 행동, 물리 폐루프 성공, 강화학습 기여, 보지 않은 조건의 일반화를 입증하지 않는다. 사전 중단 규칙에 따라 20,000 step이나 라벨 정규화 실험은 이어서 실행하지 않았다. 이전 실험의 기준과 결과도 변경하지 않았다.

## 재현

프로토콜은 학습 전에 [protocol.json](protocol.json)으로 고정했다. 실행 결과와 모든 체크포인트 수치는 [result.json](result.json)에 있다. 기존 출력 경로를 덮어쓰지 않으므로 재실행할 때는 새 출력 및 결과 경로를 사용해야 한다.

```bash
cd "$HOME/rby1-assembly-lab"

PYTHONPATH=src .venv/bin/python scripts/run_rj45_g2_optimizer_probe.py freeze \
  --manifest "$HOME/rby1-assembly-lab-rj45-data/artifacts/local-g2-overfit-20261030/mean-baseline-v2/export/manifest.json" \
  --protocol docs/evidence/rj45-g2-optimizer-probe-20261030/protocol-recheck.json

PYTHONPATH=src .venv/bin/python scripts/run_rj45_g2_optimizer_probe.py run \
  --manifest "$HOME/rby1-assembly-lab-rj45-data/artifacts/local-g2-overfit-20261030/mean-baseline-v2/export/manifest.json" \
  --protocol docs/evidence/rj45-g2-optimizer-probe-20261030/protocol-recheck.json \
  --output-dir artifacts/cloud/rj45-resume-20261030/g2-optimizer-probe-recheck \
  --result docs/evidence/rj45-g2-optimizer-probe-20261030/result-recheck.json
```
