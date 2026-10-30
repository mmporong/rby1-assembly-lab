# 전체 train 시연의 행동 예측 기준선

동결한 교사 train 데이터의 성공·유효 episode 256개, 유효 행동 95,826행에서 기존 BC의 예측 오차를 비교했다. 새 학습 없이 수행한 사후 분석이며 dev·final은 열지 않았다. 아래 MSE는 정규화된 6차원 행동의 오차로, 위치 오차나 삽입 성공률이 아니다.

| 예측 방식 | MSE |
| --- | ---: |
| 항상 0 | `2.079423e-3` |
| 전체 train 평균 행동 | `2.074200e-3` |
| episode별 평균 행동·oracle | `8.867419e-7` |
| 직전 교사 행동 복사 | `6.876696e-6` |
| 동결 BC | `6.715703e-6` |

기존 BC는 전체 평균 행동보다 오차가 작지만 직전 교사 행동 복사 대비 감소 폭은 약 2.34%였다. 관측의 `56:62`가 직전 행의 적용 행동과 일치하는지도 전 유효 행에서 확인했다. episode별 평균은 해당 episode의 전체 정답을 사용하므로 배포 가능한 정책이 아니다.

이 기록에서는 이전 행동도 교사가 실행한 값이다. 학생이 행동한 뒤 받는 관측과 다를 수 있으므로, 이 수치만으로 행동 복사가 실제 실패 원인이라고 확정하지 않는다. 학생 방문 상태의 교사 재질의와 폐루프 비교가 남아 있다. 실제 성공률·안정화·강화학습의 효과는 이 분석으로 입증되지 않는다.

## 재현

검증한 train HDF5와 actor가 필요하며 파일 SHA와 데이터 계보가 다르면 분석이 중단된다. 원본 경로·해시, 축별 오차, 출력 제한 적용 전후 비교, 관측 시간 대조와 분석 소스 해시는 [result.json](result.json)에 보존했다.

```bash
cd "$HOME/rby1-assembly-lab"
PYTHONPATH=src .venv/bin/python scripts/analyze_rj45_teacher_forcing.py \
  --train-dataset "$HOME/rby1-assembly-lab/artifacts/cloud/retrieved-4c-teacher-bc-collector-recovery-20261028/payload/artifacts/rj45-4c-teacher-bc-collector-recovery-v1-20261028/teacher-train.h5" \
  --actor "$HOME/rby1-assembly-lab/artifacts/cloud/retrieved-4c-teacher-bc-collector-recovery-20261028/payload/artifacts/rj45-4c-teacher-bc-collector-recovery-v1-20261028/bc/bc_actor_init.pt" \
  --output artifacts/teacher-forced-baselines-recheck.json
```

출력 파일은 새 경로를 사용한다. 기존 결과를 덮어쓰지 않는다.
