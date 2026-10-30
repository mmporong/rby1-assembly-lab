# RJ45 정책 진단 미디어

원본 60fps 물리 렌더와 같은 시각의 평가 측정값을 한 화면에서 확인하는 자료다. 성공 기준 1건과 BC 정책의 실제 타임아웃 2건을 사용했다. 실패를 크게 보이게 만들거나 새로운 충돌을 넣지 않았으며, 원본 프레임을 보간·복제하거나 마지막 장면을 늘리지 않았다.

## 영상에서 확인할 내용

- `v3-biased-zero-env54-diagnostic-60fps.mp4`: 기준 제어기는 횡오차와 상대 속도 기준을 만족한 상태를 0.4초 유지해 성공했다.
- `v5-bc-env54-diagnostic-60fps.mp4`: 마지막 형상은 기준 안에 들어왔지만 상대 선속도 22.69 mm/s, 상대 각속도 0.907 rad/s로 움직임이 남았다. 안정 유지 시간은 0초였다.
- `v5-bc-env42-diagnostic-60fps.mp4`: 마지막 횡오차가 91.5 µm로 프로젝트 기준 50 µm를 넘었고, 상대 선속도와 각속도 기준도 넘었다. 안정 유지 시간은 0초였다.

화면 왼쪽은 해당 시각의 물리 렌더다. 오른쪽 안정성 수치는 제어 스텝에 기록된 판정값이다. 한 제어 스텝에서 수집한 9개 물리 샘플이 모두 기준을 만족해야 안정 시간이 누적된다. 접촉 하중은 PhysX 접촉 proxy이며 손목 힘·토크 센서값이 아니다.

두 실패 영상은 동작이 비슷해 보여도 최종 성공 조건을 만족하지 못한 사례다. 여기서 확인되는 것은 평가 기준의 실패까지다. 특정 BC action이 원인이라고 단정하지 않는다. 사용한 정책은 동결된 BC actor `f0fd65b5…33b5`이며, 학습을 마친 PPO 정책 영상이 아니다. 50 µm는 이 프로젝트의 평가 기준으로 사용했다.

## 산출물과 재현

`manifest.json`에는 원본 영상·프레임 JSONL·메타데이터·보고서의 SHA-256, 정책 계보, 선택한 환경, 출력 프레임 수와 인코딩 정보가 들어 있다. `*-timeseries.csv`는 그래프와 패널에 사용한 파생 시계열이다. `*-final-preview.png`는 각 영상의 마지막 원본 프레임과 최종 판정 패널을 함께 보여준다. 원본 대용량 JSONL은 이 폴더에 복제하지 않았다.

```bash
cd "$HOME/rby1-assembly-lab-rj45-data"
python3 scripts/export_rj45_policy_media.py \
  --input-dir "$HOME/rby1-assembly-lab/artifacts/cloud/retrieved-rj45-policy-media-eval-recovery-v1-20261028" \
  --media-dir docs/media/rj45/policy-diagnostic \
  --evidence-dir docs/evidence/rj45-demonstrations-20261028/policy-media
```

선택은 촬영 뒤 장면을 보고 고른 것이 아니다. 고정 평가 결과에서 env54는 기준 제어기 성공·BC 실패가 짝을 이룬 사례로, env42는 BC 실패군의 대표 사례로 촬영 전에 정했다. 세부 입력과 결과는 `manifest.json`에서 대조할 수 있다.
