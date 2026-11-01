# RJ45 접촉 계측 진단 — 2026-11-01d

소스 커밋 `5162f3007a63bbcf6f173fcd7f0172e2a711f561`으로 train-02의 signed-six zero와 tz± oracle 참조를 실행했다. 이 8회 진단은 정식 task admission 또는 RL 효과 평가가 아니다.

## 결과

| 방법 | 조건 | 성공 | 최종 횡방향 오차 (µm) |
| --- | --- | --- | --- |
| zero | tx+ | 실패 | 88.05 |
| zero | tx− | 성공 | 48.12 |
| zero | tz+ | 실패 | 118.81 |
| zero | tz− | 실패 | 120.10 |
| zero | rx+ | 성공 | 16.30 |
| zero | rx− | 성공 | 19.46 |
| oracle GT reference | tz+ | 성공 | 5.93 |
| oracle GT reference | tz− | 성공 | 3.13 |

8회 모두 hard/forbidden contact가 없었다. tz± zero의 기록된 INSERT 제어 틱에서 삽입 net force와 normal-load sum은 모두 0이었다. 합력이 상쇄된 것으로 설명할 수는 없지만, 필터 범위와 제어 틱 사이의 모든 물리 접촉까지 없었다는 뜻은 아니다.

전체 좌표계로 재계산한 기하는 보고된 기하와 4,210행 모두 허용 오차 안에서 일치했다. 과거 y축 오프셋만 사용한 계산은 4,202행에서 불일치했다. 성공 기준과 물리 런타임은 바꾸지 않았다.

## 검증과 원본

- `completed-summary.json`: 8회 terminal, 단계별 힘, 최종 기하, trace SHA.
- `independent-completed-review.json`: 별도 verifier가 원시 4,210행의 해시 체인, report/sidecar, terminal, 기하 및 요약을 재계산한 결과.
- `local-recovery-verification.json`: 아카이브 101 regular members, payload 100개의 크기와 SHA 검증.
- 원본 아카이브 SHA-256: `d120d653568cbb72cda713da1f80fd8807a219aec233fec68a2ae69562764415`.
- 로컬 원본: `/opt/homedata/rby1-storage/recovered/rj45-contact-diagnostic-20261101d/job`.

## 해석과 다음 단계

정답 위치를 쓰면 tz 두 조건을 해결할 수 있으나, zero의 수동 관측만으로 높이 오차를 식별한다는 근거는 없다. 모든 허용 동작에서 식별 불가능하다고 결론내릴 수도 없다. 원래 오차·성공 기준을 유지한 case-blind active-z 탐색을 별도 진단으로 설계한다. 두 개발 파지 × tz± × 같은 탐색 순서의 2회 반복을 사용하고, onset·plateau·return·판정 규칙을 실행 전에 고정한다.

이 결과로 기존 formal90의 관측가능성 gate를 소급 통과시키지 않는다. `task_admission_completed=false`, `rl_pilot_admitted=false`를 유지하며 held-out은 열지 않았다.
