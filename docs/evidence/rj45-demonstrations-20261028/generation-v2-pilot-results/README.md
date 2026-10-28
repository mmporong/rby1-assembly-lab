# Generation v2 물리 파지 pilot 결과

source commit `8572ffd39f1f1959ba0e99e69b2a34791eab0b4e`에서 generation v2의 명목 실행과 pilot 4회를 마쳤다. 네 seed가 모두 시뮬레이터 물리 파지에 성공했고, 여섯 쌍이 사전에 정한 거리 조건을 통과해 최대 clique는 4였다. 요구값은 3이었다.

| 측정 항목 | 결과 |
| --- | ---: |
| 물리 파지 성공 | 4/4 |
| 비교한 파지 쌍 | 6 |
| 분리 조건을 통과한 쌍 | 6/6 |
| 쌍 사이 최소 병진 거리 | 약 0.362 mm |
| 최대 clique | 4 |
| pilot gate | 통과 |

분리 조건은 손목 기준 물체 자세의 병진 거리가 0.25 mm 이상이거나 회전 거리가 0.004 rad 이상인 경우다. 정책 성능으로 파지를 고르지 않았고, 실패한 시도를 결과에서 빼는 방식도 사용하지 않았다.

## 계약과 측정 결과

[generation v2 계약](../grasp-generation-v2-protocol.json)은 seed, catalog row, 거리 기준과 gate를 실행 전에 고정한 문서다. 그 문서의 `status`는 GPU 실행 전 검토 상태를 보존하며 pilot 결과를 뜻하지 않는다. 측정 결과는 이 폴더의 `summary.json`, `pilot-audit.json`, `attempts.json`에 분리해 보관했다.

| 파일 | 내용 |
| --- | --- |
| [summary.json](summary.json) | source commit, 명목 실행과 pilot 반환 코드, 입력 불변성과 정리 상태 |
| [pilot-audit.json](pilot-audit.json) | 여섯 쌍의 병진·회전·관절 거리와 clique 판정 |
| [attempts.json](attempts.json) | 네 seed의 catalog 배정, 성공 상태와 held artifact SHA |
| [retrieval-verification.json](retrieval-verification.json) | 회수 archive와 payload 89개 검증 결과 |
| [source-manifest.json](source-manifest.json) | 공개 파일의 상대경로·크기·SHA와 원본 archive SHA |

네 raw JSON은 검증한 회수본과 바이트가 같다. 큰 trace, held-state 원본, 로그, CAD와 에셋은 이 공개 묶음에 복사하지 않았다. 원본 archive SHA-256은 `6e566795fbaa22523330c7a728aebc67b543aecc8106ed0175d8f5039ecdc96a`이며, payload manifest SHA-256은 `9203606c6e1290efeafb4147f84f1efb16cac63e10e8443d25ed5c178417729f`다.

이 결과는 generation v2가 서로 구분되는 시뮬레이터 물리 파지를 만들 수 있는지 확인한 pilot이다. production 파지, teacher 시연, BC·PPO 성능, 실제 센서와 실물 전이는 이 결과의 검증 범위가 아니다.
