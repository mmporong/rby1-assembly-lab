# Generation v2 production 파지 bank 결과

source commit `8572ffd39f1f1959ba0e99e69b2a34791eab0b4e`에서 generation v2 production 후보 20개를 평가했다. 명목 실행 20회가 모두 반환 코드 0과 `physical_valid` 상태로 끝났고, 이 가운데 12개를 전역 파지 bank로 선정했다.

| 구분 | 선정 seed | 개수 |
| --- | --- | ---: |
| train | 20261211~20261218 | 8 |
| dev | 20261231, 20261232 | 2 |
| final | 20261241, 20261243 | 2 |

선정된 12개 사이의 66쌍은 모두 병진 거리 기준 0.25 mm를 통과했다. 가장 가까운 쌍의 병진 거리는 `0.2714145600635222 mm`였다. seed `20261242`는 기존 파지와 병진·회전 조건을 함께 만족하지 못해 `duplicate_object_to_ee_geometry`로 제외됐다. 나머지 미선정 후보는 각 split의 정원이 이미 찬 뒤 평가된 후보다. 정책 성능과 관절 차이는 선정 기준으로 사용하지 않았다.

## 원시 근거

| 파일 | 내용 |
| --- | --- |
| [summary.json](summary.json) | 20회 실행 완료, bank 생성 상태와 입력 불변성 |
| [selection-audit.json](selection-audit.json) | 전체 후보의 선정 결정과 앞서 선정된 파지까지의 거리 |
| [accepted-candidate-plan.json](accepted-candidate-plan.json) | 최종 12개 seed, split, grasp ID |
| [held-grasp-bank.json](held-grasp-bank.json) | train 8개, dev 2개, final 2개의 파지 descriptor와 artifact SHA |
| [retrieval-verification.json](retrieval-verification.json) | archive와 payload 215개 SHA 검증 결과 |
| [source-manifest.json](source-manifest.json) | 공개 파일의 원본 상대경로, 크기, SHA-256 |

위 다섯 raw JSON은 검증한 회수본과 바이트가 같다. trace, held-state 개별 파일, 로그, CAD, 에셋과 소스 스냅샷은 이 작은 공개 묶음에 복사하지 않았다.

## 후속 reset 검증 상태

파지 생성과 bank 선정은 완료됐다. 이어서 실행한 최초 N256 reset 검증은 quaternion 정규화 검사 오류 때문에 실패했다. checker 수정과 reset 재검증을 마친 뒤 교사 시연과 BC 학습까지 완료했으며, 그 결과는 [교사 데이터와 BC 초기화 결과](../teacher-bc-results/README.md)에 기록했다. 이 폴더의 production raw JSON은 당시 파지 생성·선정 기록으로 보존한다.

