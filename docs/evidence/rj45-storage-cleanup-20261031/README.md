# 이전 실험 자료 보관 정리

종료된 개발·실패·회수 실험 8개 묶음, 888개 파일을 보관 폴더로 옮겼다. 합계는 1,473,407,219바이트(약 1.47 GB)다. 원래 경로에는 상대 디렉터리 symlink를 남겨 문서·manifest·재현 명령이 같은 자료를 계속 읽는다.

파일 삭제나 중복 내용 병합은 수행하지 않았다. 이동 파일 888개와 관련 대조 파일 8개의 SHA-256·device/inode·mode·mtime·uid/gid를 대조했다. 바이트와 원본 메타데이터가 유지됐고, 실제 회수한 디스크 공간은 0바이트다. 학습 데이터·체크포인트·실패 기록과 최신 진단 자료를 보존했다.

## 위치

보관 폴더는 `$HOME/rby1-assembly-lab/artifacts/archive/20261031/cloud/`다. 아래 이름은 모두 기존 `$HOME/rby1-assembly-lab/artifacts/cloud/`에서도 열 수 있다.

| 묶음 | 파일 수 | 논리 크기 |
| --- | ---: | ---: |
| `ppo-smoke` | 5 | 3,028,851 B |
| `previous-early-failures` | 3 | 983,655 B |
| `rj45-paired-audit-sample-20261028` | 69 | 37,450,252 B |
| `rj45-paired-failed-snapshot-20261028` | 214 | 154,084,311 B |
| `retrieved-rj45-parallel-media-20261028` | 230 | 165,230,795 B |
| `retrieved-4c-generation-v2-20261028` | 94 | 174,008,222 B |
| `retrieved-4c-recovery-20261028` | 51 | 69,603,510 B |
| `retrieved-4c-bank-generation-v2-20261028` | 222 | 869,017,623 B |

이전 smoke와 실패 실행은 개발 기록이다. bank·회수 자료는 과거 실험의 재현 근거이므로 보관 상태에서도 유지한다. `.cache/upstream/IsaacLab`은 설치·출처 문서가 참조하는 clone이라 이동하지 않았다.

## 검증과 복구

실행 전 파일별 ledger를 고정하고, 각 하위 트리의 전체 상대경로·유형 집합을 이동 전후 비교했다. 같은 파일시스템 안에서 디렉터리 이름만 바꾸고 기존 경로의 링크 목적지를 검증했다. 정리 코드와 계획은 독립 리뷰에서 승인받은 뒤 실행했다. 실행 후 별도 검증 결과는 [독립 검증 기록](independent-verification.json)에 남긴다.

- [정리 요약과 경로 대응](summary.json)
- [실행 receipt](receipt.json), [단계별 기록](progress.jsonl)
- [이동 전 전체 ledger](ledger-before.json.gz): 원본 SHA-256은 summary에 기록했다.
- [실행한 일회성 코드 사본](apply_cleanup.py): 보존용이며 이 문서 폴더에서 재실행하는 용도가 아니다.

로컬 실행 기록은 `$HOME/rby1-assembly-lab/artifacts/maintenance/cleanup-20261031/`에 있다. 복구할 때는 각 기존 경로가 summary의 보관 위치를 가리키는 symlink인지 먼저 확인하고, 그 링크를 제거한 뒤 보관 디렉터리를 원래 위치로 rename한다. 원본 파일의 inode와 메타데이터는 유지돼 별도 복원 계산이 필요하지 않다. 전체 프로젝트를 복사할 때는 보관 폴더와 기존 경로의 상대 symlink를 함께 보존한다.
