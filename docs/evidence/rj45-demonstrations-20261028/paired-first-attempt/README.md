# 첫 다중 파지 PPO·BC→PPO 비교의 중단 기록

이 실행은 BC→PPO 학습 중 파지 초기화 검사에서 중단됐다. 완료된 PPO-only 측정은 보존하지만, 두 학습 방법의 비교를 마친 결과로 사용하지 않는다. 소스는 `8512a5932547b5495325b5ff0b6894fb072db98e`이며 최종 평가 조건은 열지 않았다.

## 완료된 PPO-only 측정

각 실행은 256환경 × 32 step × 200 update를 마쳤다. 같은 개발 파지 2개와 보정 조건 64개를 사용했으며, 기준 제어는 38/64였다.

| 학습 seed | 20 update | 96 update | 200 update |
| --- | ---: | ---: | ---: |
| 20261131 | 30/64 | 35/64 | 32/64 |
| 20261132 | 8/64 | 33/64 | 27/64 |
| 20261133 | 36/64 | 36/64 | 31/64 |

200-update 평균은 30/64로 기준 제어보다 낮았다. 이 수치는 개발 조건에 한정되며 다른 파지나 실물 성능을 뜻하지 않는다. 9개 평가 모두 요청한 사례를 분모에 포함했고 무효 사례와 금지 접촉 하중은 0이었다. 각 평가의 checkpoint 바이트, payload iteration, 학습 완료 manifest, provenance, runner config와 36개 소스 해시를 대조했다.

## 중단된 BC→PPO

세 BC→PPO 실행은 200 update를 마치지 못했다. seed `20261132`에서 iteration 64 이후 환경 204의 초기화가 30 step 안에 안정화 기준을 통과하지 못했고, 검사기가 오류를 발생시켜 캠페인이 중단됐다. 다른 두 BC 실행도 중단 절차에 따라 종료됐다. [`campaign-summary.json`](campaign-summary.json)의 상태는 `failed`다.

실패 당시 플러그 각속도는 약 `0.06147 rad/s`로 기준 `0.05 rad/s`를 넘었다. 파지 접촉, 리프트, 유한성, 관절 속도와 추종 오차는 통과했으나 motion-settled 조건이 거짓이었다. 기록에는 그 이전 30 step의 속도 추세와 파지 ID가 없어 반복 가능한 특정 파지 문제인지, 접촉 재형성 과정의 변동인지는 확정하지 못했다. 전체 진단과 실패 로그 해시는 [`partial-audit.json`](partial-audit.json)에 있다.

후속 복구는 같은 파지·에피소드·보정 오차를 유지한 초기화 재시도를 한 번 허용하고, 두 번째 실패에는 중단하는 별도 계약으로 검증한다. 안정화 임계값과 과제 성공 조건은 유지한다. 소스와 초기화 계약이 바뀌므로 두 학습 방법 모두 새로 실행하며, 이 폴더의 PPO 측정을 후속 비교의 대조군으로 이어 붙이지 않는다.

## 원본과 재검증 범위

원격 산출물 211개를 archive 생성 전후와 로컬 회수 후에 해시로 대조했다. 공개 폴더에는 최종 PPO 평가 3개와 해당 inspection, 기준 제어·BC-only 평가, 실패 summary, 부분 감사와 회수 검증을 포함한다. checkpoint 바이트와 전체 로그는 Git에 넣지 않고 검증된 회수본에 보존했다. 20·96-update 평가의 검증 결과도 부분 감사에 남겼다.

- [`source-manifest.json`](source-manifest.json): 공개 파일의 크기·SHA와 회수본 상대경로
- [`partial-audit.json`](partial-audit.json): 9개 PPO 평가·checkpoint의 검증과 warmup 실패 진단
- [`retrieval-verification.json`](retrieval-verification.json): archive와 211개 파일의 회수 검증

이 부분 감사에서는 승격 조건을 계산하지 않았다. BC→PPO 3개 시드의 최종 평가가 없기 때문이다.

부분 감사는 대화형 Python 실행에서 `summarize_rj45_bc_ppo_campaign.py`의 검증 함수를 가져와 생성했다. JSON의 `audit_helper_source_sha256`은 그 함수 모듈의 해시이며, 전체 캠페인 집계 CLI를 실행했다는 뜻은 아니다. 해당 CLI는 완료되지 않은 캠페인을 거부한다. 생성 과정에서는 frozen plan과 prelaunch를 대조하고 `verify_source_map`, `validate_bank_manifest`, `validate_report`, `validate_checkpoint_bundle`을 사용해 기준 제어·BC 평가와 PPO 3개 seed × 3개 checkpoint를 검사했다. 실패 로그의 `HeldGraspWarmupTimeoutError` JSON은 별도로 추출했다. 원래 대화형 실행문은 독립 파일로 보존하지 않았으므로 이 폴더만으로 부분 감사 생성 명령 전체를 재실행할 수는 없다. 각 입력과 검사 결과의 해시는 공개 JSON과 회수본에 남아 있다.
