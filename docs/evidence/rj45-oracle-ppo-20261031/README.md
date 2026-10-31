# 고정 train 조건의 oracle PPO 사전 진단

이번 Brev 실행은 PPO 학습에 도달하지 못했다. 기준 제어와 참조 제어기의 물리 trace는 끝까지 기록됐지만, 최종 보고서를 저장하면서 데이터 계약 오류가 발생했다. 두 실행의 보고서는 `invalid`로 보존하며, PPO 성능이나 강화학습의 개선 근거로 사용하지 않는다.

## 확인한 내용

| 항목 | 학습 보정 0인 기준 제어 | 정답 상태를 쓰는 참조 제어기 |
| --- | --- | --- |
| 보고서 상태 | invalid | invalid |
| 오류 발생 전 trace의 종료 판정 | valid_success | valid_success |
| 유효 제어 행 | 387 | 392 |
| 마지막 환경 시각 | 13.266667 s | 13.433334 s |
| 마지막 삽입 깊이 | 12.303611 mm | 12.292417 mm |
| 마지막 횡오차 | 15.247598 µm | 7.711869 µm |
| 참조 보정이 0이 아닌 행 | 해당 없음 | 392 |

표는 저장된 trace의 관측값이다. 최종 보고서와 종료 시점의 전체 입력 해시 검증을 마친 정상 실행 결과와 구분한다. 마지막 환경 시각은 파지 복원·warmup을 포함하는 시뮬레이션 시각이며, 실제 벽시계 실행 시간이나 순수 삽입 시간으로 해석하지 않는다. 단일 조건·단일 episode이므로 성공률 차이나 일반화 성능을 계산하지 않는다. 삽입 접촉 proxy가 두 trace 모두 0 N이므로, 접촉 복구를 시험한 증거로도 사용하지 않는다.

학습 보정을 0으로 둔 실행에도 성공 판정이 기록됐다. 이 고정 조건에서 정책이 성공하더라도 강화학습이 필요했던 과제인지 구분하기 어렵다. 다음 진단은 기준 제어가 어려움을 겪으면서 참조 제어기로 회복 가능한 train 조건을 사전에 정해 수행한다. 알고리즘 비교용 평가 분포 전체를 실패 사례만으로 구성하지 않으며, 명목 조건도 함께 남긴다. 기존 dev/final 분할을 이 단일 조건 학습에 사용하지 않았다.

## 실행 조건과 중단 원인

- L40S 1장, Isaac Sim 5.1.0.0, Isaac Lab 0.47.2, Torch 2.7.0+cu128, RSL-RL 3.0.1을 사용했다.
- frozen runtime은 `104721acf2222c1cfe8ec86cd030f5116a4fef93`의 Python 소스 35개를 바이트 해시로 검증한다. 조건은 train 파지 `rj45-train-02`, base seed `20261120`, calibration seed `66384140`, 환경 1개다.
- 계획은 zero, reference, oracle92, standard79 순서였다. 학습 군마다 32 update·4,096 transition을 예정했으나 두 학습 군 모두 시작하지 않았다.
- 원래 runner가 실제 `TerminalEvaluation`에 없는 `completion_time_s`와 `failure_reasons`를 읽었다. 첫 필드에서 `AttributeError`가 발생했다.
- Isaac 앱의 `close()`가 종료 코드 0을 발생시켜 내부 실패 코드 2를 덮었다. 외부 batch는 reference 보고서의 누락된 `evaluation`을 읽다가 코드 1로 끝났다.
- runner는 `TerminalEvaluation.as_dict()`의 실제 계약으로 저장하고, Python의 `SystemExit(0)`이 내부 실패 반환값을 덮지 않도록 수정했다. 실제 dataclass 직렬화와 가짜 앱 종료를 재현하는 테스트를 추가했다. [후속 실행](../rj45-oracle-condition-20261031/README.md)에서는 Kit 네이티브 종료 경로가 프로세스 코드 0을 내는 경우도 확인했다. 따라서 외부 배치는 종료 코드와 보고서의 `status == completed`를 모두 검사한다. 이 문서의 첫 실행 두 건은 계속 invalid로 보존한다.

## 구현한 진단 도구

`standard79`는 기존 79D actor 입력을 유지하고, `oracle92`는 상대 물체 자세·속도 13개를 추가한다. 두 critic은 같은 92D 정보를 받는다. oracle 정답 관측은 원인 분리용이며 실행 정책의 최종 성능에 합산하지 않는다. 공통 actor 가중치·critic 초기값·평가 RNG를 맞추고, 물리·보상·행동 권한·성공 기준은 archived runtime을 유지한다.

별도 CPU 합성 VecEnv에서 실제 RSL-RL 3.0.1의 두 actor 모드를 각각 2 update 실행했다. 유한한 return·advantage·loss, 가중치 변경, 추론과 checkpoint 저장을 확인했다([SDK 결과](cpu-sdk-result.json)). 이 검사는 SDK 연결 검사이며 Isaac/PhysX 실행이나 RJ45 과제 학습 성능을 뜻하지 않는다.

새 프로토콜의 초안은 저장소 루트에서 다음과 같이 만든다. 이 명령은 GPU를 시작하지 않는다. 실제 실행은 새 UTC 종료 시각과 외부 중지 장치를 포함한 프로토콜이 필요하며, 만료된 이번 프로토콜은 실행용으로 재사용하지 않는다.

```bash
cd "$HOME/rby1-assembly-lab"
.venv/bin/python scripts/run_rj45_oracle_ppo_probe.py \
  --current-root "$PWD" \
  --write-protocol artifacts/oracle-protocol-draft.json
```

## 원본과 재현 경계

- [실행 전 동결 프로토콜](protocol-attempt1.json), [분석 결과와 해시](result.json), [batch 출력](batch.log)
- [zero invalid 보고서](zero-invalid-report.json), [reference invalid 보고서](reference-invalid-report.json)
- [zero 전체 trace](zero-trace.jsonl.gz), [reference 전체 trace](reference-trace.jsonl.gz): 관측·요청/적용 행동·하중·9개 물리 표본 포함
- [실행 당시 runner](attempt1-run_rj45_oracle_ppo_probe.py.gz), [실행 당시 관측 adapter](attempt1-rj45_oracle_diagnostic.py.gz): 현재 수정본과 구분하는 압축 소스

원격 자료는 로컬로 회수했다. 인스턴스 정지 확인과 중지 장치 정리 결과는 [종료 기록](closure.json)에 별도로 기록한다. 예산 예약과 실제 청구액은 구분하며 계정 잔액·청구 내역은 공개 자료에 포함하지 않는다.
