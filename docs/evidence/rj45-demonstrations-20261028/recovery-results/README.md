# 4C 교사 수집 인프라 복구 근거

이 폴더는 교사 데이터 수집을 시작하기 전에 드러난 두 인프라 실패와 복구 결과를 묶는다. 작업 성공률을 높이려고 기준을 바꾼 기록이 아니다. `reset-recovery-check.json`의 `success_and_safety_gates_relaxed`는 `false`이고, 복구 전후 task config SHA-256은 `7ac32092e6e723be066c80c53aecbbd873792e849ce42b742dbafd92cb3af88b`로 같다.

## 1. reset 검사 오탐과 복구

최초 N=256 reset 검사에서는 socket 상태가 다르다는 오류로 실행이 멈췄다. 진단 결과 위치와 속도 차이는 0이었고, 쿼터니언 `w`가 `1.0`에서 `1.0000001192092896`으로 달라진 것이 핵심이었다. 차이 `1.1920928955078125e-07`은 1.0에서 float32 1 ULP에 해당한다. 다른 쿼터니언 성분의 최대 차이는 `9.094947017729282e-13`이었다.

커밋 `eec12601085abf558c137966258264199b2920a7`은 동일 회전을 나타내는 쿼터니언을 정규화해 비교한다. 해당 커밋을 격리한 트리에서 reset checker 테스트 20건이 통과했다. 이후 N=256에서 64개 환경씩 네 차례 부분 reset을 수행했고, 매 회차의 나머지 192개 환경 상태 보존 검사까지 통과했다.

- 최초 실패: [`first-reset-failure-summary.json`](first-reset-failure-summary.json), [`first-reset-check.json`](first-reset-check.json)
- 진단 요약: [`diagnostic-summary.json`](diagnostic-summary.json)
- 복구 검사: [`reset-recovery-check.json`](reset-recovery-check.json)
- eec nominal 종료값: [`eec-nominal-exit.json`](eec-nominal-exit.json)

## 2. 수집기 의존성 검사 실패와 복구

reset 검사를 통과한 다음 교사 수집은 의존성 검사에서 멈췄다. 고정한 배포판은 7종이었지만 검사 결과에는 `isaacsim`, `isaaclab`, `isaaclab-rl`, `torch`, `numpy`, `h5py` 여섯 종만 있었고 `rsl-rl-lib`가 빠졌다. train과 dev 모두 완료 attempt가 0이었으며 데이터셋은 발행되지 않았다. 따라서 BC 학습도 시작되지 않았다.

커밋 `104721acf2222c1cfe8ec86cd030f5116a4fef93`에서 수집 환경 검사와 실패 종료 코드를 고쳤다. 같은 커밋으로 새 nominal 실행을 수행해 return code 0과 cleanup 완료를 확인했다. 이 결과는 수집 인프라가 다시 실행 가능한 상태라는 근거이며, 교사 데이터 수집이나 BC 성능 결과를 뜻하지 않는다.

- 수집 실패: [`collector-dependency-failure-summary.json`](collector-dependency-failure-summary.json), [`collector-train-summary.json`](collector-train-summary.json)
- 복구 nominal 종료값: [`collector-nominal-exit.json`](collector-nominal-exit.json)
- 검증 수치 요약: [`recovery-verification.json`](recovery-verification.json)

## 출처와 보존 범위

원격 완료 결과 5개와 관련 control·plan·preparer를 48개 파일로 고정했다. archive 생성 전후 원본 해시가 같았고, 로컬 안전 추출 뒤 파일별 크기와 SHA-256을 모두 다시 확인했다. 실행 중이던 `artifacts/rj45-4c-teacher-bc-collector-recovery-v1-20261028`은 읽거나 포함하지 않았다.

공개 폴더의 `first-reset-*`, `collector-*-summary.json`, `reset-recovery-check.json`, `eec-nominal-exit.json`, `collector-nominal-exit.json`은 archive 안 파일과 바이트가 같은 사본이다. `diagnostic-summary.json`과 `recovery-verification.json`은 출처 값을 줄여 적은 요약이다. 전체 출처와 파일 해시는 [`source-manifest.json`](source-manifest.json)에 있다.
