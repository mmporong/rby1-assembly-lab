# RJ45 task-admission full90 중단 근거

상태: **32/90 보존 뒤 native process 중단, G1 미완료**. source `2e1f926`의 첫 full90은 `2026-10-31T16:50:00.704607Z`에 시작해 `17:51:01.308500Z`에 child exit `-11`로 끝났다. Linux signal 11인 `SIGSEGV`로 관찰된 종료 분류이며 native 원인은 확정되지 않았다. full report는 생성되지 않았다. tuning 18회와 admission 14회, 합계 32개 episode 원자 파일과 run identity를 회수했다. target은 `18:02:26.445379Z`에 `STOPPED`로 확인됐고, 이후 G1 guard timer와 service가 각각 `inactive`, `not-found`임을 확인했다.

이 중단은 강화학습 실패나 제어 방법의 성능 결과가 아니다. native 원인은 확정되지 않았고, 13,800초 process timeout도 만료되지 않았다. 실행 로그의 `Timeout (0:01:00)!`은 60초마다 stack을 남기는 traceback heartbeat다. 보존된 32회에는 tuning 후보 B가 A/B/C 동률 3/6에서 사전등록 tie-break로 선택된 결과가 포함되지만, B는 목표 4/6에 미달했다. full90 전체 판정이 없으므로 32회의 물리 성과를 집계하거나 남은 58회와 합성하지 않는다.

## 보존과 provenance 범위

32개 episode의 `run_identity_sha256`은 모두 `2249d6f2c505fa57c3403128201d7020abb20d506006706e46fdfc8b34621ef4`와 일치한다. 실행 source는 `2e1f926`, runner SHA-256은 `46efe17e7880be10ee8863f1fbeba1ef48ca88b417fa0954380be726154f3fcb`, development config SHA-256은 `ce917d703735e311c44af344486eb92d3049abfbc61ce2836c42f4e6b7719386`, frozen runtime commit은 `104721acf2222c1cfe8ec86cd030f5116a4fef93`이다. 파일별 SHA는 [`summary.json`](summary.json)에 보존한다.

이 실행의 prelaunch/end provenance map에는 물리에 영향을 주는 입력 13개 중 일부가 포함되지 않았다. 중간·종료 시점의 보조 capture로 frozen runtime 35개, current packet 10개, 물리 입력 12개의 SHA 일치를 확인했고 G0 이전 기록과 current source 9개, cache asset 3개의 선언 SHA도 일치했다. 다만 이 보조 capture는 prelaunch 전체 봉인을 소급해 만들 수 없다. `OmniPBR.mdl`은 authored Shader source로 확인했지만 실제 렌더 resolver context는 확정하지 못했다.

## 다음 실행 경계

과학 조건, bias, 성공 기준과 admission gate는 바꾸지 않는다. 입력 13개를 prelaunch와 종료 시점에 모두 봉인하고 process 격리를 보완한 뒤, 짧은 fresh-process 진단을 먼저 수행한다. 진단을 통과한 새 campaign은 처음부터 실행하며 현재 32회와 섞지 않는다. 이 진단과 새 campaign은 아직 실행하지 않았다. `task_admission_completed=false`, `rl_pilot_admitted=false`, main training 미시작이며 G2/G3도 미완료다.
