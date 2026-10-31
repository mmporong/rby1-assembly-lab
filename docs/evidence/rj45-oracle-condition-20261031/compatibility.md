# 저장된 파지 근거의 CPU 계산 호환성

새 Crusoe L40S 서버에서 학습을 시작하기 전, 과거 명목 파지 근거를 재검산하는 단계가 실패했다. 은행·보고서·trace 파일 해시는 원본과 같았다. 이 기록은 로봇 접촉 실패나 학습 결과가 아니다.

검증기는 저장된 raw pose에서 구한 기하값과 runtime 기록값을 기존 허용오차로 검사한 뒤, 재계산한 파생값 dict를 과거 JSON dict와 정확 비교한다. NumPy 1.26.0에서도 CPU의 OpenBLAS·벡터 연산 경로가 달라지면 마지막 비트가 달라질 수 있었다.

| 계산 환경 | 검사한 행 | 정확 비교 불일치 |
| --- | ---: | ---: |
| 기본 환경 | 1,360 | 573 |
| AVX-512 비활성화 | 1,360 | 316 |
| OpenBLAS Zen 고정 | 1,360 | 384 |
| 두 설정 함께 적용 | 1,360 | 0 |

첫 불일치는 orientation `0.0007914530997118108`과 `0.0007914530997118109` 사이의 1 ULP 차이였다. 전체 최대 차이는 orientation 2 ULP, lateral 1 ULP였으며 depth와 상대 자세 목록은 같았다. 수치는 [기본 환경 진단](default.json), [통합 설정 진단](compat-combined.json)에 보존한다. CPU 재현 스크립트는 [reproduce_bank_geometry.py](reproduce_bank_geometry.py)다. 이 JSON은 로컬 CPU 재현 근거이며 원격 물리 성공을 뜻하지 않는다.

기존 AWS CPU는 EPYC 7R13, 새 서버는 EPYC 9254다. 파생값 비교 코드를 바꾸는 대신 Python 시작 전에 다음 환경을 고정했다.

```bash
export OPENBLAS_CORETYPE=Zen
export NPY_DISABLE_CPU_FEATURES=AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL
```

통합 설정에서 원본 파지 로더의 train 8개 검증도 통과했다. 원격 runner는 물리 환경 생성 전에 같은 검증을 다시 수행한다. 은행·원본 소스·허용오차·성공 기준·보상은 수정하지 않았다. 호환 설정과 적용 시각은 [환경 입장권](environment-admission.json)에 고정했다.

서버의 드라이버는 [Isaac Sim 5.1 요구 조건](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)에 맞춰 565에서 580.196.07로 바꿨다. 재부팅 뒤 로드된 커널 모듈과 CUDA 연산은 [검증 로그](driver-validation.log.gz)에 있다. 별도 PhysX 검사에서는 240 Hz로 16스텝을 실행했으나 앱 종료가 180초 제한으로 끝났다. 이 검사를 전체 수명주기의 정상 종료로 표현하지 않는다.
