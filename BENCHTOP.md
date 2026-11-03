# Benchtop CITL: 15 mm 단일 평면

## 2026-11-02: LTS300/M 추가 및 SLM 켤레면 좌표 실측

현재 시스템에는 Thorlabs **LTS300/M, 일련번호 45897069**가 추가되어 있다.
**125 mm 초과는 카메라 하우징과 마지막 렌즈의 충돌 구간이다.**
`benchtop.stage.Stage`는 런타임 소프트웨어 상한을 **124 mm**, 기본 실험 범위를
**99–103 mm**, 속도를 0.5 mm/s로 제한한다. 원점 복귀·상대 이동·jog·좌표 재설정은
노출하지 않는다. Kinesis `LoadSettings`가 소프트웨어 제한을 기본값으로 되돌리는 것을
확인하여, 이 드라이버는 연결할 때마다 제한을 다시 적용하고 읽어 검증한다.
전원 재인가나 다른 프로그램의 설정 로드 이후까지 보장하는 물리 인터록은 아니다.

초기 좌표는 101.040911 mm였다. 사용자가 선정리 중 센서를 건드렸다고 알려온 뒤,
이전 측정 폴더 `results/slm_plane_20261102/` 전체를 `INVALIDATED.json`으로 제외했다.
센서를 더 건드리지 않는다는 확인 이후 `results/slm_plane_20261102_clean/`에 다시 측정했다.
SLM의 큰 암점 9개와 별도 검증점 4개로 횡방향 등록을 새로 측정했다.
조명에 편향된 밝기 중심 대신 원형 경계를 맞춰 RMS 0.188, 검증 최대 0.365
half-sensor 픽셀을 얻었다. 이 값은 깊이 정확도가 아니다.

작은 ±위상 사인파가 밝기 무늬로 변환되는 응답의 최소점을 측정했다.
25개 유효 위치 측정과 원시 green 센서 배열, 촬영 메타데이터, 위상 생성 코드를 보존했다.
넓은 탐색 7개, 0.1 mm 간격 역방향·정방향 각 7개, 직접 비교 4개이다.
넓은 탐색의 101.79 mm 지점은 SLM framebuffer 불일치로 촬영 전에 중단했으며
그 지점은 결과에서 제외했다. 검사를 우회하지 않고 다시 연결·검증했다.

| 중앙부, 24 SLM 픽셀 주기 | 역방향 최소점 | 정방향 최소점 |
| --- | ---: | ---: |
| x 방향 위상 변화 | 100.922 mm | 100.925 mm |
| y 방향 위상 변화 | 100.615 mm | 100.635 mm |
| 두 방향 응답 합의 최소점 | 100.785 mm | 100.775 mm |

**101 mm 근처라는 추정은 맞지만, 단일한 정확한 SLM 면 좌표는 아직 확정하지 않았다.**
중앙부 임시 절충값은 **100.78 mm**다. 두 방향 사이 약 0.30 mm 차이와 시야 위치 의존성이
남아 있으며 비점수차·정렬·SLM 전달 특성 등이 원인일 수 있다. 원인을 고유하게 구분한
측정은 아니다. 절충값의 왕복 차이 0.010 mm는 이 지표의 반복성이지 절대 위치 정확도가 아니다.
큰 잔류 응답과 낮은 적합도, 범위 밖 극값을 보인 x16 패턴은 절충값 계산에서 제외했다.
101.04 → 100.78 → 100.78 → 101.04 mm 직접 비교에서 중앙부 위상→밝기 잔류 응답 파워는
평균 71.6% 줄었다. 이 수치는 홀로그램 화질 개선율이 아니다.

측정 후 **101.040000 mm, 정지·homed 상태**로 복귀했고 마지막 읽기에서
124 mm 소프트웨어 상한을 확인했다. 전체 세션의 최대 이동 명령은 102.04 mm였다.
레이저 설정, 원점, 위치 카운터는 변경하지 않았다. 기존 +15 mm 학습 자료와 설정은
역사적 실험 조건으로 보존하며, 새 스테이지 좌표와 전파 거리의 부호 관계는 별도 보정이 필요하다.

결과: [요약 및 최종 장비 상태](results/slm_plane_20261102_clean/summary.json),
[측정 곡선](results/slm_plane_20261102_clean/summary.png),
[전체 적합 결과](results/slm_plane_20261102_clean/fits.json),
[안전/참조 프로필](configs/stage_lts300.json).
원시 배열과 소스 스냅샷의 해시는 같은 폴더의 `sha256.json`에 있다.
안전 경계 단위 테스트 3개, Python 컴파일 검사, 실제 장비 제한 읽기 검증을 통과했다.

## 2026-10-26: 완료된 학습 결과를 모니터에 연결

`http://127.0.0.1:8767/`의 최신 실험 선택 시 준비 과정의 빈 `live` 기록만
보이던 문제를 수정했다. `completed_all.json`의 실제 완료 상태·시각을 표시하고,
검증된 수치 부록의 색별 검증 L1 곡선, 선택 epoch, 8장 최종 재촬영 지표 및
고정/배율 보정 비교 영상을 연결했다. 예전 실험은 기존 뷰어에서 계속 탐색한다.
결과 화면의 전용 주소는 `/paper/?run=paper_cnnpropcnn_20261024`이다.
수치 부록을 읽을 때 완료 기록·최종 요약·학습 로그의 원본 해시를 대조한다.
서버만 재시작했으며 새 학습이나 카메라 촬영은 수행하지 않았다.
원본 보존·잘못된 영상 접근 차단·변경된 원본 감지·기존 상태 API의 테스트를 통과했다.
브라우저에서 세 학습 곡선, 최종 지표, 영상/밝기 전환, 초기화와 이전 실험 왕복을 확인했다.
화면 기록: `docs/learning-20261024/monitor-results-20261026.jpg`.

## 2026-10-25: 복소장 모델 단일 평면 실험 완료

`results/paper_cnnpropcnn_20261024/completed_all.json` 기준 19:43:13 KST 정상 종료.
작업 프로세스 종료와 빈 오류 로그를 확인했다. 고정 +15 mm 한 평면의 결과이며 3D 재현이 아니다.

- 색별 1,100쌍의 고정 데이터로 적색/녹색/청색 각각 61/54/73 epoch 학습했다.
  선택 모델은 51/44/63 epoch, 검증 진폭 L1은 0.024772/0.028692/0.030518이다.
  모두 검증 미개선 10회로 종료했으며, 전역 수렴을 뜻하지 않는다.
- 미사용 원본 8장의 DPAC/SGD-ASM/SGD-CNNpropCNN을 각각 3회 재촬영했다.
  두 SGD는 같은 초기 위상, Adam 0.01, 1,000회; 최종 CITL은 수행하지 않았다.
- RGB·반복·영상 동일 가중 평균 고정 백색 진폭 MSE:
  DPAC 0.104455, SGD-ASM 0.094341, SGD-CNNpropCNN 0.057681.
  CNN은 ASM보다 8/8장에서 개선됐고 평균 38.9% 낮았다.
- 채널별 사후 스칼라 보정 진폭 MSE: 0.052020 / 0.065631 / 0.021396.
  CNN은 ASM보다 평균 67.4% 낮았다. 실측 광량을 올린 결과는 아니다.
- 보정·클리핑 sRGB SSIM은 0.479834 / 0.132247 / 0.448220으로 DPAC이 가장 높았다.
  0868의 고정 MSE는 DPAC 0.122372, CNN 0.225655로 CNN이 나빴다.
- 중앙 3×3 π 유한 변화의 진폭 응답 코사인 유사도는 ASM→CNN:
  R 0.243→0.425, G 0.325→0.488, B 0.321→0.372.
  청색 광강도 응답 RMSE는 0.016135→0.016359로 증가했다.
  전체 야코비안이나 모든 위상에서의 미분 정확성을 검증한 것은 아니다.
- 최종 평가 기준 광량 변화 범위는 획득 시작 대비 R −0.96~+0.06%,
  G +0.16~+0.44%, B −0.04~+2.97%. 고정 보정값을 유지했다.
- 원시 최종 촬영 배열 72개, 섭동 촬영 배열 6개, 위상 세트 8개,
  최적화 이력 48개를 확인했다. float64로 고정/보정 진폭 MSE를 재계산한
  최대 절대 차이는 각각 4.41e-8 / 1.30e-8이었다.
- 해설 및 수치 부록: `docs/learning-20261024/neural-3d-holography-learning.html`
  (07 단원), `paper-results-evidence.json`. 이전 소형 모델의 부정적 결과를 유지한다.
  새 결과와 이전 결과는 구조·데이터·타깃·반복 수가 달라 구조만의 절제 실험이 아니다.
- 학습자료의 데스크톱·모바일 표시, 수식 99개, 펼침 설명 19개와 새 비교 도구를
  확인했다(`qa.json`). 실험·해설·학습자료 갱신 완료 후 30분 정기 점검을 종료했다.

## 2026-10-24: 논문 크기 CNNpropCNN으로 새 실험

사용자가 고정 단일 평면 장비임을 확인했다. 새 실행은
`results/paper_cnnpropcnn_20261024/`이며 이전 측정값과 모델을 덮어쓰지 않는다.
8개 깊이, RGBD 마스크, ADMM 및 깊이 일반화 재현으로 해석하지 않는다.

- `benchtop.paper_model`은 저자의 `unet.py`를 수정 없이 사용한다.
  SLM U-Net 8/32/512, target U-Net 5/8/128, 실수·허수 두 채널,
  색별 66,746,156개 파라미터이다. 관측면 CNN 이전의 ASM 위상을 반환한다.
- RGB 노출, CFS 색 슬롯 분리, off-slot 배경, 9점 등록 및 4점 별도 검증을
  새로 완료했다. 전체 SLM 1920×1080, ROI 1680×960, 거리 15 mm를 유지한다.
- 색별 100 DPAC + 1,000 SGD의 **고정된** 학습 쌍을 새로 촬영한다.
  검증/테스트는 각각 12개 원본에서 22쌍, 최종 평가 8개 새 원본을 따로 둔다.
  이전 결과의 DIV2K validation 경로 32개를 보수적으로 제외했다.
  SGD 반복 수 50~200 균등 분포는 본문에 범위가 명시되지 않아 정한 지역적 선택이다.
- 학습은 색별 독립, Adam 5e-4, batch 1, 진폭 L1, 검증 미개선 10 epoch 종료다.
  200 epoch는 안전 상한이며 도달 시 수렴이라고 하지 않는다. AMP를 사용하지 않는다.
  재계산 체크포인트는 수학적 계산을 바꾸지 않는다. optimizer 상태를 25 sample마다
  원자적으로 저장하므로 강제 종료 시 마지막 저장점 이후 최대 24개 update는 재실행한다.
- 테스트 예측, 새 8장에 대한 DPAC/SGD-ASM/SGD-CNNpropCNN 실제 출력(각 3회),
  Fig. 6의 3×3 π 변동에 대한 유한차분 응답을 분리해 평가한다. 각 SGD에는 같은
  초기 위상과 1,000 step, lr .01을 준다. 최적 위상을 업데이트 전에 복사한다.
  고정 white MSE, 최적 스칼라 진폭 MSE/PSNR, 그 스칼라 적용 후 clipped sRGB
  PSNR/SSIM을 명시적으로 구분한다. 논문 표의 숫자와 직접 동등시하지 않는다.
- 표시 위상 uint8, 등록 후 진폭 float32, 각 색 슬롯의 평균 sensor intensity float32,
  frame ID/시각, 데이터·원본·보정·소스 해시를 보존한다. 전체 bank를 유지한다.
  실행 150 GiB, 여유 공간 100 GiB 제한; 50패턴마다 고정 위상 drift를 검사하며
  기준 광량 20% 초과 변화는 중단한다. 예전 자료를 재정규화하지 않는다.

SLM이 이 Windows 세션의 유일한 기본 모니터였다. 출력 창은 fullscreen/topmost로
열고 각 표시마다 다시 올린다. 시스템 알림은 그 위를 덮을 수 있으므로, 사용자 요청에
따라 OpenAI.Codex / Windows.SystemToast.Suggested / Windows.ActionCenter.SmartOptOut의
`ShowBanner`만 0으로 설정했다. 알림 수신·보안·잠금 설정은 변경하지 않았다.
원래 값은 실행 폴더의 `notification_*_backup.json`에 있고 복구 스크립트는
`scripts/restore_paper_notification_banners.ps1`이다. 실험 중에는 복구하지 않는다.
기존 알림 제안은 Windows 알림 API로 지웠다. 실제 framebuffer 검사는 계속 켜져 있다.
다른 시스템 창이 침범하면 여전히 촬영을 중단하며 검사를 우회하지 않는다.

전체 테스트 46개가 통과했다. 새 실제 촬영 2쌍으로 전체 해상도 L1/Adam 및 복원을
검증했고, 연속 두 번째 update와 복원 후 update의 가중치 최대 차이는 0이었다.
이는 기능 검증이며 최종 성능 결과가 아니다. 자세한 기록은
`verification/training_resume.json`에 있다. 14개 RGB 패턴을 촬영한 뒤 검증을 위해
의도적으로 안전 중단했으며, 같은 고정 계획으로 이어 간다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m benchtop.paper_campaign --out results\paper_cnnpropcnn_20261024 --stage all
```

재개 전 status의 PID가 종료됐는지 확인하고, 의도된 중단 요청 파일을 제거한다.
동일 소스·보정·계획만 재개를 허용한다. 소스를 고쳐야 하면 원본 snapshot과 변경 원인을
보존하고 재검증한 후 명시적인 migration을 해야 한다. source_snapshot을 임의로
덮어쓰거나 검사를 끄지 않는다. `STOP_REQUESTED`는 이 실행 폴더에 둔다.
관찰 화면은 localhost 전용 `http://127.0.0.1:8768/`이다.
`status.json`, `acquisition.jsonl`, `training.jsonl`, `evaluation/summary.json`과
`worker_stdout.log`, `worker_stderr.log`가 실제 진행과 종료 판단의 근거다.
30분 후속 점검은 이 채팅에 연결되어 있으며 자동화 ID는 `automation`이다.

## 2026-10-23 후속 실측 진단

`results/citl_diagnosis_20261023_run02/`는 완료된 원본 실험과 분리한 진단이다.
모델과 보정 파일을 별도로 복사하고, 원본 측정·모델·데이터의 SHA-256을 실행
전후 비교한다. 가변 live/preview_cache와 기존 report 폴더는 이 해시 비교에서
제외한다. 최초 진단 폴더에는 모니터 timestamp 중복으로 중단된 초기 시도와
SLM 화면 가림 증거를 보존했다. 실제 전체 측정은 run02에 있다.

- 고정 위상: 0846/0898의 초기 CNN 위상과 기존 최적 CITL 위상을 각 8회 측정.
  순서를 섞고, 픽셀별 진폭 표준편차·평균 광량 CV·MSE 변동을 기록한다.
- 국소 민감도: 0846의 초기/기존 최적 위상에서 3개 공간 블록 방향, epsilon
  0.05/0.1 rad를 적용한다. 중앙 유한차분을 독립 2쌍 측정하고 두 번째 쌍의
  +/- 촬영 순서를 반대로 한다. 동일 위상 변동을 ASM/CNN에도 적용하여 영상
  응답 cosine과 오차를 비교한다. 실제 카메라 residual을 사용하는 autodiff
  loss 방향 미분도 비교한다. 이는 전체 SLM Jacobian 측정이 아니며, 2회 반복의
  SE는 신뢰구간이 아니다. 같은 방향의 두 epsilon도 독립 표본이 아니다.
- 학습률: frozen CNN과 같은 초기 위상에서 Adam을 새로 만들어 0.02/0.005를
  각각 200회 수행한다. 순서는 건물 0.02 → 꽃 0.005 → 건물 0.005 → 꽃 0.02.
  최적/마지막 위상을 각각 3회 재촬영하고, 50회마다 고정된 건물 기준 위상을
  추가 촬영하여 세션 중 광량 변화를 추적한다. 200회 동등 예산 비교이며
  수렴을 입증하는 실험은 아니다. 타깃별 최적화 재시드 반복도 수행하지 않았다.

이 두 이미지는 원본 최종 평가 당시에는 미사용 타깃이었지만, 후속 진단과
학습률 비교에 사용했으므로 이후 독립적인 미사용 평가 이미지로 취급하면 안 된다.
노출·레이저·LUT·등록·정규화·모델 가중치는 이 진단에서 변경하지 않는다.

```powershell
# 반드시 원본 캠페인 종료와 SLM 화면 가림 해소 후, 새 출력 폴더로 실행
.\.venv\Scripts\python.exe -X utf8 -u -m benchtop.diagnose_citl --out results\citl_diagnosis_new --steps 200 --rounds 8
.\.venv\Scripts\python.exe -X utf8 -m benchtop.diagnose_report --out results\citl_diagnosis_new
```

`report/`에 안정성·유한차분·학습률 비교 몽타주와 loss/위상 변화 곡선을 만든다.
SLM 화면을 덮은 ChatGPT 알림 배너는 Windows 설정에서 검증 중에만 숨기고,
종료 후 원래 표시 상태로 복구한다. 알림 센터 수신과 Windows 보안 설정은
변경하지 않는다. 실제 framebuffer 불일치 검사와 촬영 전후 검사는 계속 적용한다.

이 경로는 원 논문의 CITL 방식인 `simulation + (camera - simulation).detach()`를
현재 장비의 PySpin/HDMI 출력에 연결한다. `benchtop/`은 기존 연구 코드의 오래된
Lightning, FlyCapture, slmPy 의존성과 분리된 실행 경로다. 다른 리포지토리를
import하거나 BenchtopDisplay 환경을 수정하지 않는다.

## 2026-10-23: RGB 공동 학습 및 수렴 종료

`benchtop.campaign`은 RGB 세 파장별 물리 모델을 유지하면서 한 RGB 샘플의
평균 진폭 MSE를 공동 목적함수로 사용한다. 채널별로 backward하여 FFT graph를
해제한 뒤 **한 번의 Adam step으로 세 채널을 함께 갱신**한다. 색 사이의 광학
결합을 학습하는 3-channel CNN으로 교체한 것은 아니며, 소형 보정 CNN 세 분기를
사용한다. 모델 구조는 원 논문의 CNNpropCNN과 다르다.

- 실제 카메라 CITL: 기본 최대 **1000 iteration**, 최소 **200 iteration**.
  매 20회에 직전/최근의 서로 겹치지 않는 20개 loss 중앙값을 비교한다.
  차이의 절댓값이 `max(1e-5, 이전 중앙값 × 0.002)` 이하인 구간이
  3회 연속이면 `converged`로 종료한다. 계속 개선되거나 증가하면 이 조건으로
  종료하지 않으며, 1000회에 `iteration_limit`으로 종료한다. 반환 위상은
  실제 측정 loss가 가장 작았던 업데이트 이전의 RGB 위상이다.
- CNN: cycle당 최대 **350 epoch**, 최소 **50 epoch**, validation patience **30**.
  validation 정체 시 학습률을 낮추고, 종료 이유를 `validation_plateau`와
  `epoch_limit`으로 구분한다. epoch 수는 각 cycle의 데이터 집합에 대한 횟수다.
- `campaign/models/rgb_latest.pt`에 세 분기 가중치, **Adam moments/step**, scheduler,
  누적 epoch/sample 수, 현재 epoch 내 RGB sample offset을 함께 저장한다.
  다음 cycle/프로세스는 latest와 짝이 맞는 optimizer를 복원한다. 평가에는 별도
  `rgb_best.pt`와 그 RGB 세 분기 export를 사용한다. 사용자 중단 시에도 현재의
  완결된 RGB update까지 저장하고 남은 sample부터 이어갈 수 있다.
- 예전 채널별 checkpoint에는 Adam 상태가 없으므로, 최초 전환에서는 가중치만
  가져오고 새 optimizer를 시작한다. 이후부터 optimizer 상태가 보존된다.
  최근 8개 cycle의 작은 learnable tensor checkpoint도 유지한다.
- 풀컬러 target/카메라/모델 프리뷰는 상단에 배치된다. 선형 진폭을 제곱한 뒤
  sRGB로 표시하며, 위상·오차·multiplier의 색은 별도로 표시한 진단용 표현이다.
  CNN 카메라 프리뷰는 저장된 validation 측정값이다. 실시간 CITL과 구분한다.
  종료 이유, 실제 반복 횟수와 수렴 기준을 각 과정에 표시한다.

짧은 검증은 원본 실행을 변경하지 않는 새 폴더에서 실행할 수 있다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m benchtop.campaign --out results\rgb_joint_check --verify-source results\rgb80_10h_20261022 --verify-stage train --epochs 2 --min-epochs 2 --citl-max-steps 24 --citl-min-steps 12 --citl-window 4 --citl-patience 2 --citl-preview-every 4
# 위 프로세스가 종료된 다음, 새 프로세스에서 optimizer 복원 및 실측 확인
.\.venv\Scripts\python.exe -X utf8 -u -m benchtop.campaign --out results\rgb_joint_check --verify-stage resume --resume --epochs 2 --min-epochs 2 --citl-max-steps 24 --citl-min-steps 12 --citl-window 4 --citl-patience 2 --citl-preview-every 4
```

`results/rgb_joint_verify_20261023/verification.json`에 실제 검증 결과가 있다.
4개 train / 2개 validation RGB 실측 쌍으로 각 프로세스에서 2 epoch씩 수행하여,
모든 Adam step이 8에서 16으로 이어짐을 확인했다. 실제 CITL은 짧은 검증용
24회 상한으로 종료했고, 별도 합성 측정 테스트에서 수렴 조기 종료를 검증했다.
DIV2K 0865 한 장을 각 방식 3회 재촬영한 평균 진폭 MSE는 ASM 0.066224,
CNN 0.062434, CNN+CITL 0.033867이다. 이는 한 영상의 검증 결과이며 전체 데이터
일반화 평가가 아니다. 최대 CUDA allocated 1.127 GiB / reserved 1.326 GiB,
추가 실행 폴더 약 0.292 GiB. 원본 파일 hash 불변과 카메라 원래 설정 복원을 확인했다.

모니터는 기본적으로 localhost에만 바인딩한다. 현재 실행 서버는 localhost와
이 PC의 원내망 주소에만 바인딩하며 장비/학습을 제어하는 HTTP API는 없다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m benchtop.live --root results --port 8767 --host 127.0.0.1 --host 10.254.61.49
```

이 PC: `http://127.0.0.1:8767/`, 다른 PC용: `http://10.254.61.49:8767/`.
두 주소의 HTTP 200을 이 PC에서 확인했다. Windows Public 방화벽은
`BlockInbound,AllowOutbound`이며 TCP 8767 허용 규칙의 유효 여부는 조회 권한
제한으로 확인하지 못했다. 방화벽 정책은 변경하지 않았다. 다른 PC의 실제
접속은 사용자가 현재 확인하기 어려워 미검증이며, 외부 접속 허용을 보장하지 않는다.

## 기존 10시간 RGB 학습·실측 평가

기존 장시간 실행 `results/rgb80_10h_20261022/`는 사용자 요청으로 중단한 뒤,
2026-10-23 10:11 KST에 수정된 공동 RGB 설정으로 재개했다. 이전 6시간 35분의
누적 실행 시간과 208개의 rolling/validation/test 샘플을 이어받았다. 이전 모델과
실행 소스는 `campaign/migration_joint_20261023T101145/`에 별도 백업했다.
`--citl-first`로 실제 카메라 수렴 검증을 먼저 수행하고, 이후 공동 CNN 학습 및
최종 평가를 진행한다. 실제 CITL 소요 시간을 기준으로 최종 평가 시간을 확보한다.
SLM 출력은 1920×1080,
복원/학습 ROI는 **1680×960 (FHD 면적의 77.8%)**으로 확대했다. 새 RGB 보정은
ROI 가로·세로 ±42% 위치까지 9점으로 맞추고, ±48%의 네 모서리에서 따로 검증한다.
거리 15 mm, 파장, 센서 노출 및 fixed-white 정규화는 한 실행 동안 고정한다.

```powershell
# 새 폴더에서 넓은 ROI 보정 후 실행
.\.venv\Scripts\python.exe -X utf8 -u -m benchtop.rgb_run prepare --config configs\benchtop_rgb80.json --out results\rgb80_new
.\.venv\Scripts\python.exe -X utf8 -u -m benchtop.campaign --out results\rgb80_new --hours 10 --max-gib 12 --min-free-gib 50

# 전체 경과·gain·실측 평가·drift 보고서 (실행 중에도 읽기 가능)
.\.venv\Scripts\python.exe -X utf8 -m benchtop.campaign_report --out results\rgb80_new
# 종료 후: 데이터 분리, 10시간 근사 실행 시간, 최종 8장 평가와 sealed test 독립 재계산
.\.venv\Scripts\python.exe -X utf8 -m benchtop.campaign_report --out results\rgb80_new --audit
```

처음에는 train 32 / validation 8 / test 8개의 원본으로 RGB 실측을 확보한다.
이후 **RGB 공동 CNN 학습 → tracking 원본 2장의 ASM/CNN/CITL 실제
비교 → flat-field drift 확인 → 새 train 원본 16장 획득**을 반복한다. 학습은
수정된 종료 조건과 optimizer 복원은 위 절을 따른다. 기존 중단 실행은 채널별
최대 20 epoch / patience 5 / CITL 30회 설정으로 수행한 과거 결과다.
학습용 원본은 DIV2K train 800장을 순환하며, 최근 192개 RGB 실측 샘플만 유지한다.
평가용 최적 checkpoint와 학습 이어받기용 latest+optimizer를 별도로 보존한다.

Validation, sealed test, 반복 tracking, 마지막 실측용 원본은 각각 8장씩 서로
겹치지 않는다. sealed test는 마지막 모델이 정해진 뒤 한 번만 평가한다. 마지막
실측용 8장은 이전 학습·모델 선택·tracking에 쓰지 않는다. 실측 비교는 세 방식
각각 3회, 반복 블록 안에서 표시 순서를 섞어 측정한다. 약 10시간의 작업 시간에는
패턴 생성, 실측, CNN fitting, 평가가 포함된다. 마지막 평가의 길이에 따라 종료
시각은 조금 달라진다. 중단 후 `--resume`하면 중단 전 작업 시간을 이어 계산한다.
재개 전에는 `status.json`의 PID가 실제로 종료됐는지 확인해야 한다.

저장 정책은 다음과 같다.

- 학습 데이터: 최근 train 192개 + 고정 val/test 16개. 표시한 위상은 RGB uint8,
  rectified 카메라 진폭은 float16으로 저장한다. float32 위상·타깃의 중복 저장은 없다.
- 모델: RGB 각각 best/latest, 작은 learnable-only 가중치는 최근 8 cycle.
- 시각화: 최근 24개 과정, 과정별 최초와 최근 합계 최대 12개 snapshot. 전체
  epoch의 scalar loss, gain, parameter norm과 실측 평가 수치는 JSONL로 남긴다.
- Tracking 영상은 최근 2개 평가만 유지하며, 최종 8개 평가는 위상과 실측 진폭도
  보존한다. 원시 Bayer 프레임이나 매 step의 FHD float 배열은 저장하지 않는다.
- 실행 폴더 12 GiB / 여유 디스크 50 GiB 조건을 감시한다. 한 번의 파일 쓰기 크기와
  최대 30초의 용량 집계 간격만큼 한계에 오차가 있을 수 있다. 기존 실험과 DIV2K
  원본은 삭제하지 않는다. 삭제 대상은 해당 실행에서 생성한 개별 파일로 제한한다.

`campaign/status.json`은 현재 PID, 누적 작업 시간, 과정, 디스크 사용량과 GPU 최대
메모리를 담는다. `campaign/STOP_REQUESTED` 파일이 있으면 다음 안전 확인 지점에서
중단하고 카메라 설정을 복원한다. Windows를 잠그거나 SLM 디스플레이가 가려지면
기존 실제 출력 검사에서 중단한다. 작업 중에는 프로세스 수명의 절전 방지 요청을
유지하고, 종료 시 해제한다. Windows 잠금 자체를 해제하거나 보안 정책을 바꾸지 않는다.

## 실시간 인터랙티브 학습 모니터

`프리뷰 데이터` 메뉴에서 다음 화면을 전환한다. 상단에 이미지 ID, train/
validation/CITL 용도, 기록 epoch와 시각, 현재 실행 중인 작업을 별도로 표시한다.

- **선택 과정:** CITL의 실제 카메라 기록 또는 CNN의 기존 고정 validation 프리뷰.
- **Train:** 최근 완료 epoch에서 실제 사용한 마지막 train 샘플을, 저장된 shuffle
  seed와 학습 로그로 재구성한다. DIV2K 타깃·해당 샘플의 실제 기록 카메라·SLM
  코드를 표시한다. 이미지 메뉴에서 다른 train 샘플을 직접 선택할 수도 있다.
- **Validation:** 첫 validation 샘플을 고정 표시하고, 학습 프로세스가 저장한
  CNN 예측·필터·gain 프리뷰를 재사용한다. 다른 validation ID도 선택할 수 있다.

현재 worker는 배치 단위 ID를 기록하지 않으므로 **순간적으로 처리 중인 batch를
추적하는 기능은 아니다**. 화면에 `최근 완료 epoch`와 기록 시각을 명시한다.
CITL 중에는 train 프리뷰가 직전 CNN epoch의 샘플로 유지되는 것이 정상이다.
`현재 진행` 줄은 실제 진행 중인 CITL 타깃을 별도로 표시한다.

이 기능은 학습 프로세스를 재시작하거나 모델 checkpoint를 열지 않는다.
Windows에서 checkpoint 읽기가 저장 시 rename을 방해할 수 있어, append-only
학습 로그·변경되지 않는 측정 쌍·기존 live snapshot만 읽는다. 렌더링은 GPU를
사용하지 않으며 새 PNG 캐시는 최근 8개 요청으로 제한한다. 원래 학습 파일은
수정하지 않는다. 모니터 서버만 갱신했고 PID 5064 / 시작 10:11:46 KST인 학습
프로세스가 유지되는 것을 확인했다. 이미지 접근은 읽기 전용 HTTP API다.

```powershell
# 별도 터미널에서 실행. 브라우저에서 http://127.0.0.1:8767/ 열기
.\.venv\Scripts\python.exe -m benchtop.live --root results --port 8767

# 기존 실제 DIV2K 촬영 쌍으로 새 CNN 학습: 카메라/SLM 연결 불필요
# --out에는 아직 학습하지 않은 새 폴더를 지정한다.
.\.venv\Scripts\python.exe -m benchtop.rgb_run train --data-root results\div2k_rgb_20261022 --out results\live_rgb_new --epochs 15
```

`rgb_run`은 실행할 때마다 `live/<run_id>/`에 상태와 영상 기록을 자동 저장한다.
뷰어는 1초마다 갱신하며, 새 출력 폴더는 **실험** 메뉴에서 선택한다. 기존 결과의
재생을 진행 중인 학습으로 표시하지 않는다. 실제 프로세스의 heartbeat가 10초 이상
끊기면 응답 지연으로 표시하고, 정상 종료·예외 종료도 구분한다. 서버의 기본값은
localhost이며 원내망 바인딩 옵션은 위 절을 따른다. 브라우저가 닫히거나 느려져도
학습이 브라우저 응답을 기다리지 않는다.

- **CITL:** 매 step 실측 amplitude MSE, 5 step마다 target/카메라/오차/SLM 위상/
  ROI 위상/gradient. RGB 합성 프레임의 마지막 CITL에도 연결된다.
- **CNN:** 매 epoch train/validation loss, gain, gradient norm, 모든 learnable tensor의
  L2 norm, 고정 validation 샘플의 실측/ASM/CNN 영상, multiplier, 세 convolution
  layer의 실제 필터. 원래 `.pt` 및 `.npz` 기록도 계속 저장한다.
- **탐색:** 채널·과정 선택, epoch/step 슬라이더, 이전/다음, 최신 따라가기,
  loss 로그 축, 곡선 클릭으로 시점 선택, 영상 클릭 및 1/2/4배 확대.

모든 영상은 표시 범위를 고정하고 범위 밖 픽셀 비율을 표시한다. 위상/gradient는
해당 카메라 측정을 만든 optimizer update 이전 값이다. CNN의 카메라 영상은
기존 실측 validation 샘플이며 매 epoch 새로 촬영하는 영상이 아니다. 뷰어 조작은
학습이나 하드웨어를 일시 정지하지 않는다. CNN만 학습할 때는 SLM 출력이 필요하지
않지만, 실제 CITL/촬영 중에는 Windows 잠금·절전·SLM 디스플레이 차단을 피해야 한다.
`--data-root`는 train 단계 전용이고, 입력 실측 데이터와 기존 checkpoint를 보존한다.

2026-10-22 검증 실행은 `results/live_rgb_verified_20261022/`에 저장되어 있다.
저장된 DIV2K 실제 RGB 72쌍으로 채널별 15 epoch를 재학습하며 모니터를 확인했다.

## DIV2K 이미지로 SLM 패턴 생성

로컬 원본은 `C:\Users\khchoi_exp\Documents\DIV2K dataset`의
`DIV2K_train_HR` 800장, `DIV2K_valid_HR` 100장이다. `--div2k-root`를 지정하면
합성 도형 대신 이 PNG 이미지로 타깃을 만든다. 현재 DIV2K 실행 디렉터리는
`results/div2k_rgb_20261022/`다. 기본 소규모 실행은 24장으로 16 train / 4 val /
4 test를 구성하며, 별도 validation 원본 2장을 CITL demo와 새 타깃 검증에 쓴다.
원본 이미지 단위로 분리하고 crop을 서로 다른 split에 섞지 않는다.

전처리는 **sRGB → 선형 광강도 → 종횡비 유지 리사이즈/중앙 crop → sqrt 진폭**
순서다. 공간 평균을 선형 광강도에서 수행한다. 손실 타깃은 현재 ROI인 384×640이고,
위상 계산과 출력은 실제 SLM 1080×1920이다. 영상별/채널별 min-max 정규화는 하지
않는다. 중앙 crop 때문에 원본 영상의 가장자리가 일부 제외될 수 있다.

```powershell
# DIV2K로 새 실험: 타깃 생성 → 보정 → 위상 생성 → 촬영/CITL → CNN → 새 영상 실측
.\.venv\Scripts\python.exe -m benchtop.rgb_run all --div2k-root 'C:\Users\khchoi_exp\Documents\DIV2K dataset' --out results\div2k_new --pairs 24 --pattern-steps 80 --steps 60 --epochs 15

# 타깃 준비만: 카메라/SLM 연결 불필요
.\.venv\Scripts\python.exe -m benchtop.rgb_run targets --div2k-root 'C:\Users\khchoi_exp\Documents\DIV2K dataset' --out results\div2k_new --pairs 24

# prepare가 끝난 실행 디렉터리에서 패턴만 생성: 이 단계는 하드웨어를 열지 않음
.\.venv\Scripts\python.exe -m benchtop.rgb_run generate --out results\div2k_new --pairs 24 --pattern-steps 80
```

`targets/manifest.json`에는 실제 선택 파일, 원본 split, SHA-256, 원본 크기,
리사이즈 크기, crop 위치와 색 변환 조건을 기록한다. 타깃 진폭은 먼저 파일로
고정하므로 이후 원본 파일이 바뀌더라도 한 실험의 타깃이 조용히 바뀌지 않는다.
이미 생성된 계획에 다른 경로/ROI/seed/개수를 넣으면 덮어쓰지 않고 거부한다.

`patterns/`에는 각 이미지의 **RGB888 위상 PNG**, RGB float32 위상 NPY,
타깃 미리보기와 채널별 최적화 곡선이 저장된다. `generate`는 채널별 ASM으로
기본 80회 최적화한다. `acquire`는 저장된 RGB 프레임을 그대로 출력해 각 CFS
발광 슬롯을 촬영하고, 각 채널의 위상과 측정을 `pairs/`에 저장한다. 따라서
합성 도형 실험과 달리 DIV2K 학습 쌍은 처음부터 RGB 합성 프레임의 측정이다.
학습용 24장이 생성되었다고 전체 900장의 패턴을 생성한 것으로 간주하지 않는다.
후속 단계는 같은 출력 디렉터리를 지정하면 저장된 DIV2K 타깃을 자동으로 사용한다.

### DIV2K 실행 결과 (2026-10-22)

24장에 대해 RGB 각각 80회 ASM 최적화로 패턴을 생성하고 실제 RGB 프레임을
촬영해 72쌍을 확보했다. CNN은 채널별 15 epoch 학습했다. 별도 validation 원본
`0815.png`로 채널별 60회 CITL, 학습에 사용하지 않은 `0891.png`로 새 출력 비교와
채널별 40회 합성 RGB CITL을 수행했다. 사용한 원본은 총 26장이다.

| 새 DIV2K 0891 실측, 3회 평균 | RGB 진폭 MSE |
| --- | --- |
| ASM 생성 위상 | 0.040153 |
| DIV2K 실측으로 학습한 CNN 생성 위상 | 0.030530 |
| 학습 모델 + CITL | 0.023177 |

ASM 대비 CNN 단독은 23.97%, 추가 CITL까지는 42.28% 감소했다. 세 채널 모두
개선됐다. 이 수치는 단일 미사용 타깃에 대한 실측이며 DIV2K 전체의 평균 성능이
아니다. 학습과 분리한 시험 이미지 4장의 gain-only 대비 CNN 예측 오차 감소는
R/G/B 약 28.2/33.6/33.5%다. GPU 최대 할당은 약 0.564 GiB였다.

`completion_audit.json`에서 데이터 분리, 원본별 타깃 대응, 저장 RGB 위상과 촬영에
사용한 위상 일치, PNG 인코딩, 최적 checkpoint와 epoch 기록 일치를 확인했다.
테스트 19개가 통과했고, 종료 후 카메라가 원래 설정으로 복원된 것도 확인했다.

## RGB 실행과 학습 과정 기록

`benchtop.rgb_run`은 RGB 각각의 phase를 RGB888 한 장으로 출력하는 CFS 실행 경로다.
638/520/450 nm, +15 mm, 실제 SLM 1080×1920 격자를 사용한다. 초기 green smoke
실험과 별도로 `results/rgb_15mm_20261022/`에 RGB 실측을 저장한다.

1. **측정 준비**: R/G/B 트리거 delay 22/6022/12022 µs에서 채널 응답과 HDMI slot
   대응을 검증한다. 포화를 피하도록 노출을 줄이고, 같은 노출의 비발광 시간 구간
   배경을 촬영한다. 각 파장별 9점 정합 + 별도 2점 검증을 수행한다.
2. **CITL**: 모델로 초기 위상을 만든 뒤 카메라 값을 loss에 넣어 SLM 픽셀 위상을
   업데이트한다. 카메라에 정답 타깃을 촬영시키는 것이 아니라, 디지털 타깃을
   정하고 SLM 출력의 실제 영상을 측정한다. 이번 RGB loss는 flat-white 정규화를
   고정하고 영상마다 최적 gain을 다시 맞추지 않는다.
3. **전파 모델 데이터**: 다양한 타깃으로 위상을 생성해 실제 카메라 진폭과 짝을
   저장한다. 모든 위상을 CITL로 최적화할 필요는 없다. 기본값은 채널당 24쌍이며,
   target ID 기준 16 train / 4 validation / 4 test로 분리한다. CITL demo ID 1000은
   모델 학습에서 제외한다.
4. **CNN 학습**: 위상과 측정은 고정하고 CNN 필터, bias, log gain을 학습한다.
   고정 ASM 진폭을 미리 계산해 batch 1로 학습하므로 메모리와 반복 FFT를 줄인다.
   validation으로 최적 checkpoint를 선택하고, train으로만 fit한 gain-only 기준과
   untouched test에서 비교한다.
5. **새 타깃 검증**: ID 2000의 컬러 타깃을 ASM/학습 모델로 각각 생성한다.
   학습 모델 위상은 실제 RGB888 합성 프레임을 유지한 채 채널별로 한 차례씩
   추가 CITL 보정한다. 각 결과를 3회 재촬영해 비교한다.

실측 중에는 **SLM 디스플레이, Windows 로그인 상태, PC 작동 상태를 유지**한다.
작업용 모니터만 물리적으로 끄더라도 EDID 연결이 끊기면 화면 배치가 바뀔 수 있다.
실측 중에는 켜 두는 편이 확실하다. 저장한 데이터로 CNN만 학습하는 단계에서는
모니터가 필요 없지만 PC 절전은 허용하지 않는다. 현재 실행은 학습 뒤 실측을 다시
수행하므로 전체 과정이 끝날 때까지 디스플레이를 유지해야 한다.

```powershell
# 새 디렉터리에서 전체 RGB 실행
.\.venv\Scripts\python.exe -m benchtop.rgb_run all --out results\rgb_new --steps 80 --pairs 24 --epochs 15

# 단계별 실행. prepare 완료 후 순서대로 실행한다.
.\.venv\Scripts\python.exe -m benchtop.rgb_run prepare --out results\rgb_new
.\.venv\Scripts\python.exe -m benchtop.rgb_run acquire --out results\rgb_new --steps 80 --pairs 24
.\.venv\Scripts\python.exe -m benchtop.rgb_run train --out results\rgb_new --epochs 15
.\.venv\Scripts\python.exe -m benchtop.rgb_run final --out results\rgb_new --steps 80
.\.venv\Scripts\python.exe -m benchtop.rgb_run report --out results\rgb_new
```

이미 완료된 채널의 acquisition/training은 보존하고 건너뛴다. 미완료 acquisition이
남은 경우 자동으로 덮어쓰지 않는다. 실패 기록을 보존하고 새 실행 디렉터리에서
다시 측정한다. 실측 조건이 달라졌다면 예전 calibration을 재사용하지 않는다.

**중간 산출물**:

- `report.html`: 외부 서버 없이 열 수 있는 채널/iteration/epoch 선택형 보고서.
- `<channel>/citl_trace/step_*.npz`: 5 iteration마다 실제 full-resolution 위상,
  실측 진폭, phase gradient ROI. 마지막 측정은 gradient 미계산으로 0을 저장한다.
- `<channel>/model_trace/epoch_*.pt`: 매 epoch 실제 CNN 가중치/bias와 log gain.
- `<channel>/model_trace/epoch_*.npz`: 동일한 validation probe의 ASM/실측/모델
  진폭과 learned amplitude multiplier. 물리적인 SLM 수차 지도와는 다르다.
- `fullcolor/citl_<channel>/`: 다른 채널 위상을 유지한 실제 RGB 프레임 CITL 기록.
- `visualizations/`: 시간에 따라 고정된 색 범위의 PNG, 시험 오차 그래프, index JSON.
- `fullcolor/*_slm_rgb.png`: 실제 출력한 1920×1080 RGB888 위상 이미지.
- `fullcolor/*_camera_amplitude_rgb.npy`: 같은 RGB 프레임에 대한 채널별 gated
  실측을 정합한 진폭. PNG는 고정 flat-white 정규화 후 gamma를 적용한 미리보기다.

현재 학습되는 전파 모델 파라미터는 채널당 866개다. 거리, aperture, LUT, 좌표
정합, 조명 추정치는 고정값이며, 이들을 학습한 것처럼 시각화하지 않는다. Off-slot
배경은 shutter-closed dark current 측정과 다르며, RGB 미리보기는 색도계 기반
CIE 색 보정이나 RGB를 한 장에 장노출한 실측이 아니다.

### RGB 실측 결과 (2026-10-22)

`results/rgb_15mm_20261022/`에서 R/G/B 각각 80회 CITL, 총 72쌍 촬영,
채널별 15 epoch 학습을 수행했다. 새 컬러 타깃에는 RGB 합성 프레임을 유지한 채
각 채널 40회 추가 CITL을 수행했다. 실제 적용 노출은 R/G/B 174/349/698 µs,
좌표 정합 RMS는 원본 센서 기준 0.281/0.310/0.236 px다.

| 실제 출력한 새 컬러 타깃 | RGB 평균 진폭 MSE, 3회 재촬영 평균 |
| --- | --- |
| ASM으로 생성 | 0.056200 |
| 학습한 CNN 전파 모델로 생성 | 0.053760 |
| 학습 모델 + 실제 합성 프레임 CITL | 0.039860 |

ASM 대비 CNN 단독은 약 4.34%, 추가 CITL까지는 약 29.08% 개선됐다.
채널별로 보면 CNN 단독의 R/B 실측 오차는 ASM보다 증가했고 G는 감소했다.
따라서 저장한 위상 분포에서의 forward prediction 개선을 새 위상 최적화의
일반화 성능으로 그대로 해석해서는 안 된다. 추가 CITL 후에는 세 채널 모두
ASM보다 낮은 오차를 얻었다. 작은 합성 타깃 집합에 대한 결과이며 자연 영상
전체나 다중 깊이 성능을 검증한 것은 아니다.

시험 데이터의 gain-only → CNN 진폭 MSE는 R 0.027586 → 0.011597,
G 0.039574 → 0.018707, B 0.020366 → 0.019880이다. 청색 CNN의 gain-only
대비 추가 개선은 약 2.4%로 작다. 실제 출력의 고정 무늬, 비선형 phase LUT,
aperture, 시간 슬롯 사이 영향은 후속 보정 대상으로 남아 있다. 다른 HDMI
슬롯 위상을 바꿨을 때의 영상 변화량은 대상 슬롯 응답의 약 7–10%였으며,
이 수치는 광파워 누설률이나 스펙트럼 혼합률을 직접 측정한 값이 아니다.

전체 단계 중 PyTorch 최대 할당은 0.565 GiB 미만이었다. 저장된 best checkpoint가
해당 epoch의 가중치 기록과 일치하는지, 데이터 split 분리, RGB 위상 PNG 인코딩,
최종 381프레임의 timestamp 증가를 `scripts/audit_rgb.py`로 확인했다.
카메라가 exposure 12999 µs, BayerRG8, gamma on, trigger delay 22 µs,
gain 0 dB로 복원된 것도 재조회했다. 소프트웨어 테스트 15개가 통과했다.

## 최초 녹색 smoke 실행 결과 (2026-10-22)

Windows 잠금을 해제한 후 `results/benchtop_rgb_resume/`에서 전체 smoke 실행을
완료했다. RGB 광원이 모두 켜진 상태에서 녹색 CFS 시간 구간만 측정했다.

| 검증 | 실측 결과 |
| --- | --- |
| 다른 위상에 대한 광학 응답 / 반복 측정 노이즈 | 23.9배 |
| 9점 homography RMS | 원본 센서 기준 0.314 px |
| 별도 2점 위치 검증 | 원본 센서 기준 0.291 / 0.318 px |
| 측정한 릴레이 배율 x / y | 1.0021 / 1.0022, 1:1 설정과 일치 |
| CITL | 30회 최적화, 이전/이후 각각 3회 재촬영 |
| 카메라 진폭 MSE, 3회 평균 | 0.064139 → 0.038517, **39.95% 감소** |
| 실제 촬영 데이터 학습 | 12쌍 중 9 train / 3 validation, 5 epoch |
| 검증 진폭 MSE | 0.027705 → 0.024132, **12.90% 감소** |
| PyTorch GPU 최대 할당 / 예약 | 0.344 / 0.416 GiB, RTX 3090 |

저장한 `forward_model.pt`를 별도 프로세스에서 다시 읽어 validation 3쌍의 동일한
오차를 재현했다. 학습 모델을 통한 위상 gradient와 짧은 위상 최적화도 확인했다.
카메라 노출 12999 µs, BayerRG8, gamma on, trigger delay 22 µs로 원래 설정이
복원된 것을 읽어서 검증했다. 테스트 10개가 통과했다.

실측 비교는 `citl_preview.png`, 재로딩한 모델 검증은
`checkpoint_validation.json`, 종합 기록은 `results/session_20261022.json`에 있다.
이는 녹색 +15 mm 단일 평면의 소규모 동작 확인이며, RGB 전체 또는 여러 깊이의
학습 완료를 뜻하지 않는다. 소수 패턴으로 얻은 수치를 일반화 성능으로 해석하지 않는다.

추가 비교에서는 **학습 데이터만으로 맞춘 진폭 배율 하나**를 ASM에 적용해도
검증 MSE가 0.027705 → 0.024609로 약 11.2% 감소했다. CNN의 0.024132는 이
단순 보정 대비 약 1.9%의 추가 감소다. 따라서 앞의 12.9% 개선은 대부분 전체
진폭 스케일 보정으로 설명되며, 복잡한 위상 수차를 학습했다는 증거로 해석하면
안 된다. 비교 결과는 `results/benchtop_rgb_resume/gain_baseline_review.json`에 있다.

## 현재 설정

`configs/benchtop.json`에 SI 단위로 저장한다.

| 항목 | 설정 |
| --- | --- |
| SLM | HOLOEYE LETO-3-0088, 1920 × 1080, 6.4 µm |
| 센서 | FLIR BFS-U3-123S6C, serial 25222112, 4096 × 3000, 3.45 µm |
| 전파거리 | **릴레이된 SLM 평면에서 +0.015 m** |
| 릴레이 배율 | 1.0, 장비 사양서 기준 초기 가정 |
| 첫 학습 채널 | 녹색 520 nm |
| LUT | BenchtopDisplay의 inverted 8-bit, 0–2π 선형 초기 가정 |
| Fourier aperture | 주파수 격자 모서리 반경 대비 0.5, 기존 GUI 값으로 초기화 |
| 트리거 | Line0 / RisingEdge, delay 6022 µs |
| 노출 | 요청 700 µs, 장치가 적용한 값 698 µs |
| 카메라 영상 | gamma off, BayerRG16, 수동 노출/게인, 프레임 2장 평균 |
| 손실 ROI | 중앙 384 × 640 |

양의 거리는 빛의 진행 방향을 따른다. 4f 릴레이 자체의 전체 광로 길이를 이
15 mm에 더하지 않는다. 좌표 반전/이동/확대는 측정한 homography로 보정한다.
릴레이 배율이 1:1이 아니라면 `relay_magnification`도 변경해야 한다. homography
정렬만으로 전파 커널의 잘못된 픽셀 피치를 보정할 수는 없다. Fourier aperture와
실제 LUT 비선형성은 아직 광학적으로 측정된 값이 아니다.

현재 CFS의 RGB 광원 펄스는 하드웨어에서 발생한다. 약 6 ms의 녹색 시간 구간을
짧게 노출하고 Bayer G 두 샘플을 평균한다. 카메라 좌표는 2×2 Bayer cell 단위
(2048 × 1500)이므로 보정 파일의 sensor 좌표도 원 센서의 절반 해상도다.
색 보간, white balance, gamma를 적용하지 않는다. 여기서 평균은 디지털 영상
binning이며 물리적인 SLM 피치를 바꾸는 연산이 아니다.

## 환경과 실행

Python 3.10, CUDA 12.1 PyTorch 2.5.1 환경을 `uv.lock`으로 고정했다. PySpin은
제조사 배포 wheel이므로 따로 설치한다. 이 PC에서 확인한 SDK/Python 바인딩은
4.3.0.190이다.

```powershell
.\scripts\setup_benchtop.ps1 -PySpinWheel 'C:\Users\khchoi_exp\Downloads\spinnaker_python_4.3\spinnaker_python-4.3.0.190-cp310-cp310-win_amd64.whl'

# 하드웨어를 사용하지 않는 전파/최적화 확인
.\.venv\Scripts\python.exe -m benchtop.run simulate --steps 40

# 실제 표시 및 광학 응답 확인
.\.venv\Scripts\python.exe -m benchtop.run preflight

# 트리거 위치를 바꿨거나 색상 타이밍을 확인할 때
.\.venv\Scripts\python.exe -m benchtop.run scan-timing

# 보정 → 실제 CITL 30회 → 12쌍 촬영 → 소형 전파모델 5 epoch 학습
.\.venv\Scripts\python.exe -m benchtop.run smoke --steps 30 --pairs 12 --epochs 5 --out results\benchtop_smoke

# 화면이 잠겨 있다면, 최대 3분 기다린 뒤 출력이 복구되는 즉시 진행
.\.venv\Scripts\python.exe -m benchtop.run smoke --wait-for-display 180 --out results\benchtop_resume
```

**Windows 잠금을 해제한 상태로 실행해야 한다.** 잠금 화면은 SLM 보조 디스플레이를
검은 화면으로 덮는다. 매 패턴마다 실제 SLM 디스플레이 framebuffer를 요청한
위상과 비교하며, 가려져 있으면 카메라 데이터 수집 전에 중단한다. 런타임 설정은
정상 종료 및 예외 시 카메라에 원래대로 복원하고 표시 창을 닫는다. 레이저 파워,
enable, SLM 펌웨어/LUT를 직접 변경하지 않는다. 프로그램 종료 후 최적 위상은
`best_phase_slm.png` 파일로 남으며 SLM 창이 계속 열려 있는 것은 아니다.

`--wait-for-display`는 실행 중인 프로세스가 지정 시간 동안 기다리는 옵션이다.
기다리는 동안 카메라는 연결하지 않으며, Windows 잠금 해제나 보안 설정 변경을
시도하지 않는다. 제한 시간이 지나도 화면이 가려져 있으면 실패로 종료한다.
촬영 전후에도 화면을 확인하므로 노출 도중 잠긴 경우 해당 측정 전체를 버린다.

`uv sync`를 다시 실행하면 수동 설치한 PySpin이 제거될 수 있으므로 setup 스크립트를
다시 실행한다. 실험 명령은 위와 같이 `.venv` 인터프리터를 직접 사용한다.

분리 실행도 가능하다.

```powershell
.\.venv\Scripts\python.exe -m benchtop.run calibrate
.\.venv\Scripts\python.exe -m benchtop.run citl --steps 50 --out results\citl_50
# 기존 smoke 폴더의 실제 촬영 pairs로 다시 학습
.\.venv\Scripts\python.exe -m benchtop.run train --out results\benchtop_smoke --epochs 10

# 저장된 실제 모델의 validation 오차를 독립적으로 재평가 (카메라 불필요)
.\.venv\Scripts\python.exe -m benchtop.run evaluate --out results\benchtop_rgb_resume

# 학습한 전파 모델을 사용한 위상 최적화 / 실제 CITL
.\.venv\Scripts\python.exe -m benchtop.run simulate --checkpoint results\benchtop_rgb_resume\forward_model.pt --steps 50
.\.venv\Scripts\python.exe -m benchtop.run citl --checkpoint results\benchtop_rgb_resume\forward_model.pt --steps 30
# GPU 여유 메모리가 적다면 동일한 물리 격자로 CPU 실행
.\.venv\Scripts\python.exe -m benchtop.run smoke --device cpu --steps 10 --pairs 8 --epochs 3
```

## 측정과 학습의 검증

- Preflight: 같은 위상 반복 측정의 노이즈와 다른 위상의 광학 응답을 비교한다.
  응답이 노이즈의 4배 이하이면 중단한다.
- 보정: 9개 국소 렌즈 패턴으로 SLM→camera homography를 추정하고 별도 2점으로
  검증한다. RMS 2 camera-cell pixels, 검증 오차 3 pixels 이내여야 저장된다.
- 보정 파일의 장비, 거리, 파장, 노출, LUT 방향 등이 달라지면 재사용을 거부한다.
- 패턴 안정화 0.3초 후 acquisition을 새로 시작하고 첫 프레임을 버린다.
  capture frame ID와 timestamp를 결과에 기록한다.
- CITL loss는 실제 카메라 진폭으로 계산하고 gradient만 전파 모델에서 받는다.
  오차를 평가한 시점의 위상을 복제해 저장하고, 최적 위상 선정 뒤 이전/이후를
  3번씩 다시 측정한다. 한 번의 좋은 노이즈 샘플만으로 개선을 주장하지 않는다.
- 학습 ROI가 센서 바깥에 걸리거나 포화 픽셀이 0.1%를 넘으면 중단한다.

광학 보정 실패 시 파일을 임의의 identity transform으로 대체하지 않는다.
위상 변조, LUT 방향, 편광, 릴레이 배율, 실제 센서 거리를 확인한 후 재실행한다.

## 메모리와 모델 범위

FFT는 실제 1080×1920 SLM 격자에서 2배 zero padding으로 계산한다. complex64를
사용하며 주파수 커널을 한 번만 만든다. SLM 패턴을 작게 resize하여 픽셀 피치를
잘못 해석하는 절약 방법은 사용하지 않는다. 자동 장치 선택은 CUDA 여유 공간이
2 GiB보다 적으면 CPU를 사용하고, 사용자가 `--device cpu`로 지정할 수도 있다.

카메라 데이터로 학습하는 모델은 고정 ASM + ROI에 적용하는 8-channel residual
CNN과 광량 gain이다. batch 1, 한 쌍씩 디스크에서 읽는 방식이며 대형 multi-plane
CNNpropCNN의 재현이나 완성된 3D 학습 모델이 아니다. 고정 센서 하나로 확인 가능한
것은 현재 +15 mm 평면이다. 다른 깊이의 학습에는 해당 위치에서 추가 촬영이 필요하다.

소형 CNN은 실측 flat-field의 저주파 진폭을 광원 초기 추정치로 사용한다. 실제
SLM 평면의 진폭이나 phase LUT를 직접 측정했다고 간주하지 않는다. 학습은 고정
노출에서 측정한 진폭의 MSE를 사용하며, validation pair는 optimizer에 넣지 않는다.
기본 12쌍은 9 train / 3 validation이고 연구용 일반화 성능을 평가하기에는 작다.

## 결과 파일

`results/<run>/`에 config, preflight, 실측 위상/진폭, CITL 곡선과 반복 평가,
프레임 메타데이터, `pairs/manifest.json`, validation 기준 checkpoint
`forward_model.pt`, `forward_training.json`, 비교 PNG가 저장된다. checkpoint에는
모델 종류, 설정, CNN/gain, 광원 추정치가 포함된다. `completed.json`은 전체 요청된
파이프라인이 완료된 경우에만 만들어지고, 실패 원인은 `failure.json`에 남는다.
대용량 캡처/결과와 로컬 `.venv`는 Git에서 제외한다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

실측 최종 기록은 `results/session_20261022.json`, 초기 시뮬레이션 결과는
`results/offline_smoke/`를 참고한다. 시뮬레이션 결과는 실제 CITL 결과와 구분해 저장한다.
