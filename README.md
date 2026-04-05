# chessboard-ar-viewer

체스보드 패턴을 이용한 카메라 포즈 추정 기반 AR 오브젝트 오버레이

## 데모

![demo](assets/ar_output.mp4)

## 사용법

### 1단계 — 캘리브레이션

```bash
python camera-pose-estimation.py --calibrate --video input.mp4
```

### 2단계 — AR 실행

```bash
python camera-pose-estimation.py --run --video input.mp4 --model model.obj
```

## 설정

코드 상단에서 아래 값을 수정하세요.

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `BOARD_COLS` | 체스보드 내부 코너 가로 수 | 7 |
| `BOARD_ROWS` | 체스보드 내부 코너 세로 수 | 7 |
| `SQUARE_SIZE` | 체스보드 한 칸 실제 크기 (미터) | 0.025 |

## 동작 원리

1. `--calibrate` — 동영상에서 체스보드를 자동 추출해 카메라 내부 파라미터(K, dist)를 계산하고 `calibration_result.npz`로 저장
2. `--run` — 매 프레임마다 `cv2.solvePnP`로 카메라 6-DoF 포즈를 추정하고 `.obj` 3D 모델을 체스보드 위에 투영
