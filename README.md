# chessboard-ar-viewer

체스보드 패턴을 이용한 카메라 포즈 추정 기반 AR 오브젝트 오버레이

## 데모

![demo](https://github.com/user-attachments/assets/a819d668-062b-41cd-865e-0ffc3a61a048)

## 캘리브레이션 결과

| 항목 | 값 |
|------|-----|
| 사용 샘플 수 | 53장 |
| 재투영 오차 | 0.0498 px |

**Camera Matrix (K)**
```
[[868.39  0.      548.78]
 [  0.   868.64   952.17]
 [  0.     0.       1.  ]]
```

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
