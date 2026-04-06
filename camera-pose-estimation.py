import cv2
import numpy as np
import math
import time
import os
import sys
import argparse

CALIB_FILE   = "calibration_result.npz"
BOARD_COLS   = 7     # 내부 코너 가로 수
BOARD_ROWS   = 7        # 내부 코너 세로 수
SQUARE_SIZE  = 0.025    # 한 칸 실제 크기 (미터)
FRAME_SKIP   = 15       # 캘리브레이션 시 몇 프레임마다 추출할지
MIN_SAMPLES  = 20       # 캘리브레이션 최소 샘플 수
DEFAULT_OBJ  = "model.obj"
OUTPUT_FILE  = "ar_output.avi"


# Calibration
def run_calibration(video_path, preview=False):
    board_size = (BOARD_COLS, BOARD_ROWS)
    criteria   = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    obj_pt = np.zeros((BOARD_ROWS * BOARD_COLS, 3), np.float32)
    obj_pt[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)
    obj_pt *= SQUARE_SIZE

    obj_points, img_points = [], []
    img_size = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 동영상을 열 수 없어요: {video_path}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] 동영상: {total}프레임 | 매 {FRAME_SKIP}프레임마다 탐색 중...")

    frame_idx, detected = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, board_size, None)
        if found:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(obj_pt)
            img_points.append(corners2)
            img_size = gray.shape[::-1]
            detected += 1
            print(f"  ✓ {detected}장 수집 (프레임 {frame_idx}/{total})")

            if preview:
                vis = frame.copy()
                cv2.drawChessboardCorners(vis, board_size, corners2, found)
                cv2.putText(vis, f"Collected: {detected}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("Calibration Preview", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()

    if detected < MIN_SAMPLES:
        print(f"[ERROR] 샘플 부족: {detected}장 (최소 {MIN_SAMPLES}장 필요)")
        print("  → FRAME_SKIP을 줄이거나 더 긴 동영상을 사용하세요.")
        sys.exit(1)

    print(f"\n[INFO] {detected}장으로 캘리브레이션 중...")
    _, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None)

    total_err = 0
    for i in range(len(obj_points)):
        proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        total_err += cv2.norm(img_points[i], proj, cv2.NORM_L2) / len(proj)
    print(f"[RESULT] 재투영 오차: {total_err/len(obj_points):.4f} px  (낮을수록 좋음)")
    print(f"[RESULT] Camera Matrix:\n{K}")

    np.savez(CALIB_FILE, camera_matrix=K, dist_coeffs=dist)
    print(f"[SAVED] → {CALIB_FILE}")


# OBJ Loader
def load_obj(path):
    vertices, faces = [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])

    if not vertices:
        raise ValueError(f"꼭짓점 없음: {path}")

    verts = np.array(vertices, dtype=np.float32)
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2
    verts -= center
    scale = verts.max()
    if scale > 0:
        verts /= scale

    if len(faces) > 5000:
        step = len(faces) // 5000
        faces = faces[::step]
    return verts, faces


def transform_obj(verts, scale, cx, cy):
    v = verts.copy()
    v = v[:, [0, 2, 1]]
    v[:, 2] *= -1
    v *= scale
    v[:, 0] += cx
    v[:, 1] += cy
    return v



# AR Drawing
def project(pts, rvec, tvec, K, dist):
    p, _ = cv2.projectPoints(pts.astype(np.float32), rvec, tvec, K, dist)
    return p.reshape(-1, 2).astype(int)



def draw_obj_model(img, rvec, tvec, K, dist, verts, faces, color=(0,200,255)):
    if not faces or len(verts) == 0:
        return
    R, _ = cv2.Rodrigues(rvec)
    cam_dir = R.T @ np.array([0,0,1], dtype=np.float64)
    p2d = project(verts, rvec, tvec, K, dist)

    face_depths = sorted(
        [((R @ verts[f].mean(axis=0) + tvec.flatten())[2], fi)
         for fi, f in enumerate(faces) if max(f) < len(verts)],
        reverse=True
    )
    light = np.array([0.5, -0.5, -1.0])
    light /= np.linalg.norm(light)

    for _, fi in face_depths:
        f = faces[fi]
        v0,v1,v2 = verts[f[0]], verts[f[1]], verts[f[2]]
        n = np.cross(v1-v0, v2-v0)
        norm_val = np.linalg.norm(n)
        if norm_val == 0:
            continue
        n /= norm_val
        if np.dot(n, cam_dir) < 0:
            continue
        shade = max(0.2, np.dot(n, -light))
        fc = tuple(int(c*shade) for c in color)
        pts = np.array([p2d[i] for i in f], np.int32)
        cv2.fillPoly(img,  [pts], fc)
        cv2.polylines(img, [pts], True, tuple(int(c*0.4) for c in color), 1)



# HUD
MODE_NAMES = {"1":"Pyramid","2":"OBJ Model"}

def draw_hud(img, frame_idx, total_frames, model_loaded):
    h, w = img.shape[:2]
    ov = img.copy()
    cv2.rectangle(ov, (0,0),(w,44),(0,0,0),-1)
    img[:] = cv2.addWeighted(ov, 0.5, img, 0.5, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "OBJ Model", (10, 30), font, 0.8, (225, 220, 0), 2)

    pct = frame_idx / max(total_frames, 1) * 100
    cv2.putText(img, f"{frame_idx}/{total_frames}  ({pct:.1f}%)",
                (w-260,30), font, 0.65, (200,255,200), 2)

    bar_w = int(w * frame_idx / max(total_frames, 1))
    cv2.rectangle(img, (0,h-6),(w,h),(50,50,50),-1)
    cv2.rectangle(img, (0,h-6),(bar_w,h),(0,200,100),-1)



# AR Main (video file)
def run_ar(video_path, model_path):
    if not os.path.exists(CALIB_FILE):
        print(f"[ERROR] '{CALIB_FILE}' 없음.")
        print(f"  → 먼저 실행: python ar_chessboard.py --calibrate --video {video_path}")
        sys.exit(1)

    data = np.load(CALIB_FILE)
    K    = data["camera_matrix"]
    dist = data["dist_coeffs"]
    print("[OK] 캘리브레이션 로드 완료")

    raw_verts, raw_faces = None, []
    model_loaded = False
    if os.path.exists(model_path):
        try:
            raw_verts, raw_faces = load_obj(model_path)
            model_loaded = True
            print(f"[OK] OBJ 로드: {len(raw_verts)}개 꼭짓점, {len(raw_faces)}개 면")
        except Exception as e:
            print(f"[WARN] OBJ 로드 실패: {e}")
    else:
        print(f"[WARN] '{model_path}' 없음 → 모드 5는 빈 화면으로 표시됩니다.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 동영상을 열 수 없어요: {video_path}")
        sys.exit(1)

    fps_video    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] 동영상: {width}x{height}, {fps_video:.1f}fps, {total_frames}프레임")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out    = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps_video, (width, height))

    board_size = (BOARD_COLS, BOARD_ROWS)
    obj_pts = np.zeros((BOARD_ROWS * BOARD_COLS, 3), np.float32)
    obj_pts[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)
    obj_pts *= SQUARE_SIZE
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    s  = SQUARE_SIZE
    cx = (BOARD_COLS // 2) * s
    cy = (BOARD_ROWS // 2) * s
    model_scale = s * 3

    print(f"[INFO] AR 처리 시작... → {OUTPUT_FILE} 로 저장됩니다\n")

    frame_idx  = 0
    detected_n = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.undistort(frame, K, dist)

        frame_idx += 1
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, board_size, None)

        if found:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            ok, rvec, tvec = cv2.solvePnP(obj_pts, corners2, K, dist)

            if ok:
                detected_n += 1
                cv2.drawChessboardCorners(frame, board_size, corners2, found)
                t = frame_idx / fps_video

                if model_loaded:
                    vw = transform_obj(raw_verts, model_scale, cx, cy)
                    draw_obj_model(frame, rvec, tvec, K, dist, vw, raw_faces)

        draw_hud(frame, frame_idx, total_frames, model_loaded)
        out.write(frame)

        if frame_idx % 100 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / frame_idx * (total_frames - frame_idx)
            print(f"  진행: {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)"
                  f"  인식: {detected_n}프레임  ETA: {eta:.0f}초")

    cap.release()
    out.release()

    elapsed = time.time() - start_time
    print(f"\n[DONE] 완료! ({elapsed:.1f}초 소요)")
    print(f"[SAVED] → {OUTPUT_FILE}")
    print(f"[INFO] 총 {frame_idx}프레임 중 {detected_n}프레임 체스보드 인식 "
          f"({detected_n/max(frame_idx,1)*100:.1f}%)")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AR Chessboard Viewer")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--calibrate", action="store_true", help="캘리브레이션 모드")
    group.add_argument("--run",       action="store_true", help="AR 실행 모드")
    parser.add_argument("--video",   required=True,        help="입력 동영상 경로 (.mp4 등)")
    parser.add_argument("--model",   default=DEFAULT_OBJ,  help=".obj 모델 경로")
    parser.add_argument("--preview", action="store_true",  help="캘리브레이션 미리보기")
    args = parser.parse_args()

    if args.calibrate:
        run_calibration(args.video, preview=args.preview)
    elif args.run:
        run_ar(args.video, args.model)
