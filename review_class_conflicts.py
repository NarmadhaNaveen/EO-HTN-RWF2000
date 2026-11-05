import os
import cv2
import numpy as np

root = "RWF-2000"
names = [
    "EFv961C5RgY_0.avi",
    "MxkTo5F-nvU_0.avi",
    "6Rl7q_kXYbg_0.avi",
    "JECBfnp2ZXc_0.avi",
    "1MVS2QPWbHc_0.avi",
    "39BFeYnbu-I_0.avi",
    "7gLKFV5voOg_0.avi",
    "A7FCl8G35Cs_0.avi"
]

width, height = 320, 180
rows = []

def sample_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open " + path)

    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    positions = np.linspace(0, max(0, count - 1), 6).astype(int)   
    frames = []

    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (width, height))
        frames.append(frame)

    cap.release()
    return frames, count, count / fps if fps > 0 else 0

for name in names:
    print("Inspecting:", name)

    fight_path = os.path.join(root, "val", "Fight", name)
    normal_path = os.path.join(root, "val", "NonFight", name)

    if not os.path.isfile(fight_path) or not os.path.isfile(normal_path):
        print("Pair missing; skipped.")
        continue

    ff, fc, fd = sample_video(fight_path)
    nf, nc, nd = sample_video(normal_path)

    print(f"Fight: {fc} frames, {fd:.2f} seconds")
    print(f"NonFight: {nc} frames, {nd:.2f} seconds")

    fight_row = np.vstack([
    np.hstack(ff[:3]),
    np.hstack(ff[3:])
    ])
    normal_row = np.vstack([
    np.hstack(nf[:3]),
    np.hstack(nf[3:])
    ])

    header = np.zeros((40, width * 3, 3), dtype=np.uint8)
    cv2.putText(header, name, (8, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    fight_label = np.zeros((30, width * 3, 3), dtype=np.uint8)
    cv2.putText(fight_label, "FIGHT folder", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    normal_label = np.zeros((30, width * 3, 3), dtype=np.uint8)
    cv2.putText(normal_label, "NONFIGHT folder", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    rows.append(np.vstack([
        header, fight_label, fight_row, normal_label, normal_row
    ]))

if rows:
    sheet = np.vstack(rows)
    os.makedirs("outputs", exist_ok=True)
    output = "outputs/class_conflict_review.jpg"
    cv2.imwrite(output, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("\nSaved:", os.path.abspath(output))
else:
    print("No matching pairs were processed.")
