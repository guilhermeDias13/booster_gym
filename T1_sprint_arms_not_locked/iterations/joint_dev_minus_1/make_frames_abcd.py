"""Quatro linhas de 6 frames seguidos, com a letra no canto superior direito.

Mesmo recorte e o mesmo espaçamento da figura de T1_sprint_arms_locked.
O phase usa o vídeo 2026-10-26-04-06-33.mp4.
"""

from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
N_FRAMES = 6
ZOOM = 0.35
ROW_GAP = 28

ROWS = (
    ("a", HERE / "BASE" / "videos" / "2026-10-20-17-35-05.mp4", 1.0),
    ("b", HERE / "dup" / "videos" / "2026-10-20-20-11-09.mp4", 4.0),
    ("c", HERE / "loss" / "videos" / "2026-09-27-04-51-31.mp4", 29.0),
    ("d", HERE / "phase" / "videos" / "2026-10-26-04-06-33.mp4", 1.0),
)


def crop_center(img, frac):
    w, h = img.size
    cw, ch = max(1, round(w * frac)), max(1, round(h * frac))
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def read_six(path, t_start):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"não abriu {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = int(round(t_start * fps))
    if start + N_FRAMES > n:
        start = n - N_FRAMES
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for i in range(N_FRAMES):
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"{path.name} acabou no frame {start + i}")
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(crop_center(img, ZOOM))
    cap.release()
    return frames, start / fps


def label(row, letter):
    size = round(max(64, round(row.height * 0.34)) * 0.75)
    font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", size)
    pad = max(14, size // 5)
    draw = ImageDraw.Draw(row)
    bbox = font.getbbox(letter)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    box_w, box_h = tw + 2 * pad, th + 2 * pad
    x1 = row.width - pad - box_w
    y1 = pad
    draw.rectangle([x1, y1, x1 + box_w - 1, y1 + box_h - 1], fill="white", outline="black")
    draw.text((x1 + pad - bbox[0], y1 + pad - bbox[1]), letter, font=font, fill="black")


def main():
    rows = []
    for letter, path, t_start in ROWS:
        frames, actual = read_six(path, t_start)
        w, h = frames[0].size
        row = Image.new("RGB", (w * N_FRAMES, h))
        for i, frame in enumerate(frames):
            row.paste(frame, (i * w, 0))
        label(row, letter)
        rows.append(row)
        print(f"{letter}  pedido {t_start:.2f}s  inicio {actual:.2f}s  {path.name}")

    width = rows[0].width
    height = sum(r.height for r in rows) + ROW_GAP * (len(rows) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height + ROW_GAP

    png = HERE / "frames_abcd.png"
    pdf = HERE / "frames_abcd.pdf"
    canvas.save(png)
    canvas.save(pdf, "PDF", resolution=150.0)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
