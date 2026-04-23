"""
PTB-XL → ECG Image Generator
==============================
Fetches real PTB-XL records and renders them as realistic ECG paper images
so you can feed them back into ecg_digitizer.py and measure round-trip accuracy.

Renders three image types to stress-test the digitizer:
  1. clean   — perfect scan (no noise, ideal grid)
  2. photo   — simulated phone photo (perspective skew, lighting gradient, blur)
  3. monitor — green-on-black ICU monitor style

Usage
-----
  python generate_ecg_images.py                   # 5 random test-set records, all styles
  python generate_ecg_images.py --n 20            # 20 records
  python generate_ecg_images.py --ecg_id 1 2 3   # specific ECG IDs
  python generate_ecg_images.py --style clean     # one style only
  python generate_ecg_images.py --roundtrip       # auto-run digitizer + plot correlation

Output
------
  ecg_images/
  ├── ecg_0001_clean.png
  ├── ecg_0001_photo.png
  ├── ecg_0001_monitor.png
  ├── ecg_0001_original.npy    ← ground truth (5000,12) array
  └── roundtrip_report.png     ← correlation plot (if --roundtrip)

Requirements
------------
  pip install wfdb numpy pandas matplotlib opencv-python scipy scikit-image
"""

import os
import ast
import argparse
import numpy as np
import pandas as pd
import wfdb
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "./"
OUT_DIR     = "ecg_images"
LEAD_ORDER  = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
LEAD_LAYOUT = [
    ["I",   "II",  "III"],
    ["aVR", "aVL", "aVF"],
    ["V1",  "V2",  "V3"],
    ["V4",  "V5",  "V6"],
]
DURATION_S = 10
FS         = 500


# ═══════════════════════════════════════════════════════════════════════════════
#  Load PTB-XL metadata
# ═══════════════════════════════════════════════════════════════════════════════

def load_metadata():
    Y = pd.read_csv(DATA_PATH + "ptbxl_database.csv", index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)
    return Y


def fetch_record(row) -> np.ndarray:
    """Returns (5000, 12) float32 in mV."""
    signal, _ = wfdb.rdsamp(DATA_PATH + row["filename_hr"])
    return signal.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  Core renderer
# ═══════════════════════════════════════════════════════════════════════════════

def render_ecg_figure(signal: np.ndarray, ecg_id: int, style: str) -> plt.Figure:
    """Render a 12-lead ECG in standard 4-row x 3-col layout."""

    if style == "monitor":
        bg_color    = "#000000"
        grid_minor  = "#1a1a1a"
        grid_major  = "#2a2a2a"
        trace_color = "#00cc44"
        text_color  = "#00cc44"
        label_color = "#888888"
    else:   # clean / photo — paper ECG
        bg_color    = "#fff5f5"
        grid_minor  = "#ffcccc"
        grid_major  = "#ff8888"
        trace_color = "#111111"
        text_color  = "#111111"
        label_color = "#333333"

    fig = plt.figure(figsize=(20, 13), facecolor=bg_color, dpi=120)
    gs  = gridspec.GridSpec(4, 3, figure=fig,
                             left=0.05, right=0.97,
                             top=0.90,  bottom=0.05,
                             hspace=0.55, wspace=0.25)

    time = np.linspace(0, DURATION_S, signal.shape[0])

    for row_i, row_leads in enumerate(LEAD_LAYOUT):
        for col_i, lead in enumerate(row_leads):
            ax = fig.add_subplot(gs[row_i, col_i])
            lead_idx = LEAD_ORDER.index(lead)
            sig = signal[:, lead_idx]

            # Grid
            ax.set_facecolor(bg_color)
            ax.minorticks_on()
            ax.grid(which="minor", color=grid_minor, linewidth=0.4, linestyle="-")
            ax.grid(which="major", color=grid_major, linewidth=0.9, linestyle="-")
            ax.set_xticks(np.arange(0, DURATION_S + 0.001, 0.2))

            y_range  = max(2.0, float(np.ptp(sig)) + 0.5)
            y_center = float(sig.mean())
            ax.set_yticks(np.arange(
                np.floor((y_center - y_range/2) * 2) / 2,
                np.ceil( (y_center + y_range/2) * 2) / 2 + 0.01,
                0.5
            ))
            ax.set_xlim(0, DURATION_S)
            ax.set_ylim(y_center - y_range/2, y_center + y_range/2)

            # 1 mV calibration pulse at start
            cal_x = [0, 0, 0.1, 0.1, 0.2, 0.2]
            cal_y = [y_center, y_center+1.0, y_center+1.0, y_center, y_center, y_center]
            ax.plot(cal_x, cal_y, color=trace_color, linewidth=1.2, zorder=3)

            # Waveform
            ax.plot(time, sig, color=trace_color, linewidth=0.9, zorder=3)

            # Labels
            ax.set_title(lead, color=text_color, fontsize=11,
                         fontweight="bold", pad=3, loc="left")
            ax.tick_params(colors=label_color, labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor(grid_major)
                spine.set_linewidth(0.8)

            if row_i == 3:
                ax.set_xlabel("seconds", color=label_color, fontsize=7)
            else:
                ax.set_xticklabels([])

    fig.text(0.5, 0.96,
             f"12-Lead ECG  |  ECG ID: {ecg_id}  |  25mm/s  10mm/mV",
             ha="center", color=text_color, fontsize=10, fontweight="bold")
    fig.text(0.5, 0.93,
             "PTB-XL Test Record — Generated for digitizer validation",
             ha="center", color=label_color, fontsize=8)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  Photo simulation
# ═══════════════════════════════════════════════════════════════════════════════

def apply_photo_effect(img: np.ndarray) -> np.ndarray:
    """Simulate phone photo: skew, lighting gradient, blur, JPEG artifacts."""
    h, w = img.shape[:2]

    # Slight rotation
    angle = float(np.random.uniform(-3, 3))
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Perspective distortion
    margin = int(w * 0.02)
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = np.float32([
        [int(np.random.randint(0, margin)), int(np.random.randint(0, margin))],
        [w - int(np.random.randint(0, margin)), int(np.random.randint(0, margin))],
        [w - int(np.random.randint(0, margin)), h - int(np.random.randint(0, margin))],
        [int(np.random.randint(0, margin)), h - int(np.random.randint(0, margin))],
    ])
    M2  = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img, M2, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Lighting gradient (brighter centre)
    Y_g, X_g = np.mgrid[0:h, 0:w].astype(np.float32)
    dist     = np.sqrt(((X_g - w/2)/(w/2))**2 + ((Y_g - h/2)/(h/2))**2)
    gradient = np.clip(1.0 - 0.25 * dist, 0.6, 1.0)
    img = (img.astype(np.float32) * gradient[:, :, np.newaxis]).clip(0,255).astype(np.uint8)

    # Blur
    ksize = int(np.random.choice([1, 3]))
    if ksize > 1:
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # JPEG compression
    quality = int(np.random.randint(60, 80))
    _, buf  = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    img     = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    return img


# ═══════════════════════════════════════════════════════════════════════════════
#  Save helpers
# ═══════════════════════════════════════════════════════════════════════════════

def fig_to_cv2(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # RGBA
    return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)


def save_record_images(ecg_id: int, signal: np.ndarray, styles: list):
    os.makedirs(OUT_DIR, exist_ok=True)

    npy_path = os.path.join(OUT_DIR, f"ecg_{ecg_id:04d}_original.npy")
    np.save(npy_path, signal)

    paths = {}
    for style in styles:
        fig = render_ecg_figure(signal, ecg_id, style)
        img = fig_to_cv2(fig)
        plt.close(fig)

        if style == "photo":
            img = apply_photo_effect(img)

        out_path = os.path.join(OUT_DIR, f"ecg_{ecg_id:04d}_{style}.png")
        cv2.imwrite(out_path, img)
        paths[style] = out_path
        print(f"  Saved: {out_path}")

    return paths, npy_path


# ═══════════════════════════════════════════════════════════════════════════════
#  Round-trip validation
# ═══════════════════════════════════════════════════════════════════════════════

def run_roundtrip(generated: list):
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("ecg_digitizer", "ecg_digitizer.py")
        digitizer = importlib.util.load_from_spec(spec)
        spec.loader.exec_module(digitizer)
    except Exception as e:
        print(f"\n[Round-trip] Cannot import ecg_digitizer.py: {e}")
        return

    results = []

    for ecg_id, style, img_path, npy_path in generated:
        orig = np.load(npy_path)
        print(f"  Digitizing {img_path} ...")
        try:
            recovered = digitizer.digitize(img_path, manual=False)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        for i, lead in enumerate(LEAD_ORDER):
            o = orig[:, i];      o = (o - o.mean()) / (o.std() + 1e-8)
            r = recovered[:, i]; r = (r - r.mean()) / (r.std() + 1e-8)
            corr = float(np.corrcoef(o, r)[0, 1])
            results.append((ecg_id, style, lead, corr))

    if not results:
        print("No round-trip results.")
        return

    df = pd.DataFrame(results, columns=["ecg_id","style","lead","correlation"])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#0d1117")
    fig.suptitle("Round-trip Correlation: Original vs Digitized", color="white", fontsize=13)

    colors = {"clean":"#00e5ff", "photo":"#ff9f43", "monitor":"#69ff47"}
    for ax, style in zip(axes, ["clean","photo","monitor"]):
        sub = df[df["style"] == style]
        if sub.empty:
            ax.set_visible(False)
            continue
        means = sub.groupby("lead")["correlation"].mean().reindex(LEAD_ORDER)
        ax.bar(LEAD_ORDER, means, color=colors.get(style,"#aaa"), edgecolor="#333", linewidth=0.5)
        ax.set_facecolor("#161b22")
        ax.set_title(f"{style.capitalize()} style", color="white", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.9, color="#ff4444", linewidth=1, linestyle="--")
        ax.set_xlabel("Lead", color="#888", fontsize=9)
        ax.set_ylabel("Pearson r", color="#888", fontsize=9)
        ax.tick_params(colors="#888", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.text(0.5, 0.97, f"Mean r = {means.mean():.3f}",
                transform=ax.transAxes, ha="center", va="top", color="white", fontsize=10)

    plt.tight_layout()
    report_path = os.path.join(OUT_DIR, "roundtrip_report.png")
    plt.savefig(report_path, dpi=120, bbox_inches="tight", facecolor="#0d1117")
    print(f"\nRound-trip report saved → {report_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",         type=int, default=5)
    parser.add_argument("--ecg_id",    type=int, nargs="+")
    parser.add_argument("--style",     choices=["clean","photo","monitor","all"], default="all")
    parser.add_argument("--roundtrip", action="store_true")
    args = parser.parse_args()

    styles = ["clean","photo","monitor"] if args.style == "all" else [args.style]

    print("Loading PTB-XL metadata...")
    Y = load_metadata()

    if args.ecg_id:
        selected = Y.loc[Y.index.isin(args.ecg_id)]
    else:
        test_set = Y[Y["strat_fold"] == 10]
        selected = test_set.sample(n=min(args.n, len(test_set)), random_state=42)

    print(f"Rendering {len(selected)} records | styles: {styles}\n")

    generated = []
    for ecg_id, row in selected.iterrows():
        print(f"ECG {ecg_id} | {row['filename_hr']}")
        signal = fetch_record(row)
        paths, npy_path = save_record_images(ecg_id, signal, styles)
        for style, img_path in paths.items():
            generated.append((ecg_id, style, img_path, npy_path))

    print(f"\nDone — {len(generated)} images in ./{OUT_DIR}/")
    print("\nExample usage:")
    sample = f"ecg_{list(selected.index)[0]:04d}"
    print(f"  python ecg_digitizer.py --image {OUT_DIR}/{sample}_clean.png --predict")
    print(f"  python ecg_digitizer.py --image {OUT_DIR}/{sample}_photo.png --manual --predict")

    if args.roundtrip:
        print("\nRunning round-trip validation...")
        run_roundtrip(generated)
