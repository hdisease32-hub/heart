"""
ECG Image Digitizer
====================
Converts a photo/scan/screenshot of a standard 12-lead ECG into a
(5000, 12) float32 numpy array — the exact format PTB-XL / your model expects.

Pipeline
--------
  Image → Deskew → Grid detection → Perspective warp →
  Per-lead ROI crop → Waveform trace → Resample to 5000 pts →
  Normalize → save as .npy  +  verification plot

Usage
-----
  # Fully automatic (good for clean scans / EHR screenshots)
  python ecg_digitizer.py --image my_ecg.jpg

  # Manual alignment UI (recommended for photos / crooked scans)
  python ecg_digitizer.py --image my_ecg.jpg --manual

  # Run output directly through your model
  python ecg_digitizer.py --image my_ecg.jpg --predict

Requirements
------------
  pip install opencv-python numpy scipy matplotlib scikit-image torch wfdb
  (torch only needed for --predict flag)
"""

import argparse
import sys
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import resample, medfilt
from scipy.ndimage import gaussian_filter1d
from skimage.transform import hough_line, hough_line_peaks

# ── Lead layout (standard 12-lead, 4 rows × 3 cols) ──────────────────────────
# Row 0: I   II   III
# Row 1: aVR aVL  aVF
# Row 2: V1  V2   V3
# Row 3: V4  V5   V6
LEAD_LAYOUT = [
    ["I",   "II",  "III"],
    ["aVR", "aVL", "aVF"],
    ["V1",  "V2",  "V3"],
    ["V4",  "V5",  "V6"],
]
LEAD_ORDER = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
TARGET_LEN = 5000   # samples expected by model (10s @ 500Hz)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Load & deskew
# ═══════════════════════════════════════════════════════════════════════════════

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    # Resize to a standard width for consistent processing (keep aspect ratio)
    h, w = img.shape[:2]
    target_w = 2400
    if w != target_w:
        scale = target_w / w
        img = cv2.resize(img, (target_w, int(h * scale)), interpolation=cv2.INTER_CUBIC)
    return img


def deskew(img: np.ndarray) -> np.ndarray:
    """Detect dominant horizontal angle via Hough and rotate to correct skew."""
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # Only look at a horizontal strip in the middle (avoids border noise)
    h, w = edges.shape
    strip = edges[h//3 : 2*h//3, :]
    tested, angles, acc = hough_line(strip)
    _, peak_angles, _ = hough_line_peaks(tested, angles, acc, num_peaks=20)
    if len(peak_angles) == 0:
        return img
    # Filter to near-horizontal lines (within ±10°)
    horiz = peak_angles[np.abs(np.degrees(peak_angles)) < 10]
    if len(horiz) == 0:
        return img
    angle_deg = np.degrees(np.median(horiz))
    # Don't correct tiny angles (< 0.5°) — resampling artifacts not worth it
    if abs(angle_deg) < 0.5:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2a — Manual alignment UI (click 4 corners of the ECG grid)
# ═══════════════════════════════════════════════════════════════════════════════

def manual_align(img: np.ndarray) -> np.ndarray:
    """
    Opens an OpenCV window. User clicks 4 corners of the ECG grid:
      Top-Left → Top-Right → Bottom-Right → Bottom-Left
    Returns perspective-corrected image.
    """
    points = []
    clone  = img.copy()

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(clone, (x, y), 8, (0, 255, 0), -1)
            labels = ["TL", "TR", "BR", "BL"]
            cv2.putText(clone, labels[len(points)-1], (x+10, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow("ECG Aligner — click TL, TR, BR, BL  |  press ENTER when done", clone)

    # Scale down for display if the image is too large for the screen
    disp = clone.copy()
    dh, dw = disp.shape[:2]
    max_disp = 1000
    scale = min(max_disp / dw, max_disp / dh, 1.0)
    disp_small = cv2.resize(disp, (int(dw*scale), int(dh*scale)))

    win = "ECG Aligner — click TL, TR, BR, BL  |  press ENTER when done"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, int(dw*scale), int(dh*scale))
    cv2.imshow(win, disp_small)

    # Re-map clicks from display scale to full-res
    real_points = []

    def click_scaled(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(real_points) < 4:
            rx, ry = int(x / scale), int(y / scale)
            real_points.append((rx, ry))
            cv2.circle(clone, (rx, ry), 8, (0, 255, 0), -1)
            labels = ["TL", "TR", "BR", "BL"]
            cv2.putText(clone, labels[len(real_points)-1], (rx+10, ry-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
            cv2.imshow(win, cv2.resize(clone, (int(dw*scale), int(dh*scale))))

    cv2.setMouseCallback(win, click_scaled)
    print("\n[Manual Alignment]")
    print("Click the 4 corners of the ECG grid in order:")
    print("  1. Top-Left  2. Top-Right  3. Bottom-Right  4. Bottom-Left")
    print("Press ENTER when done.\n")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and len(real_points) == 4:   # ENTER
            break
        if key == 27:   # ESC
            cv2.destroyAllWindows()
            sys.exit("Aborted by user.")

    cv2.destroyAllWindows()

    if len(real_points) != 4:
        raise ValueError("Need exactly 4 corner points.")

    src = np.float32(real_points)
    # Output size: standard ECG aspect ratio (roughly 4:3 wide)
    out_w, out_h = 2400, 1200
    dst = np.float32([[0,0],[out_w,0],[out_w,out_h],[0,out_h]])
    M   = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h))


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2b — Automatic grid detection & crop
# ═══════════════════════════════════════════════════════════════════════════════

def auto_crop_grid(img: np.ndarray) -> np.ndarray:
    """
    Finds the ECG grid region automatically by:
    1. Suppressing the red/pink grid lines (they dominate in paper ECGs)
    2. Finding the bounding box of the content area
    """
    # Convert to HSV to isolate the red grid
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Mask red grid lines (paper ECG)
    mask_r1 = cv2.inRange(hsv, (0,  50,  50), (15, 255, 255))
    mask_r2 = cv2.inRange(hsv, (165,50,  50), (180,255, 255))
    red_mask = cv2.bitwise_or(mask_r1, mask_r2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold to find dark content (waveform + text) on light background
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    # Remove red-grid pixels from binary so we don't box on grid lines
    content = cv2.bitwise_and(binary, cv2.bitwise_not(red_mask))

    # Find bounding box of content
    coords = cv2.findNonZero(content)
    if coords is None:
        return img   # fallback: return as-is
    x, y, w, h = cv2.boundingRect(coords)
    # Add small margin
    pad = 20
    x  = max(0, x - pad)
    y  = max(0, y - pad)
    w  = min(img.shape[1] - x, w + 2*pad)
    h  = min(img.shape[0] - y, h + 2*pad)
    return img[y:y+h, x:x+w]


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Detect grid line spacing (→ pixels per mm)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_grid_spacing(img: np.ndarray) -> float:
    """
    Detect the small-grid spacing (1mm squares) by finding the dominant
    frequency in the horizontal gradient of the red channel.
    Returns pixels_per_mm (float).
    """
    # Red channel is most informative for paper ECGs
    # For monitor screenshots, use inverse of value channel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, (0,30,30),(15,255,255)) | \
               cv2.inRange(hsv, (165,30,30),(180,255,255))

    # Use a horizontal slice through the middle
    h, w = red_mask.shape
    strip = red_mask[h//2 - 20 : h//2 + 20, :].astype(np.float32)
    profile = strip.mean(axis=0)

    # FFT to find dominant spatial frequency
    fft    = np.abs(np.fft.rfft(profile - profile.mean()))
    freqs  = np.fft.rfftfreq(len(profile))
    # Ignore DC and very low frequencies
    fft[:3] = 0
    dominant_freq = freqs[np.argmax(fft)]
    if dominant_freq <= 0:
        # Fallback: assume 300 DPI scan → 1mm ≈ 11.8 px
        return 11.8
    px_per_mm = 1.0 / dominant_freq
    # Sanity check: 1mm grid at typical scan/photo DPI is 6–25 px
    px_per_mm = np.clip(px_per_mm, 6.0, 25.0)
    return float(px_per_mm)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Isolate waveform (remove grid, keep trace)
# ═══════════════════════════════════════════════════════════════════════════════

def isolate_waveform(img: np.ndarray) -> np.ndarray:
    """
    Returns a grayscale image where the ECG trace is white, background black.
    Handles: paper (black trace on red grid), monitor (green on black),
    EHR (dark trace on white).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect image type from dominant colors
    mean_sat = hsv[:,:,1].mean()
    mean_val = hsv[:,:,2].mean()

    # --- Paper ECG: bright background, red grid, dark waveform ---
    if mean_val > 150:
        # Suppress red grid
        red_mask = cv2.inRange(hsv,(0,40,100),(15,255,255)) | \
                   cv2.inRange(hsv,(165,40,100),(180,255,255))
        inpainted = cv2.inpaint(img, red_mask, 3, cv2.INPAINT_NS)
        gray2 = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
        # Dark trace on light background → invert after threshold
        _, waveform = cv2.threshold(gray2, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- Monitor ECG: dark background, colored waveform ---
    elif mean_val < 80:
        # Green waveform on black background
        green_mask = cv2.inRange(hsv,(40,50,50),(85,255,255))
        # Fallback: anything bright on dark background
        _, bright  = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
        waveform   = cv2.bitwise_or(green_mask, bright)

    # --- EHR screenshot: white/grey background, dark blue/black trace ---
    else:
        _, waveform = cv2.threshold(gray, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Thin the trace to 1px for accurate centroid extraction
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))
    waveform = cv2.morphologyEx(waveform, cv2.MORPH_CLOSE, kernel)
    return waveform


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Crop per-lead ROI and extract 1D signal
# ═══════════════════════════════════════════════════════════════════════════════

def crop_lead_rois(waveform_img: np.ndarray) -> dict:
    """
    Divide the waveform image into 4 rows × 3 cols and return a dict
    {lead_name: waveform_strip (2D grayscale)}.
    """
    h, w = waveform_img.shape
    row_h = h // 4
    col_w = w // 3

    rois = {}
    for r, row in enumerate(LEAD_LAYOUT):
        for c, lead in enumerate(row):
            y0, y1 = r * row_h, (r+1) * row_h
            x0, x1 = c * col_w, (c+1) * col_w
            rois[lead] = waveform_img[y0:y1, x0:x1]
    return rois


def extract_signal_from_strip(strip: np.ndarray) -> np.ndarray:
    """
    From a binary (white trace on black) strip, extract a 1D voltage signal
    by finding the y-centroid of the trace at each x column.
    Returns signal in normalised units (0 = baseline, ±1 = ±1mV equivalent).
    """
    h, w = strip.shape
    signal = np.full(w, np.nan)

    for x in range(w):
        col = strip[:, x]
        if col.max() == 0:
            continue  # no trace in this column
        # Centroid of white pixels
        ys = np.where(col > 0)[0]
        signal[x] = ys.mean()

    # Interpolate NaN gaps (calibration pulse, missing pixels)
    nans = np.isnan(signal)
    if nans.all():
        return np.zeros(w)
    idx  = np.arange(w)
    signal[nans] = np.interp(idx[nans], idx[~nans], signal[~nans])

    # Invert: in image coords y increases downward, voltage increases upward
    signal = h - signal

    # Remove slow baseline wander (high-pass via median subtraction)
    baseline = medfilt(signal, kernel_size=min(201, w // 5 * 2 + 1))
    signal   = signal - baseline

    return signal.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Resample to 5000 pts and normalise per-lead
# ═══════════════════════════════════════════════════════════════════════════════

def resample_lead(signal: np.ndarray, target: int = TARGET_LEN) -> np.ndarray:
    """Resample to exactly `target` points using scipy FFT resample."""
    if len(signal) == target:
        return signal
    resampled = resample(signal, target).astype(np.float32)
    return resampled


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Per-lead z-score normalisation — same as your training pipeline."""
    mean = signal.mean()
    std  = signal.std() + 1e-8
    return (signal - mean) / std


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — Assemble (5000, 12) array
# ═══════════════════════════════════════════════════════════════════════════════

def assemble_output(lead_signals: dict) -> np.ndarray:
    """
    lead_signals: {lead_name: np.ndarray shape (5000,)}
    Returns: np.ndarray shape (5000, 12) — same layout as PTB-XL wfdb records
    """
    out = np.zeros((TARGET_LEN, 12), dtype=np.float32)
    for i, lead in enumerate(LEAD_ORDER):
        sig = lead_signals.get(lead, np.zeros(TARGET_LEN))
        out[:, i] = normalize_signal(resample_lead(sig))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 8 — Verification plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_verification(ecg_array: np.ndarray, output_path: str = "ecg_digitized_verify.png"):
    """Plot all 12 leads in the standard layout for visual verification."""
    fig = plt.figure(figsize=(20, 14), facecolor="#0d1117")
    fig.suptitle("Digitized ECG — Verification Plot\n(visually compare against original image)",
                 color="white", fontsize=14, y=0.98)

    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.6, wspace=0.3)
    colors = ["#00e5ff","#00e5ff","#00e5ff",
              "#69ff47","#69ff47","#69ff47",
              "#ff9f43","#ff9f43","#ff9f43",
              "#ff6b9d","#ff6b9d","#ff6b9d"]

    time = np.linspace(0, 10, TARGET_LEN)

    for idx, lead in enumerate(LEAD_ORDER):
        r, c = divmod(idx, 3)
        ax = fig.add_subplot(gs[r, c])
        ax.plot(time, ecg_array[:, idx], color=colors[idx], linewidth=0.8)
        ax.set_facecolor("#161b22")
        ax.set_title(lead, color="white", fontsize=11, pad=4)
        ax.set_xlabel("s", color="#666", fontsize=8)
        ax.tick_params(colors="#555", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.axhline(0, color="#333", linewidth=0.5, linestyle="--")
        ax.set_xlim(0, 10)

    plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="#0d1117")
    print(f"Verification plot saved → {output_path}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def digitize(image_path: str, manual: bool = False) -> np.ndarray:
    """
    Main entry point.
    Returns ecg_array of shape (5000, 12) ready for the model.
    """
    print(f"\n[1/6] Loading image: {image_path}")
    img = load_image(image_path)

    print("[2/6] Deskewing...")
    img = deskew(img)

    if manual:
        print("[3/6] Manual alignment UI — follow on-screen instructions...")
        img = manual_align(img)
    else:
        print("[3/6] Auto-cropping grid region...")
        img = auto_crop_grid(img)

    print("[4/6] Isolating waveform trace...")
    waveform_img = isolate_waveform(img)

    print("[5/6] Extracting per-lead signals...")
    rois   = crop_lead_rois(waveform_img)
    lead_signals = {}
    for lead, strip in rois.items():
        sig = extract_signal_from_strip(strip)
        lead_signals[lead] = sig
        print(f"      {lead:4s} → {len(sig)} samples extracted")

    print("[6/6] Resampling to 5000 pts & normalising...")
    ecg_array = assemble_output(lead_signals)
    print(f"      Output shape: {ecg_array.shape}  dtype: {ecg_array.dtype}")

    return ecg_array


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTIONAL — run straight into your model
# ═══════════════════════════════════════════════════════════════════════════════

def predict_from_array(ecg_array: np.ndarray):
    """Run the digitized ECG through your trained ECGResNet."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    CLASSES    = ["NORM", "MI", "STTC", "CD", "HYP"]
    THRESHOLDS = {"NORM": 0.55, "MI": 0.65, "STTC": 0.55, "CD": 0.70, "HYP": 0.55}
    MODEL_PATH = "checkpoints/cardi_model.pt"

    class ResidualBlock(nn.Module):
        def __init__(self, in_c, out_c, k=15, s=1):
            super().__init__()
            self.c1   = nn.Conv1d(in_c, out_c, k, s, k//2, bias=False)
            self.b1   = nn.BatchNorm1d(out_c)
            self.c2   = nn.Conv1d(out_c, out_c, k, 1, k//2, bias=False)
            self.b2   = nn.BatchNorm1d(out_c)
            self.skip = nn.Sequential() if (s==1 and in_c==out_c) else \
                        nn.Sequential(nn.Conv1d(in_c,out_c,1,s,bias=False), nn.BatchNorm1d(out_c))
        def forward(self, x):
            return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.skip(x))

    class ECGResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem   = nn.Sequential(nn.Conv1d(12,64,15,2,7,bias=False),nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(3,2,1))
            self.s1     = nn.Sequential(ResidualBlock(64,64),  ResidualBlock(64,64))
            self.s2     = nn.Sequential(ResidualBlock(64,128,s=2), ResidualBlock(128,128))
            self.s3     = nn.Sequential(ResidualBlock(128,256,s=2),ResidualBlock(256,256))
            self.s4     = nn.Sequential(ResidualBlock(256,512,s=2),ResidualBlock(512,512))
            self.pool   = nn.AdaptiveAvgPool1d(1)
            self.drop   = nn.Dropout(0.5)
            self.fc     = nn.Linear(512, 5)
        def forward(self, x):
            return self.fc(self.drop(self.pool(self.s4(self.s3(self.s2(self.s1(self.stem(x)))))).squeeze(-1)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ECGResNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # ecg_array is (5000, 12) → model expects (1, 12, 5000)
    x = torch.FloatTensor(ecg_array.T).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).cpu().numpy()[0]

    print("\n" + "="*45)
    print("  Model Predictions")
    print("="*45)
    print(f"  {'Class':<8} {'Prob':>6}  {'Result':>8}")
    print("  " + "-"*30)
    for cls, prob in zip(CLASSES, probs):
        pred = "POSITIVE" if prob >= THRESHOLDS[cls] else "negative"
        flag = "⚠️ " if pred == "POSITIVE" else "  "
        print(f"  {flag}{cls:<8} {prob:.3f}  {pred}")
    print("="*45)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Digitize ECG image → numpy array")
    parser.add_argument("--image",   required=True, help="Path to ECG image (jpg/png/bmp)")
    parser.add_argument("--manual",  action="store_true", help="Use manual alignment UI (recommended for photos)")
    parser.add_argument("--predict", action="store_true", help="Run model on digitized signal")
    parser.add_argument("--out",     default="ecg_digitized.npy", help="Output .npy path")
    args = parser.parse_args()

    ecg = digitize(args.image, manual=args.manual)

    np.save(args.out, ecg)
    print(f"\nSaved → {args.out}  (shape: {ecg.shape})")

    plot_verification(ecg, args.out.replace(".npy", "_verify.png"))

    if args.predict:
        predict_from_array(ecg)
