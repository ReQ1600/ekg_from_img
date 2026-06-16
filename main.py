import numpy as np
import cv2
import os
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import argparse

CHANNEL_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# measured length of the 200 ms reference interval in pixels
PX200MS = 226
PX_PER_MS = PX200MS / 200.0

def load_images(folder="imgs", flags=cv2.IMREAD_GRAYSCALE):
    images = []

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)

        img = cv2.imread(filepath, flags)
        if img is not None:
            images.append((filename, img))

    return images

def expand_channel_bands(bands, image_height):
    centers = np.array([(y1 + y2) // 2 for y1, y2 in bands])

    if len(centers) >= 2:
        spacing = np.median(np.diff(centers))

        # the initial detected band may contain only the brightest part of the trace
        # so the window is expanded to include larger ECG deflections
        half_height = int(2.0 * spacing)
    else:
        half_height = 40

    expanded = []

    for c in centers:
        y1 = max(0, int(c - half_height))
        y2 = min(image_height, int(c + half_height))
        expanded.append((y1, y2))

    return expanded

def get_column_candidates(band_column, y_offset):
    ys = np.where(band_column)[0]

    if len(ys) == 0:
        return []

    candidates = []
    start = ys[0]
    prev = ys[0]

    # consecutive bright pixels are treated as one thick line segment
    # the candidate is the center of that segment
    for y in ys[1:]:
        if y == prev + 1:
            prev = y
        else:
            candidates.append((start + prev) / 2 + y_offset)
            start = y
            prev = y

    candidates.append((start + prev) / 2 + y_offset)

    return candidates

def extract_trace_from_band_tracking(mask, y1, y2, x1, x2, channel_center_y):
    band = mask[y1:y2, x1:x2]
    h, w = band.shape

    trace_y = np.full(w, np.nan)

    prev_y = channel_center_y

    for x in range(w):
        candidates = get_column_candidates(band[:, x], y1)

        if len(candidates) == 0:
            continue

        candidates = np.array(candidates)

        # prefer points close to the previous trace position
        # a small penalty keeps the tracker near the current channel center
        costs = (
            np.abs(candidates - prev_y)
            + 0.15 * np.abs(candidates - channel_center_y)
        )

        best_y = candidates[np.argmin(costs)]

        trace_y[x] = best_y
        prev_y = best_y

    xs = np.arange(w)
    valid = ~np.isnan(trace_y)

    if valid.sum() < 2:
        return None

    # fill missing columns where no bright pixel was detected
    trace_y = np.interp(xs, xs[valid], trace_y[valid])

    return trace_y

def get_channels(mask, min_height=10, max_height=200, merge_gap=8):
    h, w = mask.shape

    y_offset = int(0.12 * h)
    y_end = int(0.98 * h)

    x_offset = int(0.033 * w)
    x_end = int(0.995 * w)

    mask_signal_area = mask[y_offset:y_end, x_offset:x_end]

    v_proj = mask_signal_area.sum(axis=1)

    # smoothing for robustness
    v_proj_smooth = gaussian_filter1d(v_proj.astype(float), sigma=2)

    thresh = max(
        np.percentile(v_proj_smooth, 75),
        0.02 * v_proj_smooth.max()
    )

    active = v_proj_smooth > thresh

    bands = []
    start = None

    for y, is_active in enumerate(active):
        if is_active and start is None:
            start = y
        elif not is_active and start is not None:
            end = y
            bands.append((start, end))
            start = None

    if start is not None:
        bands.append((start, len(active)))

    merged = []

    for band in bands:
        if not merged:
            merged.append(band)
            continue

        prev_start, prev_end = merged[-1]
        curr_start, curr_end = band

        # merge nearby active regions that likely belong to the same channel
        if curr_start - prev_end <= merge_gap:
            merged[-1] = (prev_start, curr_end)
        else:
            merged.append(band)

    filtered = []

    for y1, y2 in merged:
        height = y2 - y1
        if min_height <= height <= max_height:
            filtered.append((y1 + y_offset, y2 + y_offset))

    return filtered, x_offset, x_end

def estimate_baseline(trace_y):
    y_smooth = gaussian_filter1d(trace_y.astype(float), sigma=2)

    dy = np.gradient(y_smooth)
    abs_dy = np.abs(dy)

    threshold = np.percentile(abs_dy, 35)

    # flat parts of the signal are treated as candidates for the isoelectric line
    flat_mask = abs_dy <= threshold
    flat_values = y_smooth[flat_mask]

    if len(flat_values) < 10:
        return np.median(y_smooth)

    return np.median(flat_values)

def resample_to_1ms(trace_y, baseline_y):
    amplitude = baseline_y - trace_y

    x_pixels = np.arange(len(amplitude))
    time_raw_ms = x_pixels / PX_PER_MS

    max_time = int(np.floor(time_raw_ms[-1]))
    time_target = np.arange(0, max_time + 1, 1.0)

    signal_1ms = np.interp(time_target, time_raw_ms, amplitude)

    return time_target, signal_1ms

def ekg2csv(img, debug=False):
    # bright pixels belong mainly to ekg traces
    thresh = np.percentile(img, 95)
    mask = img > thresh

    channels, x1, x2 = get_channels(mask)

    h, w = img.shape

    channel_windows = expand_channel_bands(channels, h)

    channel_windows = channel_windows[:12]

    traces = {}
    baselines = {}

    for name, (y1, y2) in zip(CHANNEL_NAMES, channel_windows):
        channel_center_y = (y1 + y2) / 2

        trace_y = extract_trace_from_band_tracking(
            mask,
            y1,
            y2,
            x1,
            x2,
            channel_center_y
        )

        if trace_y is None:
            print(f"channel acquisition failure: {name}")
            continue

        baseline_y = estimate_baseline(trace_y)

        traces[name] = trace_y
        baselines[name] = baseline_y

    if debug:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for i, (name, (y1, y2)) in enumerate(zip(CHANNEL_NAMES, channel_windows)):
            color = (0, 255, 255)

            if name in traces:
                trace_y = traces[name]
                baseline_y = baselines[name]

                pts = []

                for x_local, y in enumerate(trace_y):
                    x_global = x1 + x_local
                    pts.append([x_global, int(round(y))])

                pts = np.array(pts, dtype=np.int32)

                cv2.polylines(vis, [pts], False, color, 2)

        window_name = "Traces and baselines"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        cv2.imshow(window_name, vis)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # CSV
    result = None
    time_ms = None

    for name in CHANNEL_NAMES:
        if name not in traces:
            continue

        baseline_y = baselines[name]
        trace_y = traces[name]

        t, signal = resample_to_1ms(trace_y, baseline_y)

        if result is None:
            time_ms = t
            result = pd.DataFrame({"time_ms": time_ms})

        signal = np.interp(time_ms, t, signal)

        result[name] = signal

    if result is None:
        raise RuntimeError("No channels extracted from image")

    # keep the required csv schema even when some channel was not extracted
    for name in CHANNEL_NAMES:
        if name not in result.columns:
            result[name] = 0

    result = result[["time_ms"] + CHANNEL_NAMES]

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    imgs = load_images(args.input_dir, flags=cv2.IMREAD_GRAYSCALE)
    
    if len(imgs) == 0:
        print("No images found in the input directory")
        exit()

    os.makedirs(args.output_dir, exist_ok=True)

    for filename, img in imgs:
        df = ekg2csv(img, debug=args.debug)

        output_filename = os.path.splitext(filename)[0] + ".csv"
        output_path = os.path.join(args.output_dir, output_filename)
        
        df.to_csv(output_path, index=False)

        print(f"Saved {output_path}")

    cv2.destroyAllWindows()