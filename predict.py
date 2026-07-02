import os
import sys
import pickle
import traceback
from typing import List, Tuple

import cv2
import numpy as np

IMG_SIZE = 512

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURE_NAMES_PATH = "feature_names.pkl"

LABEL_MAP = {0: "REAL", 1: "FAKE"}


def read_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"Failed to read image (corrupt or unsupported): {path}")

    if image.shape[0] != IMG_SIZE or image.shape[1] != IMG_SIZE:
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    return image


def extract_fft_features(gray: np.ndarray) -> Tuple[List[float], List[str]]:
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1.0)

    fft_mean = float(np.mean(magnitude_spectrum))
    fft_std = float(np.std(magnitude_spectrum))
    fft_max = float(np.max(magnitude_spectrum))
    fft_energy = float(np.sum(magnitude_spectrum ** 2))

    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    max_radius = np.sqrt(cy ** 2 + cx ** 2)

    low_freq_mask = dist_from_center <= (0.15 * max_radius)
    high_freq_mask = dist_from_center > (0.5 * max_radius)

    low_freq_energy = float(np.sum(magnitude_spectrum[low_freq_mask] ** 2))
    high_freq_energy = float(np.sum(magnitude_spectrum[high_freq_mask] ** 2))
    high_low_ratio = high_freq_energy / (low_freq_energy + 1e-8)

    values = [fft_mean, fft_std, fft_max, fft_energy, high_freq_energy, high_low_ratio]
    names = [
        "fft_mean",
        "fft_std",
        "fft_max",
        "fft_energy",
        "fft_high_freq_energy",
        "fft_high_low_ratio",
    ]
    return values, names


def extract_patch_fft_features(gray: np.ndarray, grid: int = 4) -> Tuple[List[float], List[str]]:
    h, w = gray.shape
    ph, pw = h // grid, w // grid

    means, stds, maxs, high_low_ratios = [], [], [], []

    for gy in range(grid):
        for gx in range(grid):
            patch = gray[gy * ph:(gy + 1) * ph, gx * pw:(gx + 1) * pw]
            if patch.size == 0:
                continue

            f = np.fft.fft2(patch.astype(np.float64))
            fshift = np.fft.fftshift(f)
            mag = np.log(np.abs(fshift) + 1.0)

            means.append(float(np.mean(mag)))
            stds.append(float(np.std(mag)))
            maxs.append(float(np.max(mag)))

            ph_, pw_ = mag.shape
            cy, cx = ph_ // 2, pw_ // 2
            Y, X = np.ogrid[:ph_, :pw_]
            dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
            max_r = np.sqrt(cy ** 2 + cx ** 2) + 1e-8

            low_mask = dist <= (0.15 * max_r)
            high_mask = dist > (0.5 * max_r)
            low_energy = float(np.sum(mag[low_mask] ** 2))
            high_energy = float(np.sum(mag[high_mask] ** 2))
            high_low_ratios.append(high_energy / (low_energy + 1e-8))

    values = [
        float(np.mean(means)), float(np.mean(stds)),
        float(np.mean(maxs)), float(np.mean(high_low_ratios)),
        float(np.std(high_low_ratios)),
    ]
    names = [
        "patch_fft_mean_avg", "patch_fft_std_avg",
        "patch_fft_max_avg", "patch_fft_high_low_ratio_avg",
        "patch_fft_high_low_ratio_std",
    ]
    return values, names


def extract_laplacian_feature(gray: np.ndarray) -> Tuple[List[float], List[str]]:
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(laplacian.var())
    lap_mean_abs = float(np.mean(np.abs(laplacian)))

    values = [lap_var, lap_mean_abs]
    names = ["laplacian_variance", "laplacian_mean_abs"]
    return values, names


def extract_edge_density_features(gray: np.ndarray) -> Tuple[List[float], List[str]]:
    edges = cv2.Canny(gray, 100, 200)
    edge_pixel_count = int(np.count_nonzero(edges))
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_density = float(edge_pixel_count) / float(total_pixels)

    values = [edge_density]
    names = ["edge_density"]
    return values, names


def extract_hsv_features(image_bgr: np.ndarray) -> Tuple[List[float], List[str]]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float64)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    h_mean, h_std = float(np.mean(h_channel)), float(np.std(h_channel))
    s_mean, s_std = float(np.mean(s_channel)), float(np.std(s_channel))
    v_mean, v_std = float(np.mean(v_channel)), float(np.std(v_channel))

    values = [h_mean, h_std, s_mean, s_std, v_mean, v_std]
    names = ["hsv_h_mean", "hsv_h_std", "hsv_s_mean", "hsv_s_std", "hsv_v_mean", "hsv_v_std"]
    return values, names


def extract_brightness_features(gray: np.ndarray) -> Tuple[List[float], List[str]]:
    brightness_mean = float(np.mean(gray))
    brightness_var = float(np.var(gray))

    overexposed_ratio = float(np.mean(gray >= 250))
    underexposed_ratio = float(np.mean(gray <= 5))

    values = [brightness_mean, brightness_var, overexposed_ratio, underexposed_ratio]
    names = ["brightness_mean", "brightness_variance", "overexposed_ratio", "underexposed_ratio"]
    return values, names


def extract_noise_features(gray: np.ndarray) -> Tuple[List[float], List[str]]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    high_freq_residual = gray.astype(np.float64) - blurred.astype(np.float64)

    noise_std = float(np.std(high_freq_residual))
    noise_mean_abs = float(np.mean(np.abs(high_freq_residual)))

    h, w = gray.shape
    block_size = 8
    h_trim = h - (h % block_size)
    w_trim = w - (w % block_size)
    trimmed = gray[:h_trim, :w_trim].astype(np.float32)

    ac_energies = []
    for by in range(0, h_trim, block_size):
        for bx in range(0, w_trim, block_size):
            block = trimmed[by:by + block_size, bx:bx + block_size]
            dct_block = cv2.dct(block)
            ac_energy = float(np.sum(dct_block ** 2) - dct_block[0, 0] ** 2)
            ac_energies.append(ac_energy)

    ac_energies = np.array(ac_energies) if len(ac_energies) > 0 else np.array([0.0])
    dct_ac_energy_mean = float(np.mean(ac_energies))
    dct_ac_energy_std = float(np.std(ac_energies))

    values = [noise_std, noise_mean_abs, dct_ac_energy_mean, dct_ac_energy_std]
    names = ["noise_std", "noise_mean_abs", "dct_ac_energy_mean", "dct_ac_energy_std"]
    return values, names


def compute_lbp_image(gray: np.ndarray, radius: int = 1) -> np.ndarray:
    gray_f = gray.astype(np.int16)
    padded = cv2.copyMakeBorder(
        gray_f, radius, radius, radius, radius, borderType=cv2.BORDER_REPLICATE
    )

    center = padded[radius:-radius, radius:-radius]

    offsets = [
        (-radius, -radius), (-radius, 0), (-radius, radius),
        (0, radius), (radius, radius), (radius, 0),
        (radius, -radius), (0, -radius),
    ]

    lbp = np.zeros_like(center, dtype=np.uint8)
    h, w = center.shape

    for bit_index, (dy, dx) in enumerate(offsets):
        neighbor = padded[radius + dy: radius + dy + h, radius + dx: radius + dx + w]
        bit = (neighbor >= center).astype(np.uint8)
        lbp |= (bit << bit_index)

    return lbp


def extract_lbp_features(gray: np.ndarray, n_bins: int = 16) -> Tuple[List[float], List[str]]:
    lbp = compute_lbp_image(gray, radius=1)

    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, 256))
    hist = hist.astype(np.float64)
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist = hist / hist_sum

    lbp_mean = float(np.mean(lbp))
    lbp_std = float(np.std(lbp))

    eps = 1e-8
    lbp_entropy = float(-np.sum(hist * np.log2(hist + eps)))

    values = list(hist) + [lbp_mean, lbp_std, lbp_entropy]
    names = [f"lbp_hist_bin_{i}" for i in range(n_bins)] + ["lbp_mean", "lbp_std", "lbp_entropy"]
    return values, names


def extract_glare_features(gray: np.ndarray, threshold: int = 245) -> Tuple[List[float], List[str]]:
    bright_mask = (gray >= threshold).astype(np.uint8)
    bright_ratio = float(np.mean(bright_mask))

    num_labels, _ = cv2.connectedComponents(bright_mask)
    bright_region_count = int(max(num_labels - 1, 0))

    values = [bright_ratio, float(bright_region_count)]
    names = ["glare_bright_pixel_ratio", "glare_bright_region_count"]
    return values, names


def extract_all_features(image_bgr: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    all_values = []
    all_names = []

    for extractor, args in [
        (extract_fft_features, (gray,)),
        (extract_patch_fft_features, (gray,)),
        (extract_laplacian_feature, (gray,)),
        (extract_edge_density_features, (gray,)),
        (extract_hsv_features, (image_bgr,)),
        (extract_brightness_features, (gray,)),
        (extract_noise_features, (gray,)),
        (extract_lbp_features, (gray,)),
        (extract_glare_features, (gray,)),
    ]:
        values, names = extractor(*args)
        all_values.extend(values)
        all_names.extend(names)

    return np.array(all_values, dtype=np.float64), all_names


_model = None
_scaler = None
_feature_names = None


def _load_artifacts():
    global _model, _scaler, _feature_names

    if _model is not None and _scaler is not None and _feature_names is not None:
        return _model, _scaler, _feature_names

    for path in (MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Required artifact not found: {path}. "
                f"Make sure model.pkl, scaler.pkl, and feature_names.pkl "
                f"(produced by train.py) are present in the working directory."
            )

    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        _scaler = pickle.load(f)

    with open(FEATURE_NAMES_PATH, "rb") as f:
        _feature_names = pickle.load(f)

    return _model, _scaler, _feature_names


def predict_image(image_path: str) -> Tuple[str, float]:
    model, scaler, feature_names = _load_artifacts()

    image = read_image(image_path)
    feature_vector, extracted_names = extract_all_features(image)

    if extracted_names != feature_names:
        raise ValueError(
            "Feature mismatch between predict.py and the saved feature_names.pkl. "
            "This should never happen if predict.py stays in sync with train.py."
        )

    X = feature_vector.reshape(1, -1)
    X_scaled = scaler.transform(X)

    pred_label_idx = int(model.predict(X_scaled)[0])
    label = LABEL_MAP.get(pred_label_idx, str(pred_label_idx))

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        confidence = float(proba[pred_label_idx]) * 100.0
    else:
        confidence = 100.0

    return label, confidence


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        label, confidence = predict_image(image_path)
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2f}%")
    except Exception as exc:
        print(f"[ERROR] Prediction failed: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()