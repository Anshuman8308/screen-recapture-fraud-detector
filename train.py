import os
import sys
import pickle
import random
import traceback
from typing import List, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

DATASET_ROOT = "dataset_final"
REAL_DIR = os.path.join(DATASET_ROOT, "real")
FAKE_DIR = os.path.join(DATASET_ROOT, "fake")

IMG_SIZE = 512
RANDOM_STATE = 42

MODEL_OUTPUT_PATH = "model.pkl"
FEATURE_NAMES_OUTPUT_PATH = "feature_names.pkl"
SCALER_OUTPUT_PATH = "scaler.pkl"


def load_image_paths(real_dir: str, fake_dir: str) -> Tuple[List[str], List[int]]:
    valid_extensions = (".jpg", ".jpeg", ".JPG", ".JPEG")
    paths = []
    labels = []

    if not os.path.isdir(real_dir):
        raise FileNotFoundError(f"Real image directory not found: {real_dir}")
    if not os.path.isdir(fake_dir):
        raise FileNotFoundError(f"Fake image directory not found: {fake_dir}")

    for fname in sorted(os.listdir(real_dir)):
        if fname.endswith(valid_extensions):
            paths.append(os.path.join(real_dir, fname))
            labels.append(0)

    for fname in sorted(os.listdir(fake_dir)):
        if fname.endswith(valid_extensions):
            paths.append(os.path.join(fake_dir, fname))
            labels.append(1)

    if len(paths) == 0:
        raise RuntimeError("No valid JPG images found in dataset directories.")

    return paths, labels


def read_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"Failed to read image (corrupt or unsupported): {path}")

    if image.shape[0] != IMG_SIZE or image.shape[1] != IMG_SIZE:
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    return image


def augment_image(image: np.ndarray) -> List[np.ndarray]:
    augmented = []

    bright = cv2.convertScaleAbs(image, alpha=1.0, beta=random.choice([-30, -15, 15, 30]))
    augmented.append(bright)

    blur = cv2.GaussianBlur(image, (5, 5), sigmaX=random.uniform(0.5, 1.5))
    augmented.append(blur)

    contrast = cv2.convertScaleAbs(image, alpha=random.choice([0.8, 0.9, 1.1, 1.2]), beta=0)
    augmented.append(contrast)

    angle = random.uniform(-10, 10)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rot_mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    augmented.append(rotated)

    noise = np.random.normal(0, random.uniform(5, 15), image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    augmented.append(noisy)

    return augmented


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


def build_feature_matrix(
    paths: List[str],
    labels: List[int],
    use_augmentation: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feature_rows = []
    valid_labels = []
    feature_names = None

    for idx, (path, label) in enumerate(zip(paths, labels)):
        try:
            image = read_image(path)
            feature_vector, names = extract_all_features(image)

            if feature_names is None:
                feature_names = names

            feature_rows.append(feature_vector)
            valid_labels.append(label)

            if use_augmentation:
                for aug_image in augment_image(image):
                    aug_feature_vector, _ = extract_all_features(aug_image)
                    feature_rows.append(aug_feature_vector)
                    valid_labels.append(label)

        except Exception as exc:
            print(f"[WARNING] Skipping image due to error: {path} -> {exc}")
            traceback.print_exc()
            continue

        if (idx + 1) % 10 == 0 or (idx + 1) == len(paths):
            print(f"Processed {idx + 1}/{len(paths)} images...")

    if len(feature_rows) == 0:
        raise RuntimeError("No features could be extracted from any image.")

    X = np.vstack(feature_rows)
    y = np.array(valid_labels, dtype=np.int32)

    return X, y, feature_names


def get_candidate_param_grids() -> dict:
    grids = {
        "RandomForestClassifier": {
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            "param_grid": {
                "n_estimators": [200, 500, 1000],
                "max_depth": [None, 10, 20, 30],
            },
        },
        "ExtraTreesClassifier": {
            "estimator": ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            "param_grid": {
                "n_estimators": [200, 500, 1000],
                "max_depth": [None, 10, 20],
            },
        },
        "GradientBoostingClassifier": {
            "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
            },
        },
    }
    return grids


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
    }
    return metrics


def train_and_select_best_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    grids = get_candidate_param_grids()
    results = {}

    for name, cfg in grids.items():
        print(f"\n{'=' * 60}")
        print(f"GridSearchCV tuning: {name}")
        print(f"{'=' * 60}")

        try:
            search = GridSearchCV(
                cfg["estimator"],
                cfg["param_grid"],
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                scoring="accuracy",
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            print(f"Best params for {name}: {search.best_params_}")

            metrics = evaluate_model(model, X_test, y_test)

            print(f"Accuracy : {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall   : {metrics['recall']:.4f}")
            print(f"F1 Score : {metrics['f1']:.4f}")
            print("Confusion Matrix:")
            print(metrics["confusion_matrix"])
            print("\nClassification Report:")
            print(classification_report(y_test, metrics["y_pred"], target_names=["real", "fake"], zero_division=0))

            results[name] = {"model": model, "metrics": metrics}

        except Exception as exc:
            print(f"[ERROR] Training failed for {name}: {exc}")
            traceback.print_exc()
            continue

    if not results:
        raise RuntimeError("All candidate models failed to train.")

    best_name = max(
        results.keys(),
        key=lambda n: (results[n]["metrics"]["accuracy"], results[n]["metrics"]["f1"]),
    )

    print(f"\n{'#' * 60}")
    print(f"BEST MODEL SELECTED: {best_name}")
    print(f"{'#' * 60}")

    return best_name, results[best_name]["model"], results[best_name]["metrics"], results


def run_cross_validation(model, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> float:
    from sklearn.base import clone

    cv_model = clone(model)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    scores = cross_val_score(cv_model, X, y, cv=skf, scoring="accuracy", n_jobs=-1)

    print(f"\n{n_splits}-Fold Cross Validation Scores: {scores}")
    print(f"Average CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

    return float(scores.mean())


def print_feature_importance(model, feature_names: List[str], top_n: int = 20) -> None:
    if not hasattr(model, "feature_importances_"):
        print("Selected model does not expose feature importances.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\nTop {top_n} Most Important Features:")
    print(f"{'Rank':<6}{'Feature':<30}{'Importance':<10}")
    for rank, idx in enumerate(indices[:top_n], start=1):
        print(f"{rank:<6}{feature_names[idx]:<30}{importances[idx]:.6f}")


def save_artifacts(model, feature_names: List[str], scaler: StandardScaler = None) -> None:
    try:
        with open(MODEL_OUTPUT_PATH, "wb") as f:
            pickle.dump(model, f)
        print(f"\nSaved best model to: {MODEL_OUTPUT_PATH}")

        with open(FEATURE_NAMES_OUTPUT_PATH, "wb") as f:
            pickle.dump(feature_names, f)
        print(f"Saved feature names to: {FEATURE_NAMES_OUTPUT_PATH}")

        if scaler is not None:
            with open(SCALER_OUTPUT_PATH, "wb") as f:
                pickle.dump(scaler, f)
            print(f"Saved scaler to: {SCALER_OUTPUT_PATH}")

    except Exception as exc:
        print(f"[ERROR] Failed to save artifacts: {exc}")
        traceback.print_exc()
        raise


def main():
    try:
        print("=" * 60)
        print("STEP 1: LOADING DATASET")
        print("=" * 60)

        paths, labels = load_image_paths(REAL_DIR, FAKE_DIR)

        print(f"Found {len(paths)} images "
              f"({labels.count(0)} real, {labels.count(1)} fake).")

        print("\n" + "=" * 60)
        print("STEP 2: TRAIN/TEST SPLIT (before feature extraction)")
        print("=" * 60)

        paths_train, paths_test, labels_train, labels_test = train_test_split(
            paths,
            labels,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=labels
        )

        print(f"Train images: {len(paths_train)} | Test images: {len(paths_test)}")

        print("\n" + "=" * 60)
        print("STEP 3: FEATURE EXTRACTION")
        print("(augmentation ONLY on training set)")
        print("=" * 60)

        X_train, y_train, feature_names = build_feature_matrix(
            paths_train,
            labels_train,
            use_augmentation=True
        )

        X_test, y_test, _ = build_feature_matrix(
            paths_test,
            labels_test,
            use_augmentation=False
        )

        print(f"\nTrain samples (augmented): {X_train.shape[0]}")
        print(f"Test samples (original): {X_test.shape[0]}")
        print(f"Feature vector length: {X_train.shape[1]}")

        print("\n" + "=" * 60)
        print("STEP 4: FEATURE SCALING (StandardScaler)")
        print("=" * 60)

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_full_for_cv = np.vstack([X_train, X_test])
        y_full_for_cv = np.concatenate([y_train, y_test])

        print("\n" + "=" * 60)
        print("STEP 5 & 6: MODEL TRAINING (GridSearchCV) AND METRICS")
        print("=" * 60)

        best_name, best_model, best_metrics, all_results = train_and_select_best_model(
            X_train,
            y_train,
            X_test,
            y_test
        )

        print("\n" + "=" * 60)
        print("STEP 7: CROSS VALIDATION")
        print("=" * 60)

        run_cross_validation(
            best_model,
            X_full_for_cv,
            y_full_for_cv,
            n_splits=5
        )

        print("\n" + "=" * 60)
        print("STEP 8: FEATURE IMPORTANCE")
        print("=" * 60)

        print_feature_importance(
            best_model,
            feature_names,
            top_n=20
        )

        print("\n" + "=" * 60)
        print("STEP 9: SAVING MODEL, FEATURE NAMES, AND SCALER")
        print("=" * 60)

        save_artifacts(
            best_model,
            feature_names,
            scaler
        )

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)

        print(f"Best model: {best_name}")
        print(f"Test Accuracy : {best_metrics['accuracy']:.4f}")
        print(f"Test Precision: {best_metrics['precision']:.4f}")
        print(f"Test Recall   : {best_metrics['recall']:.4f}")
        print(f"Test F1 Score : {best_metrics['f1']:.4f}")

    except Exception as exc:
        print(f"\n[FATAL ERROR] Pipeline failed: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()