#!/usr/bin/env python3
"""
Utility for projecting standard perspective images or re-sampling existing equirectangular
panoramas into a normalized lat/long texture.

When the source is a pinhole camera capture the horizontal field-of-view must be provided
(defaults to 90 degrees) and the vertical FOV is inferred from the aspect ratio unless
explicitly overridden. For panoramic sources the script can automatically detect common
2:1 equirectangular images and re-map them while honoring yaw/pitch/roll offsets.

Pixels outside a perspective camera frustum are filled with a configurable background
color and the resulting panorama spans 360 degrees horizontally and 180 degrees vertically.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, Tuple, Sequence, Dict, Any

import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
SIZE_PRESETS = {
    "2K (2048 x 1024)": 2048,
    "4K (4096 x 2048)": 4096,
    "8K (8192 x 4096)": 8192,
}
QUALITY_PRESETS = {
    "High (95)": 95,
    "Medium (90)": 90,
    "Low (80)": 80,
}


def rotation_matrix(yaw_rad: float, pitch_rad: float, roll_rad: float) -> np.ndarray:
    """Create a camera-to-world rotation matrix using yaw, pitch, roll (in radians)."""
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)

    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    # Intrinsic Tait-Bryan rotation order: yaw (Y), pitch (X), roll (Z)
    return rz @ rx @ ry


def parse_color(value: str | None, channels: int) -> np.ndarray:
    """Parse a color string like `255,0,0` or `#RRGGBB` into an array of length `channels`."""
    if value is None:
        components = [0]
    elif value.startswith("#"):
        value = value.lstrip("#")
        if len(value) not in (3, 6, 8):
            raise ValueError("Hex colors must be #RGB, #RRGGBB or #RRGGBBAA")
        if len(value) == 3:
            value = "".join(2 * ch for ch in value)
        comp = [int(value[i : i + 2], 16) for i in range(0, len(value), 2)]
        components = comp
    else:
        components = [int(part.strip()) for part in value.split(",")]

    if len(components) == 1 and channels > 1:
        components = components * channels
    elif len(components) < channels:
        last = components[-1]
        components = components + [last] * (channels - len(components))
    elif len(components) > channels:
        components = components[:channels]

    return np.array(components, dtype=np.float32)


def bilinear_sample(
    src: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    wrap_u: bool = False,
) -> np.ndarray:
    """Sample `src` at floating point coordinates (u, v) using bilinear interpolation."""
    h, w = src.shape[:2]

    if wrap_u:
        u = np.mod(u, w)
    u_clipped = np.clip(u, 0, w - 1)
    v_clipped = np.clip(v, 0, h - 1)

    x0 = np.floor(u_clipped).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(v_clipped).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, h - 1)

    x_weight = u_clipped - x0
    y_weight = v_clipped - y0

    Ia = src[y0, x0]
    Ib = src[y0, x1]
    Ic = src[y1, x0]
    Id = src[y1, x1]

    top = Ia * (1.0 - x_weight[..., None]) + Ib * x_weight[..., None]
    bottom = Ic * (1.0 - x_weight[..., None]) + Id * x_weight[..., None]
    return top * (1.0 - y_weight[..., None]) + bottom * y_weight[..., None]


def project_to_equirectangular(
    image: Image.Image,
    width: int,
    height: int,
    hfov_deg: float,
    vfov_deg: float | None,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    fill_color: str | None,
    input_projection: str = "auto",
    return_metadata: bool = False,
) -> Image.Image | Tuple[Image.Image, Dict[str, Any]]:
    """Project image data onto an equirectangular panorama."""
    if image.mode not in ("RGB", "RGBA", "L"):
        image = image.convert("RGBA")

    src = np.array(image, dtype=np.float32)
    if src.ndim == 2:
        src = src[..., None]

    src_h, src_w = src.shape[:2]
    if src_h == 0 or src_w == 0:
        raise ValueError("Input image must have non-zero dimensions.")

    projection_mode = input_projection.lower()
    if projection_mode not in {"auto", "perspective", "equirectangular"}:
        raise ValueError(f"Unknown projection type: {input_projection}")
    if projection_mode == "auto":
        aspect = src_w / src_h
        projection_mode = "equirectangular" if abs(aspect - 2.0) <= 0.1 else "perspective"

    hfov_rad = math.radians(hfov_deg)
    if vfov_deg is None:
        vfov_rad = 2.0 * math.atan(math.tan(hfov_rad / 2.0) * (src_h / src_w))
    else:
        vfov_rad = math.radians(vfov_deg)

    rotation = rotation_matrix(
        math.radians(yaw_deg), math.radians(pitch_deg), math.radians(roll_deg)
    )
    world_to_cam = rotation.T

    lon = (np.arange(width, dtype=np.float64) + 0.5) / width * (2.0 * math.pi) - math.pi
    lat = math.pi / 2.0 - (np.arange(height, dtype=np.float64) + 0.5) / height * math.pi

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    cos_lat = np.cos(lat_grid)

    dirs_world = np.stack(
        [cos_lat * np.sin(lon_grid), np.sin(lat_grid), cos_lat * np.cos(lon_grid)],
        axis=-1,
    )
    dirs_cam = dirs_world @ world_to_cam

    metadata: Dict[str, Any] | None = None

    if projection_mode == "equirectangular":
        lon_src = np.arctan2(dirs_cam[..., 0], dirs_cam[..., 2])
        lat_src = np.arcsin(np.clip(dirs_cam[..., 1], -1.0, 1.0))

        u = ((lon_src + math.pi) % (2.0 * math.pi)) / (2.0 * math.pi) * (src_w - 1)
        v = ((math.pi / 2.0 - lat_src) / math.pi) * (src_h - 1)

        sampled = bilinear_sample(src, u, v, wrap_u=True)
        output = sampled
        mask = np.ones((height, width), dtype=bool)
        effective_hfov = 2.0 * math.pi
        effective_vfov = math.pi
        fill_vec = np.zeros((src.shape[2],), dtype=np.float32)
    else:
        z = dirs_cam[..., 2]
        x = dirs_cam[..., 0]
        y = dirs_cam[..., 1]

        front_mask = z > 1e-6

        denom = np.where(np.abs(z) < 1e-6, np.sign(z) * 1e-6, z)
        x_norm = x / denom
        y_norm = y / denom

        h_limit = max(math.tan(hfov_rad / 2.0), 1e-6)
        v_limit = max(math.tan(vfov_rad / 2.0), 1e-6)

        within_fov = (
            (np.abs(x_norm) <= h_limit)
            & (np.abs(y_norm) <= v_limit)
            & front_mask
        )

        output = np.empty((height, width, src.shape[2]), dtype=np.float32)
        fill_vec = parse_color(fill_color, src.shape[2])
        output[:] = fill_vec

        if np.any(within_fov):
            u = ((x_norm / h_limit) + 1.0) * 0.5 * (src_w - 1)
            v = ((y_norm / v_limit) + 1.0) * 0.5 * (src_h - 1)
            sampled = bilinear_sample(src, u, v)
            output[within_fov] = sampled[within_fov]

        mask = within_fov
        effective_hfov = hfov_rad
        effective_vfov = vfov_rad

    if image.mode == "L":
        output = output[..., 0]

    result = np.clip(output, 0, 255).astype(np.uint8)
    mode = image.mode if image.mode in ("RGB", "RGBA", "L") else None
    result_img = Image.fromarray(result, mode=mode)

    if return_metadata:
        metadata = {
            "lon_grid": lon_grid,
            "lat_grid": lat_grid,
            "mask": mask,
            "hfov_rad": effective_hfov,
            "vfov_rad": effective_vfov,
            "fill_color": fill_vec,
            "projection_mode": projection_mode,
        }
        return result_img, metadata
    return result_img


def build_polar_cap(
    source: np.ndarray,
    orientation: str,
    angle_samples: int,
    radius_samples: int,
) -> np.ndarray:
    """Create a polar representation (radius x angle) extracted from the source image."""
    if orientation not in {"top", "bottom"}:
        raise ValueError("orientation must be 'top' or 'bottom'")

    height, width = source.shape[:2]
    size = min(height, width)
    if size < 2:
        return np.zeros((radius_samples, angle_samples, source.shape[2]), dtype=np.float32)

    left = (width - size) // 2
    right = left + size
    if orientation == "top":
        top = 0
        bottom = size
        patch = source[top:bottom, left:right]
    else:
        bottom = height
        top = bottom - size
        patch = source[top:bottom, left:right]

    patch = patch.astype(np.float32)
    if orientation == "top":
        patch = np.flipud(patch)

    patch_h, patch_w = patch.shape[:2]
    center = (patch_w - 1) / 2.0
    max_radius = center

    radius = np.linspace(0.0, max_radius, radius_samples, endpoint=True)
    theta = np.linspace(-math.pi, math.pi, angle_samples, endpoint=False)
    radius_grid, theta_grid = np.meshgrid(radius, theta, indexing="ij")

    x = center + radius_grid * np.cos(theta_grid)
    y = center + radius_grid * np.sin(theta_grid)

    polar = bilinear_sample(patch, x, y)
    if orientation == "top":
        polar = np.flipud(polar)
    return polar


def fill_poles_with_polar(
    panorama: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    mask: np.ndarray,
    src_image: Image.Image,
    vfov_rad: float,
    fill_color: np.ndarray,
    fill_color_specified: bool,
    blur_kernel_size: int,
    noise_std: float,
    blend_power: float = 1.75,
    min_radius_samples: int = 96,
) -> np.ndarray:
    """Fill regions near the poles by sampling polar-mapped content from the source image."""
    filled = panorama.copy()
    channels = panorama.shape[2] if panorama.ndim == 3 else 1
    if channels == 1:
        src = np.array(src_image.convert("L"), dtype=np.float32)[..., None]
    elif channels == 3:
        src = np.array(src_image.convert("RGB"), dtype=np.float32)
    elif channels == 4:
        src = np.array(src_image.convert("RGBA"), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported channel count for polar fill: {channels}")
    fill_vec = fill_color[:channels].astype(np.float32).reshape(1, 1, channels)

    def process_cap(orientation: str) -> None:
        nonlocal filled, mask
        if orientation == "top":
            lat_threshold = vfov_rad / 2.0
            row_selector = lat_grid[:, 0] > lat_threshold
        else:
            lat_threshold = -vfov_rad / 2.0
            row_selector = lat_grid[:, 0] < lat_threshold

        rows = np.where(row_selector)[0]
        if rows.size == 0:
            return

        pole_height = rows.size
        angle_samples = panorama.shape[1]
        radius_samples = max(min_radius_samples, pole_height * 2)

        polar_patch = build_polar_cap(
            src,
            orientation=orientation,
            angle_samples=angle_samples,
            radius_samples=radius_samples,
        )
        anchor_rows = max(1, radius_samples // 6)
        if orientation == "top":
            anchor_slice = polar_patch[-anchor_rows:]
        else:
            anchor_slice = polar_patch[:anchor_rows]
        anchor_color = np.mean(anchor_slice, axis=(0, 1), dtype=np.float32).reshape(
            1, 1, channels
        )
        base_color = (
            0.5 * fill_vec + 0.5 * anchor_color if fill_color_specified else anchor_color
        )

        theta_idx = ((lon_grid[rows][:, :] + math.pi) / (2.0 * math.pi)) * (angle_samples - 1)

        lat_values = lat_grid[rows, 0]
        if orientation == "top":
            clamp = math.pi / 2.0 - lat_threshold
            clamp = clamp if clamp > 1e-6 else 1e-6
            radius_frac = (math.pi / 2.0 - lat_values) / clamp
        else:
            clamp = lat_threshold + math.pi / 2.0
            clamp = clamp if clamp > 1e-6 else 1e-6
            radius_frac = (lat_values + math.pi / 2.0) / clamp
        radius_frac = np.clip(radius_frac, 0.0, 1.0)
        radius_idx = radius_frac[:, None] * (radius_samples - 1)

        sampled = bilinear_sample(polar_patch, theta_idx, radius_idx)

        blend = np.expand_dims(1.0 - (radius_frac[:, None] ** blend_power), axis=-1)
        blend = blend.astype(np.float32)

        fill_array = blend * base_color + (1.0 - blend) * sampled

        if blur_kernel_size > 1:
            fill_array = _blur_vertical(fill_array, kernel_size=blur_kernel_size)
        if noise_std > 0.0:
            fill_array = fill_array + np.random.normal(
                loc=0.0, scale=noise_std, size=fill_array.shape
            ).astype(np.float32)

        if channels == 1:
            filled[rows, :] = fill_array[..., 0][:, :, None]
        else:
            filled[rows, :] = fill_array
        mask[rows, :] = True

    process_cap("top")
    process_cap("bottom")
    return filled


def _blur_vertical(data: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply a simple box blur along the vertical axis."""
    kernel_size = int(kernel_size)
    if kernel_size <= 1 or data.shape[0] < 2:
        return data
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    padded = np.pad(data, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    kernel = np.ones((kernel_size,), dtype=np.float32) / float(kernel_size)
    blurred = np.empty_like(data)
    for col in range(data.shape[1]):
        for ch in range(data.shape[2]):
            column = padded[:, col, ch]
            blurred[:, col, ch] = np.convolve(column, kernel, mode="valid")
    return blurred


def convert_image_file(
    src_path: Path,
    dest_path: Path,
    output_width: int,
    output_height: int,
    hfov_deg: float,
    vfov_deg: float | None,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    fill_color: str | None,
    input_projection: str = "auto",
    quality: int | None = None,
    auto_fill_poles: bool = True,
    polar_blur_kernel: int = 3,
    polar_noise_std: float = 0.0,
) -> Path:
    """Convert a single image file and save it to destination."""
    with Image.open(src_path) as img:
        projection = project_to_equirectangular(
            img,
            width=output_width,
            height=output_height,
            hfov_deg=hfov_deg,
            vfov_deg=vfov_deg,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            fill_color=fill_color,
            input_projection=input_projection,
            return_metadata=auto_fill_poles,
        )

        if auto_fill_poles:
            converted_img, metadata = projection  # type: ignore[misc]
        else:
            converted_img = projection  # type: ignore[assignment]
            metadata = None

        apply_polar_fill = (
            auto_fill_poles
            and metadata is not None
            and metadata.get("projection_mode") == "perspective"
            and not np.all(metadata["mask"])
        )
        if apply_polar_fill:
            converted_array = np.array(converted_img, dtype=np.float32)
            filled_array = fill_poles_with_polar(
                panorama=converted_array if converted_array.ndim == 3 else converted_array[..., None],
                lon_grid=metadata["lon_grid"],
                lat_grid=metadata["lat_grid"],
                mask=metadata["mask"].copy(),
                src_image=img,
                vfov_rad=metadata["vfov_rad"],
                fill_color=metadata["fill_color"],
                fill_color_specified=fill_color is not None,
                blur_kernel_size=max(0, int(polar_blur_kernel)),
                noise_std=max(0.0, float(polar_noise_std)),
            )
            if converted_array.ndim == 2:
                filled_array = filled_array[..., 0]
            converted_array = np.clip(filled_array, 0, 255).astype(np.uint8)
            converted_img = Image.fromarray(converted_array, mode=converted_img.mode)

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {}
        if quality is not None and dest_path.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs["quality"] = quality
        converted_img.save(dest_path, **save_kwargs)
    return dest_path


def build_outputs(
    input_path: Path, output_path: Path | None, suffix: str
) -> Iterable[Tuple[Path, Path]]:
    """Yield (source, destination) pairs based on CLI input."""
    if input_path.is_dir():
        candidates = sorted(
            p for p in input_path.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if output_path is not None and not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        for src in candidates:
            dest_dir = output_path if output_path else src.parent
            target = dest_dir / f"{src.stem}{suffix}{src.suffix}"
            yield src, target
    else:
        if output_path:
            if output_path.exists() and output_path.is_dir():
                destination = output_path / f"{input_path.stem}{suffix}{input_path.suffix}"
            elif output_path.suffix:
                destination = output_path
            else:
                destination = output_path / f"{input_path.stem}{suffix}{input_path.suffix}"
        else:
            destination = input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")
        if destination.parent:
            destination.parent.mkdir(parents=True, exist_ok=True)
        yield input_path, destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project a standard perspective image into an equirectangular panorama. "
            "By default assumes a 90-degree horizontal field-of-view."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="Input file or directory containing images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file or directory. Defaults to alongside the input with '_equi' suffix.",
    )
    parser.add_argument("--output-width", type=int, default=4096, help="Width of the output panorama (default: 4096).")
    parser.add_argument(
        "--output-height",
        type=int,
        help="Optional height of the output panorama. Defaults to width / 2 to preserve 2:1 ratio.",
    )
    parser.add_argument(
        "--hfov",
        type=float,
        default=90.0,
        help="Horizontal field-of-view of the input camera in degrees (default: 90).",
    )
    parser.add_argument(
        "--input-projection",
        choices=["auto", "perspective", "equirectangular"],
        default="auto",
        help=(
            "Projection model of the input imagery. "
            "'auto' infers from the aspect ratio, 'perspective' treats the source as a rectilinear photo, "
            "and 'equirectangular' assumes a full lat/long panorama."
        ),
    )
    parser.add_argument(
        "--vfov",
        type=float,
        help="Vertical field-of-view of the input camera in degrees. Defaults to auto computed.",
    )
    parser.add_argument("--yaw", type=float, default=0.0, help="Yaw offset in degrees (rotation around vertical axis).")
    parser.add_argument("--pitch", type=float, default=0.0, help="Pitch offset in degrees (rotation around X axis).")
    parser.add_argument("--roll", type=float, default=0.0, help="Roll offset in degrees (rotation around Z axis).")
    parser.add_argument(
        "--fill-color",
        default=None,
        help="Background color for unmapped pixels. Accepts '#RRGGBB' or 'r,g,b' formats (default: black).",
    )
    parser.add_argument(
        "--disable-polar-fill",
        action="store_true",
        help="Skip polar coordinate based filling for the zenith and nadir regions.",
    )
    parser.add_argument(
        "--polar-blur-kernel",
        type=int,
        default=3,
        help="Box blur kernel (pixels) applied to polar fills. Use 0 or 1 to disable (default: 3).",
    )
    parser.add_argument(
        "--polar-noise-std",
        type=float,
        default=0.0,
        help="Additive gaussian noise (0-255 range) for polar fills to break banding (default: 0).",
    )
    parser.add_argument(
        "--suffix",
        default="_equi",
        help="Suffix appended to the output filename when writing alongside the input (default: _equi).",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the graphical interface.",
    )
    args = parser.parse_args(argv)
    if not args.gui and args.input is None:
        parser.error("the following arguments are required: input")
    return args


def launch_gui() -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except ImportError as exc:  # pragma: no cover - Tk fails only when missing
        raise RuntimeError("Tkinter is required for GUI mode.") from exc

    root = tk.Tk()
    root.title("Equirectangular Converter")
    root.resizable(False, False)

    input_var = tk.StringVar()
    output_var = tk.StringVar()
    size_choice = tk.StringVar(value="4K (4096 x 2048)")
    quality_choice = tk.StringVar(value="High (95)")
    projection_var = tk.StringVar(value="auto")
    hfov_var = tk.DoubleVar(value=90.0)
    yaw_var = tk.DoubleVar(value=0.0)
    pitch_var = tk.DoubleVar(value=0.0)
    roll_var = tk.DoubleVar(value=0.0)
    polar_fill_var = tk.BooleanVar(value=True)
    polar_blur_var = tk.IntVar(value=3)
    polar_noise_var = tk.DoubleVar(value=0.0)

    def choose_input() -> None:
        path = filedialog.askopenfilename(
            title="Select input image",
            filetypes=[
                ("Supported images", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            input_var.set(path)
            if not output_var.get():
                src = Path(path)
                output_var.set(str(src.with_name(f"{src.stem}_equi{src.suffix}")))

    def choose_output() -> None:
        initial = output_var.get() or input_var.get()
        def_ext = ".jpg"
        if initial:
            def_ext = Path(initial).suffix or ".jpg"
        path = filedialog.asksaveasfilename(
            title="Select output image location",
            defaultextension=def_ext,
            filetypes=[
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("TIFF", "*.tif *.tiff"),
                ("Bitmap", "*.bmp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            output_var.set(path)

    def run_conversion() -> None:
        if not input_var.get():
            messagebox.showerror("Missing input", "Please choose an input image.")
            return
        if not output_var.get():
            messagebox.showerror("Missing output", "Please choose where to save the panorama.")
            return

        src_path = Path(input_var.get())
        dest_path = Path(output_var.get())
        if not src_path.exists():
            messagebox.showerror("Invalid input", "Selected input image does not exist.")
            return

        width = SIZE_PRESETS.get(size_choice.get(), 4096)
        height = max(2, width // 2)
        quality = QUALITY_PRESETS.get(quality_choice.get())

        try:
            convert_image_file(
                src_path,
                dest_path,
                output_width=width,
                output_height=height,
                hfov_deg=hfov_var.get(),
                vfov_deg=None,
                yaw_deg=yaw_var.get(),
                pitch_deg=pitch_var.get(),
            roll_deg=roll_var.get(),
            fill_color=None,
            input_projection=projection_var.get(),
            quality=quality,
            auto_fill_poles=polar_fill_var.get(),
            polar_blur_kernel=polar_blur_var.get(),
            polar_noise_std=polar_noise_var.get(),
        )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            messagebox.showerror("Conversion failed", str(exc))
            return

        messagebox.showinfo("Success", f"Wrote {dest_path}")

    padding = {"padx": 10, "pady": 4}

    tk.Label(root, text="Input image").grid(row=0, column=0, sticky="w", **padding)
    tk.Entry(root, textvariable=input_var, width=40).grid(row=0, column=1, **padding)
    tk.Button(root, text="Browse...", command=choose_input).grid(row=0, column=2, **padding)

    tk.Label(root, text="Output image").grid(row=1, column=0, sticky="w", **padding)
    tk.Entry(root, textvariable=output_var, width=40).grid(row=1, column=1, **padding)
    tk.Button(root, text="Save As...", command=choose_output).grid(row=1, column=2, **padding)

    tk.Label(root, text="Output size").grid(row=2, column=0, sticky="w", **padding)
    tk.OptionMenu(root, size_choice, *SIZE_PRESETS.keys()).grid(row=2, column=1, sticky="ew", **padding)

    tk.Label(root, text="JPEG quality").grid(row=3, column=0, sticky="w", **padding)
    tk.OptionMenu(root, quality_choice, *QUALITY_PRESETS.keys()).grid(row=3, column=1, sticky="ew", **padding)

    tk.Label(root, text="Input projection").grid(row=4, column=0, sticky="w", **padding)
    tk.OptionMenu(root, projection_var, "auto", "perspective", "equirectangular").grid(
        row=4, column=1, sticky="ew", **padding
    )

    tk.Label(root, text="Horizontal FOV (deg)").grid(row=5, column=0, sticky="w", **padding)
    tk.Scale(
        root,
        variable=hfov_var,
        from_=30.0,
        to=160.0,
        resolution=1.0,
        orient="horizontal",
        length=220,
    ).grid(row=5, column=1, columnspan=2, sticky="ew", **padding)

    tk.Label(root, text="Yaw / Pitch / Roll (deg)").grid(row=6, column=0, sticky="w", **padding)
    orientation_frame = tk.Frame(root)
    orientation_frame.grid(row=6, column=1, columnspan=2, sticky="ew", **padding)
    tk.Scale(
        orientation_frame,
        label="Yaw",
        variable=yaw_var,
        from_=-180.0,
        to=180.0,
        resolution=1.0,
        orient="horizontal",
        length=160,
    ).grid(row=0, column=0, padx=(0, 8))
    tk.Scale(
        orientation_frame,
        label="Pitch",
        variable=pitch_var,
        from_=-90.0,
        to=90.0,
        resolution=1.0,
        orient="horizontal",
        length=160,
    ).grid(row=0, column=1, padx=(0, 8))
    tk.Scale(
        orientation_frame,
        label="Roll",
        variable=roll_var,
        from_=-180.0,
        to=180.0,
        resolution=1.0,
        orient="horizontal",
        length=160,
    ).grid(row=0, column=2)

    tk.Label(root, text="Polar blur kernel (px)").grid(row=7, column=0, sticky="w", **padding)
    tk.Scale(
        root,
        variable=polar_blur_var,
        from_=0,
        to=31,
        resolution=1,
        orient="horizontal",
        length=220,
    ).grid(row=7, column=1, columnspan=2, sticky="ew", **padding)

    tk.Label(root, text="Polar noise std (0-255)").grid(row=8, column=0, sticky="w", **padding)
    tk.Scale(
        root,
        variable=polar_noise_var,
        from_=0.0,
        to=40.0,
        resolution=0.5,
        orient="horizontal",
        length=220,
    ).grid(row=8, column=1, columnspan=2, sticky="ew", **padding)

    tk.Checkbutton(
        root,
        text="Fill poles using polar coordinate extrapolation",
        variable=polar_fill_var,
    ).grid(row=9, column=0, columnspan=3, sticky="w", padx=10, pady=(4, 0))

    tk.Button(root, text="Convert", command=run_conversion).grid(row=10, column=0, columnspan=3, pady=12)

    root.mainloop()


def main(argv: Sequence[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        launch_gui()
        return

    args = parse_args(argv)

    if args.gui:
        launch_gui()
        return

    if not args.input.exists():
        raise FileNotFoundError(f"Input path does not exist: {args.input}")

    output_height = args.output_height or max(2, args.output_width // 2)

    for src_path, dest_path in build_outputs(args.input, args.output, args.suffix):
        quality = None
        if dest_path.suffix.lower() in {".jpg", ".jpeg"}:
            quality = QUALITY_PRESETS["High (95)"]
        convert_image_file(
            src_path,
            dest_path,
            output_width=args.output_width,
            output_height=output_height,
            hfov_deg=args.hfov,
            vfov_deg=args.vfov,
            yaw_deg=args.yaw,
            pitch_deg=args.pitch,
            roll_deg=args.roll,
            fill_color=args.fill_color,
            input_projection=args.input_projection,
            quality=quality,
            auto_fill_poles=not args.disable_polar_fill,
            polar_blur_kernel=args.polar_blur_kernel,
            polar_noise_std=args.polar_noise_std,
        )
        print(f"Wrote {dest_path}")


if __name__ == "__main__":
    main()
