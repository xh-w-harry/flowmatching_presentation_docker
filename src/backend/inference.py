import io
import json
import math
import os
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import UploadFile

from model import load_flow_matching_model


@dataclass
class BoreholeSpec:
    x: int
    y: int
    azimuth_degrees: float | None = None
    vertical_degrees: float | None = None


def load_npy_from_upload(upload: UploadFile) -> np.ndarray:
    content = upload.file.read()
    if not content:
        raise ValueError(f"{upload.filename} is empty.")
    try:
        return np.load(io.BytesIO(content), allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"{upload.filename} is not a valid .npy file.") from exc


def parse_json_field(raw: str, name: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON.") from exc


def parse_bounds(raw: str) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    value = parse_json_field(raw, "bounds")
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("bounds must be a JSON list with three [min, max] pairs.")
    parsed = []
    for pair in value:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("bounds must be a JSON list with three [min, max] pairs.")
        parsed.append((float(pair[0]), float(pair[1])))
    return tuple(parsed)  # type: ignore[return-value]


def parse_boreholes(
    raw: str,
    default_azimuth_degrees: float | None = None,
    default_vertical_degrees: float | None = None,
) -> list[BoreholeSpec]:
    value = parse_json_field(raw, "boreholes")
    if not isinstance(value, list) or not value:
        raise ValueError("boreholes must be a non-empty JSON list.")

    specs = []
    for item in value:
        if not isinstance(item, dict) or "x" not in item or "y" not in item:
            raise ValueError("Each borehole must contain x and y.")
        azimuth = item.get("azimuth_degrees", default_azimuth_degrees)
        vertical = item.get("vertical_degrees", default_vertical_degrees)
        specs.append(
            BoreholeSpec(
                x=int(item["x"]),
                y=int(item["y"]),
                azimuth_degrees=float(azimuth) if azimuth is not None else None,
                vertical_degrees=float(vertical) if vertical is not None else None,
            )
        )
    return specs


def normalize_display_volume(volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(f"GeoData must be a 3D .npy array, got shape {volume.shape}.")

    display_volume = volume.astype(np.float32, copy=True)
    finite_values = display_volume[np.isfinite(display_volume)]
    if finite_values.size == 0:
        raise ValueError("GeoData contains no finite voxels.")

    rounded = np.round(finite_values)
    if not np.allclose(finite_values, rounded):
        raise ValueError("GeoData must contain integer-like class ids.")

    if finite_values.min() >= 0 and rounded.max() == 15:
        display_volume[np.isfinite(display_volume)] -= 1

    display_volume[~np.isfinite(display_volume)] = np.nan
    return display_volume


def normalize_borehole_volume(volume: np.ndarray) -> np.ndarray:
    borehole_volume = normalize_display_volume(volume)
    if borehole_volume.ndim != 3:
        raise ValueError(f"Borehole data must be a 3D .npy array, got shape {borehole_volume.shape}.")
    return borehole_volume


def find_surface_z(display_volume: np.ndarray, x: int, y: int, air_value=-1) -> int | None:
    if not (0 <= x < display_volume.shape[0]) or not (0 <= y < display_volume.shape[1]):
        return None

    column = display_volume[x, y, :]
    valid_surface = np.isfinite(column) & (column != air_value)
    surface_indices = np.flatnonzero(valid_surface)
    if surface_indices.size == 0:
        return None
    return int(surface_indices[-1])


def directional_indices(
    display_volume: np.ndarray,
    spec: BoreholeSpec,
    azimuth_degrees: float,
    vertical_degrees: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if not (0 <= azimuth_degrees <= 360):
        raise ValueError("azimuth_degrees must be in [0, 360].")
    if not (0 <= vertical_degrees <= 180):
        raise ValueError("vertical_degrees must be in [0, 180].")

    surface_z = find_surface_z(display_volume, spec.x, spec.y)
    if surface_z is None:
        raise ValueError(f"No surface found at x={spec.x}, y={spec.y}.")

    azimuth = math.radians(azimuth_degrees)
    vertical = math.radians(vertical_degrees)
    direction_x = math.cos(vertical) * math.sin(azimuth)
    direction_y = math.cos(vertical) * math.cos(azimuth)
    direction_z = -math.sin(vertical)

    max_steps = int(sum(display_volume.shape) * 2)
    steps = np.arange(max_steps, dtype=np.float32)
    x_indices = np.rint(spec.x + steps * direction_x).astype(int)
    y_indices = np.rint(spec.y + steps * direction_y).astype(int)
    z_indices = np.rint(surface_z + steps * direction_z).astype(int)

    valid = (
        (x_indices >= 0) & (x_indices < display_volume.shape[0]) &
        (y_indices >= 0) & (y_indices < display_volume.shape[1]) &
        (z_indices >= 0) & (z_indices < display_volume.shape[2])
    )

    if not np.any(valid):
        raise ValueError(f"Borehole at x={spec.x}, y={spec.y} leaves the volume immediately.")

    coords = np.stack([x_indices[valid], y_indices[valid], z_indices[valid]], axis=1)
    _, unique_indices = np.unique(coords, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    coords = coords[unique_indices]
    return coords[:, 0], coords[:, 1], coords[:, 2], surface_z


def build_borehole_volume(
    geo_volume: np.ndarray,
    specs: list[BoreholeSpec],
    default_azimuth_degrees: float,
    default_vertical_degrees: float,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    display_volume = normalize_display_volume(geo_volume)
    borehole_volume = np.full(display_volume.shape, np.nan, dtype=np.float32)
    metadata = []

    for index, spec in enumerate(specs, start=1):
        azimuth = default_azimuth_degrees if spec.azimuth_degrees is None else spec.azimuth_degrees
        vertical = default_vertical_degrees if spec.vertical_degrees is None else spec.vertical_degrees
        x_idx, y_idx, z_idx, surface_z = directional_indices(
            display_volume,
            spec,
            azimuth_degrees=azimuth,
            vertical_degrees=vertical,
        )
        values = display_volume[x_idx, y_idx, z_idx]
        valid_values = np.isfinite(values)
        borehole_volume[x_idx[valid_values], y_idx[valid_values], z_idx[valid_values]] = values[valid_values]
        metadata.append(
            {
                "index": index,
                "x": spec.x,
                "y": spec.y,
                "azimuth_degrees": float(azimuth),
                "vertical_degrees": float(vertical),
                "surface_z": surface_z,
                "samples": int(valid_values.sum()),
            }
        )

    return borehole_volume, metadata


def process_physics_field(field: np.ndarray | None, target_shape: tuple[int, int, int]) -> torch.Tensor:
    if field is None:
        return torch.zeros(target_shape, dtype=torch.float32)

    arr = np.asarray(field)
    if arr.ndim == 1:
        side = int(math.sqrt(arr.shape[0]))
        if side * side != arr.shape[0]:
            raise ValueError("1D physics fields must have a square length.")
        arr = arr.reshape(side, side)

    tensor = torch.tensor(arr, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(tensor, size=target_shape[1:], mode="bilinear", align_corners=False)
        tensor = tensor[0, 0].unsqueeze(0).repeat(target_shape[0], 1, 1)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(tensor, size=target_shape, mode="trilinear", align_corners=False)
        tensor = tensor[0, 0]
    else:
        raise ValueError(f"Physics field must be 1D, 2D, or 3D, got shape {arr.shape}.")

    return (tensor - tensor.mean()) / (tensor.std() + 1e-8)


def build_condition_tensor(
    borehole_volume: np.ndarray,
    geo_volume: np.ndarray | None,
    gravity: np.ndarray | None,
    magnetics: np.ndarray | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    display_borehole = normalize_borehole_volume(borehole_volume)
    target_shape = tuple(int(v) for v in display_borehole.shape)
    bore_np = np.nan_to_num(display_borehole, nan=-2.0) + 1.0

    bore_vol = torch.tensor(bore_np, dtype=torch.float32).unsqueeze(0)
    grav_vol = process_physics_field(gravity, target_shape).unsqueeze(0)
    mag_vol = process_physics_field(magnetics, target_shape).unsqueeze(0)
    cond = torch.cat([bore_vol, grav_vol, mag_vol], dim=0).unsqueeze(0)

    target = None
    if geo_volume is not None:
        display_geo = normalize_display_volume(geo_volume)
        if display_geo.shape != display_borehole.shape:
            raise ValueError(
                f"GeoData shape {display_geo.shape} must match borehole data shape {display_borehole.shape}."
            )
        target_np = np.nan_to_num(display_geo, nan=-1.0) + 1.0
        target = torch.tensor(target_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return cond, target


def run_inference(
    model_path: Path,
    run_root: Path,
    borehole_volume: np.ndarray,
    geo_volume: np.ndarray | None,
    gravity: np.ndarray | None,
    magnetics: np.ndarray | None,
    boreholes: list[BoreholeSpec],
    azimuth_degrees: float,
    vertical_degrees: float,
    num_classes: int,
    num_generations: int,
    ode_steps: int,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    cancel_event: Event | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    run_id = uuid.uuid4().hex[:12]
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    display_borehole = normalize_borehole_volume(borehole_volume)
    cond, target = build_condition_tensor(display_borehole, geo_volume, gravity, magnetics)

    files = ["metadata.json", "input_boreholes.npy"]
    np.save(run_dir / "input_boreholes.npy", display_borehole.astype(np.float32))
    if target is not None:
        np.save(run_dir / "ground_truth.npy", target.cpu().squeeze().numpy().astype(np.int16))
        files.append("ground_truth.npy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fm_model = load_flow_matching_model(str(model_path), device=device)
    cond = cond.to(device)
    if target is not None:
        target = target.to(device)

    outputs = []
    for gen_idx in range(num_generations):
        if cancel_event is not None and cancel_event.is_set():
            break

        predicted = fm_model.sample_ode(cond, steps=ode_steps)
        if cancel_event is not None and cancel_event.is_set():
            break

        pred_discrete = torch.round(predicted).clamp(0, num_classes - 1)

        borehole_cond = cond[:, 0:1, :, :, :]
        known_mask = borehole_cond >= 0
        pred_discrete[known_mask] = borehole_cond[known_mask]

        if target is not None:
            air_mask = target == 0
            pred_discrete[air_mask] = 0

        output_name = f"generation_{gen_idx + 1:02d}.npy"
        output_path = run_dir / output_name
        np.save(output_path, pred_discrete.cpu().squeeze().numpy().astype(np.int16))
        outputs.append(output_name)
        files.append(output_name)
        if progress_callback is not None:
            progress_callback(
                {
                    "completed_generations": gen_idx + 1,
                    "num_generations": num_generations,
                    "latest_output": output_name,
                }
            )

    cancelled = cancel_event is not None and cancel_event.is_set()
    metadata = {
        "run_id": run_id,
        "cancelled": cancelled,
        "num_classes": num_classes,
        "num_generations": num_generations,
        "ode_steps": ode_steps,
        "bounds": bounds,
        "default_azimuth_degrees": azimuth_degrees,
        "default_vertical_degrees": vertical_degrees,
        "ground_truth_used": geo_volume is not None,
        "gravity_used": gravity is not None,
        "magnetics_used": magnetics is not None,
        "borehole_shape": list(display_borehole.shape),
        "boreholes": [
            {
                "index": index,
                "x": spec.x,
                "y": spec.y,
                "azimuth_degrees": spec.azimuth_degrees,
                "vertical_degrees": spec.vertical_degrees,
            }
            for index, spec in enumerate(boreholes, start=1)
        ],
        "outputs": outputs,
        "files": files,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    zip_path = run_dir / "outputs.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in files:
            zf.write(run_dir / filename, arcname=filename)

    return metadata
