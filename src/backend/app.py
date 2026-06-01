import os
from pathlib import Path
from threading import Event
from typing import Any
from uuid import uuid4

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from inference import (
    load_npy_from_upload,
    parse_boreholes,
    parse_bounds,
    run_inference,
)


APP_ROOT = Path(os.getenv("APP_ROOT", "/app"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", APP_ROOT / "models" / "best_fm_multimodal.pth"))
RUN_ROOT = Path(os.getenv("RUN_ROOT", APP_ROOT / "runs"))
TASKS: dict[str, dict[str, Any]] = {}

app = FastAPI(title="Directional Borehole Flow Matching API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_exists": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
    }


@app.post("/api/infer")
def infer(
    background_tasks: BackgroundTasks,
    borehole_file: UploadFile = File(...),
    geo_file: UploadFile = File(...),
    gravity_file: UploadFile | None = File(default=None),
    magnetics_file: UploadFile | None = File(default=None),
    boreholes: str = Form(default="[]"),
    azimuth_degrees: float = Form(default=90),
    vertical_degrees: float = Form(default=90),
    num_classes: int = Form(...),
    num_generations: int = Form(...),
    bounds: str = Form(...),
    ode_steps: int = Form(default=50),
):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=500, detail=f"Model file not found: {MODEL_PATH}")
    if num_classes < 2:
        raise HTTPException(status_code=400, detail="num_classes must be at least 2.")
    if num_generations < 1:
        raise HTTPException(status_code=400, detail="num_generations must be at least 1.")
    if ode_steps < 1:
        raise HTTPException(status_code=400, detail="ode_steps must be at least 1.")

    try:
        borehole_volume = load_npy_from_upload(borehole_file)
        geo_volume = load_npy_from_upload(geo_file)
        gravity = load_npy_from_upload(gravity_file) if gravity_file is not None else None
        magnetics = load_npy_from_upload(magnetics_file) if magnetics_file is not None else None
        parsed_bounds = parse_bounds(bounds)
        borehole_specs = []
        if boreholes.strip() not in ("", "[]"):
            borehole_specs = parse_boreholes(
                boreholes,
                default_azimuth_degrees=azimuth_degrees,
                default_vertical_degrees=vertical_degrees,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    task_id = uuid4().hex[:12]
    cancel_event = Event()
    TASKS[task_id] = {
        "task_id": task_id,
        "status": "queued",
        "cancel_requested": False,
        "cancel_event": cancel_event,
        "metadata": None,
        "error": None,
        "progress": {
            "completed_generations": 0,
            "num_generations": num_generations,
        },
    }
    background_tasks.add_task(
        _run_inference_task,
        task_id,
        cancel_event,
        borehole_volume,
        geo_volume,
        gravity,
        magnetics,
        borehole_specs,
        azimuth_degrees,
        vertical_degrees,
        num_classes,
        num_generations,
        ode_steps,
        parsed_bounds,
    )
    return {"task_id": task_id, "status_url": f"/api/tasks/{task_id}"}


def _run_inference_task(
    task_id: str,
    cancel_event: Event,
    borehole_volume: np.ndarray,
    geo_volume: np.ndarray,
    gravity: np.ndarray | None,
    magnetics: np.ndarray | None,
    borehole_specs,
    azimuth_degrees: float,
    vertical_degrees: float,
    num_classes: int,
    num_generations: int,
    ode_steps: int,
    parsed_bounds,
):
    task = TASKS[task_id]
    task["status"] = "running"

    def update_progress(progress: dict[str, Any]):
        task["progress"].update(progress)

    try:
        metadata = run_inference(
            model_path=MODEL_PATH,
            run_root=RUN_ROOT,
            borehole_volume=borehole_volume,
            geo_volume=geo_volume,
            gravity=gravity,
            magnetics=magnetics,
            boreholes=borehole_specs,
            azimuth_degrees=azimuth_degrees,
            vertical_degrees=vertical_degrees,
            num_classes=num_classes,
            num_generations=num_generations,
            ode_steps=ode_steps,
            bounds=parsed_bounds,
            cancel_event=cancel_event,
            progress_callback=update_progress,
        )
        run_id = metadata["run_id"]
        metadata["download_zip_url"] = f"/api/runs/{run_id}/outputs.zip"
        metadata["files"] = [
            {"name": filename, "url": f"/api/runs/{run_id}/files/{filename}"}
            for filename in metadata["files"]
        ]
        task["metadata"] = metadata
        task["status"] = "cancelled" if metadata.get("cancelled") else "completed"
    except Exception as exc:
        task["status"] = "failed"
        task["error"] = str(exc)


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    task = TASKS.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return _serialize_task(task)


@app.post("/api/tasks/{task_id}/cancel")
def cancel_task(task_id: str):
    task = TASKS.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    if task["status"] in ("completed", "failed", "cancelled"):
        return _serialize_task(task)
    task["cancel_requested"] = True
    task["cancel_event"].set()
    task["status"] = "cancelling"
    return _serialize_task(task)


def _serialize_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "status": task["status"],
        "cancel_requested": task["cancel_requested"],
        "progress": task["progress"],
        "metadata": task["metadata"],
        "error": task["error"],
    }


@app.get("/api/runs/{run_id}/files/{filename}")
def get_run_file(run_id: str, filename: str):
    file_path = _resolve_run_file(run_id, filename)
    return FileResponse(file_path, filename=filename)


@app.get("/api/runs/{run_id}/outputs.zip")
def get_run_zip(run_id: str):
    zip_path = RUN_ROOT / run_id / "outputs.zip"
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Run archive not found.")
    return FileResponse(zip_path, filename=f"{run_id}_outputs.zip")


@app.get("/api/runs/{run_id}/preview/{filename}")
def preview_run_file(run_id: str, filename: str):
    file_path = _resolve_run_file(run_id, filename)
    suffix = file_path.suffix.lower()

    if suffix == ".json":
        return JSONResponse(file_path.read_text(encoding="utf-8"))

    if suffix != ".npy":
        raise HTTPException(status_code=400, detail="Preview is only supported for .npy and .json files.")

    try:
        arr = np.load(file_path, allow_pickle=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not load .npy file: {exc}") from exc

    finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.number) else np.array([])
    summary = {
        "filename": filename,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "finite_count": int(finite.size),
        "nan_count": int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0,
    }

    if np.issubdtype(arr.dtype, np.number) and arr.size <= 1_000_000:
        values, counts = np.unique(arr[np.isfinite(arr)], return_counts=True)
        summary["value_counts"] = [
            {"value": float(value), "count": int(count)}
            for value, count in zip(values[:64], counts[:64])
        ]

    if arr.ndim >= 2:
        preview = _make_slice_preview(arr)
        summary["slice_preview"] = preview

    return summary


def _resolve_run_file(run_id: str, filename: str) -> Path:
    run_dir = (RUN_ROOT / run_id).resolve()
    file_path = (run_dir / filename).resolve()
    if not str(file_path).startswith(str(run_dir)) or not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return file_path


def _make_slice_preview(arr: np.ndarray) -> dict:
    preview_arr = arr
    if arr.ndim == 3:
        preview_arr = arr[:, :, arr.shape[2] // 2]
    elif arr.ndim > 3:
        preview_arr = np.squeeze(arr)
        if preview_arr.ndim == 3:
            preview_arr = preview_arr[:, :, preview_arr.shape[2] // 2]
        elif preview_arr.ndim > 2:
            preview_arr = preview_arr.reshape(preview_arr.shape[-2], preview_arr.shape[-1])

    preview_arr = np.asarray(preview_arr)
    row_idx = np.linspace(0, preview_arr.shape[0] - 1, min(32, preview_arr.shape[0]), dtype=int)
    col_idx = np.linspace(0, preview_arr.shape[1] - 1, min(32, preview_arr.shape[1]), dtype=int)
    sampled = preview_arr[np.ix_(row_idx, col_idx)]
    sampled = np.where(np.isfinite(sampled), sampled, np.nan)
    return {
        "source": "middle_z_slice" if arr.ndim == 3 else "sampled_2d_view",
        "shape": list(sampled.shape),
        "values": sampled.tolist(),
    }
