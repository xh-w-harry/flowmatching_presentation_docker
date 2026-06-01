# Directional Borehole Inference Project

This project is self-contained under `project/`. It does not import code from the parent demo repository.

## Layout

- `src/backend`: FastAPI inference service on port `6052`
- `src/frontend`: static frontend served by Nginx on port `6051`
- `models/best_fm_multimodal.pth`: copied model weights
- `runs/`: inference outputs created at runtime

## Run

```bash
docker compose up --build
```

Open:

```text
http://localhost:6051
```

Backend health:

```text
http://localhost:6052/health
```

## Inputs

Required:

- `Borehole .npy`: sparse 3D categorical borehole volume used as the model condition.
- `GeoData .npy`: ground truth/evaluation volume. It is not used to build the borehole condition.
- `NUM_CLASSES`, `NUM_GENERATIONS`, `BOUNDS`.

Optional:

- `Gravity .npy`
- `Magnetics .npy`
- Borehole position/angle rows: optional metadata for the request. The inference condition comes from `Borehole .npy`.

If gravity or magnetics are not uploaded, the backend fills that model channel with zeros. The model still receives a 3-channel condition tensor to stay compatible with the copied weights.

## Outputs

Each run creates:

- `metadata.json`
- `input_boreholes.npy`
- `ground_truth.npy`
- `generation_XX.npy`
- `outputs.zip`

The frontend shows links to each file after inference completes.
