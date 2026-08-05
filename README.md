# GTA Parking Lab

This repository is now centered on one honest first goal: collect clean forward-bay parking demonstrations, train a parking-aware planner, and inspect every attempt from one local control surface.

The first milestone is intentionally forward-only. Reverse and parallel parking need an explicit drive-direction/gear output plus a safe actuator interlock; the current steer/throttle/brake contract cannot represent reverse without ambiguity.

## Back in five minutes

1. Open [`parking-lab.code-workspace`](parking-lab.code-workspace).
2. Run the `Parking Lab: Doctor` task.
3. Install dependencies, then build and deploy the FiveM resource. On the current machine the resource is at the path shown below; change it if the server moves:

   ```bash
   npm --prefix fivem install
   export FIVEM_RESOURCE_DIR=/home/eddy/Fivem/data/cfx-server-data/resources/FSD
   npm --prefix fivem run build:deploy
   ```

4. Start the Windows backend from PowerShell. Live screen capture and the virtual controller are Windows-only:

   ```powershell
   .\scripts\dev-backend.ps1
   ```

5. In the FiveM server console, run `restart FSD` after deploying. Join the session, then open [http://127.0.0.1:8080](http://127.0.0.1:8080).

The workspace exposes **FiveM: Check deploy target**, **FiveM: Build and deploy resource**, and **FiveM: Watch and deploy resource** tasks. They prompt for the resource directory and default to the current machine's path. The deploy workflow validates an existing FiveM manifest, rejects filesystem roots and symbolic-link destinations, and copies only `dist/client.js`, `dist/server.js`, and `fxmanifest.lua`; it never restarts the server. `npm --prefix fivem run build` and the **FiveM: Build** task remain repository-only validation builds.

The backend binds to loopback by default because its control endpoints are not authenticated. Set `HOST` explicitly only when you intentionally need remote access on a trusted network.

The Python model server is not required to collect expert demonstrations. Start it later when you want the Models page, training jobs, or inference:

```powershell
.\scripts\dev-model-server.ps1
```

If you use a custom data root, pass the same `-DataRoot` to both PowerShell launchers (or set `FSD_DATA_ROOT` before using the workspace tasks). The WSL launchers accept either `/mnt/...` or Windows drive syntax and pass a Windows-native path to both services. That single root now owns captures, training jobs, checkpoints, and inference sample lookup.

## First parking session

The Parking tab is the default workspace and keeps the workflow linear:

1. Select **Start Setup Car**.
2. Manually place the car in the exact final pose you want inside a parking bay.
3. Select **Calibrate Current Bay**. The current vehicle position and heading become the target; no coordinates need to be copied.
4. Choose `1–50` attempts and, optionally, a seed. Reusing a seed reproduces the same start-pose curriculum.
5. Select **Collect Attempts**. FiveM creates varied forward approaches and uses GTA's native parking task as the expert.
6. Watch longitudinal, lateral, heading, distance, alignment, and settled-state telemetry in the Parking tab. **Stop & Hold** ends the active run safely.

A successful attempt must finish inside the calibrated bay, aligned with the goal, slow enough, settled for the required duration, and without a collision. Collision, timeout, stop, and invalid-vehicle outcomes remain in metadata for diagnosis. The Python dataset loader excludes those unsuccessful attempts from expert training by default.

## Data layout

Set `FSD_DATA_ROOT` when you do not want the Windows default of `S:\fsd_fivem_data`.

```text
<data-root>/runs/<run-id>/parking-forward-bay_default/
├── run.jsonl
├── trip-000/
│   ├── video.mkv
│   ├── video.log
│   ├── metadata.json
│   ├── processing.json
│   ├── dataset.jsonl
│   └── frames/
└── trip-001/
```

`metadata.json` and `run.jsonl` record the calibrated goal, deterministic start offset, tolerances, attempt index/seed, and final parking outcome. The existing Data page and `process-runs` command continue to work because parking attempts use the standard top-level `trip-*` format.

Every fresh sample records `Steering` as the normalized controller command contract in `[-1, 1]`. For diagnostics, `wheelAngle` retains FiveM's raw physical front-wheel angle in radians and `wheelSteeringFullLock` records the per-vehicle full-lock angle in radians used for normalization.

## Training from a clean slate

The checked-in [`fsd_trainer/train_config.toml`](fsd_trainer/train_config.toml) has no old checkpoint or run IDs. It enables target presence, current speed, and signed longitudinal/lateral/heading error inputs; caps speed at parking pace; and disables road-turn oversampling.

Create the Python environment if the existing local `.venv` is unavailable:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r fsd_trainer\requirements.txt
```

For GPU training, install the PyTorch build matching the machine's CUDA runtime before the remaining requirements. Then:

1. Start the model server.
2. Use **Data** to inspect/process collected runs.
3. Use **Models** to select train and validation run IDs and queue training.
4. Load a resulting checkpoint explicitly from **Advanced**.
5. Return to **Parking**, reuse a seed if you want a direct comparison, and select **Prepare Model Evaluation Start**.
6. In **Advanced**, select **Start Inference**, then return to **Parking** to watch the live score and terminal result.
7. Stop the completed inference session and prepare a fresh evaluation start before the next attempt.

Inference refuses to start unless the virtual controller is ready, telemetry is fresh, the checkpoint exposes the parking state inputs/control heads, and the car is inside the complete forward-curriculum envelope. A success, collision, reverse motion, unsafe pose, stale or misaligned telemetry, model error, timeout, or manual stop latches a safe vehicle hold before model control ends.

## Components

| Area | Responsibility |
|---|---|
| `fivem/` | Vehicle setup, bay calibration, deterministic parking curriculum, expert execution, telemetry, and trip metadata |
| `backend/` | Capture/process API, command queue, embedded Parking/Data/Models UI, inference bridge, and virtual-controller safety |
| `fsd_trainer/` | Dataset loading, parking state normalization, temporal planner training, checkpoint serving, and inference |
| `scripts/` | Environment checks, service launchers, and full validation |

The old general-driving controls remain under **Advanced** for reference and gradual reuse. They are no longer the default mental model or primary workflow.

## Validation

Run everything from the repository root:

```bash
bash scripts/validate.sh
```

Or run the layers independently:

```bash
npm --prefix fivem test
npm --prefix fivem run build
(cd backend && ../scripts/go.sh test ./...)
(cd fsd_trainer/src/gta_fsd && ../../../scripts/python.sh -B -m unittest discover -p 'test_*.py')
```

The live acceptance check still requires FiveM and the Windows capture stack: verify that left/right manual steering has the same sign in normalized telemetry, calibrate a real bay, collect at least one successful and one failed attempt, then verify both trip metadata and the successful attempt's processed frames/dataset. Finally, run one prepared model evaluation and confirm that success or a deliberately induced safety failure holds the vehicle and ends model control.
