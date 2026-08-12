#!/usr/bin/env node

import {copyFile, mkdir, readFile, writeFile} from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const DEFAULT_API = "http://127.0.0.1:8080";
const DEFAULT_CATALOG = path.resolve("backend/cmd/web/stop-sign-locations.csv");
const DEFAULT_SOURCE = "monitor-2";
const POLL_MS = 250;

const options = parseArgs(process.argv.slice(2));
const apiBase = options.api ?? DEFAULT_API;
const catalog = await loadCatalog(options.catalog ?? DEFAULT_CATALOG);
let state = await fetchState(apiBase);
assertControlReady(state);
const location = selectLocation(catalog, options.id, state.telemetry);
const profile = options.profile ?? "calibrate";
const headingOffsetDeg = numberOption(options.heading, 0);
if (headingOffsetDeg !== 0 && headingOffsetDeg !== 180) {
    throw new Error("--heading must be 0 or 180");
}

console.log(`[probe] location=${location.id} profile=${profile} headingOffset=${headingOffsetDeg}`);
state = await ensureSetupCar(apiBase, state);
state = await calibrateLocation(apiBase, state, location, headingOffsetDeg);

const runKey = [
    new Date().toISOString().replaceAll(":", "-").replaceAll(".", "-"),
    location.id,
    profile,
].join("_");
const before = await captureEvidence(apiBase, options.source ?? DEFAULT_SOURCE, runKey, "calibrated");
await persistEvidence(before.outputFile, "calibrated-state.json", {
    runKey,
    location,
    profile,
    headingOffsetDeg,
    controlState: state,
});

if (profile === "calibrate") {
    console.log(JSON.stringify({status: "calibrated", runKey, location: location.id, snapshot: before.outputFile}, null, 2));
    process.exit(0);
}

const batch = await queueProbeBatch(apiBase, state, location, profile, options);
const completed = await waitForState(apiBase, `batch ${batch.planId}`, (candidate) => {
    const progress = candidate.runtime?.stopSignBatch;
    return progress?.batchId === batch.planId && ["completed", "failed", "stopped"].includes(progress.state)
        ? candidate
        : null;
}, numberOption(options.timeout, 150_000));
const outcome = completed.runtime?.stopSignBatch?.lastAttemptOutcome;
const after = await captureEvidence(apiBase, options.source ?? DEFAULT_SOURCE, runKey, "completed");
await persistEvidence(after.outputFile, "completed-state.json", {
    runKey,
    location,
    profile,
    headingOffsetDeg,
    batch,
    outcome,
    controlState: completed,
});
if (!outcome) {
    throw new Error(
        `Batch ${batch.planId} finished without a lastAttemptOutcome; runtime=${completed.runtime?.lastError || "unknown"}`,
    );
}
const sequence = await collectProcessedEvidence(apiBase, batch, before.outputFile, profile === "success");

const expectedSuccess = profile === "success";
if (outcome.success !== expectedSuccess) {
    throw new Error(
        `Probe profile ${profile} expected success=${expectedSuccess}, got status=${outcome.status} reason=${outcome.failureReason || "none"}`,
    );
}
console.log(JSON.stringify({
    status: "accepted",
    runKey,
    location: location.id,
    profile,
    outcome,
    snapshots: [before.outputFile, after.outputFile],
    video: sequence.videoFile,
    phaseSnapshots: sequence.phaseSnapshots,
}, null, 2));

function parseArgs(args) {
    const parsed = {};
    for (let index = 0; index < args.length; index += 1) {
        const token = args[index];
        if (!token.startsWith("--")) {
            throw new Error(`Unexpected argument ${token}`);
        }
        const [rawKey, inlineValue] = token.slice(2).split("=", 2);
        const next = inlineValue ?? args[index + 1];
        if (inlineValue == null) {
            index += 1;
        }
        if (next == null || next.startsWith("--")) {
            throw new Error(`Missing value for --${rawKey}`);
        }
        parsed[rawKey] = next;
    }
    return parsed;
}

async function loadCatalog(file) {
    const raw = await readFile(file, "utf8");
    const lines = raw.split(/\r?\n/).filter(Boolean);
    const headers = parseCSVLine(lines.shift() ?? "");
    const rows = lines.map((line) => Object.fromEntries(headers.map((header, index) => [header, parseCSVLine(line)[index] ?? ""])));
    return rows.map((row) => ({
        id: row.id,
        x: Number(row.x),
        y: Number(row.y),
        z: Number(row.z),
        model: row.model,
    })).filter((row) => row.id && [row.x, row.y, row.z].every(Number.isFinite));
}

function parseCSVLine(line) {
    const cells = [];
    let current = "";
    let quoted = false;
    for (let index = 0; index < line.length; index += 1) {
        const character = line[index];
        if (character === '"') {
            if (quoted && line[index + 1] === '"') {
                current += '"';
                index += 1;
            } else {
                quoted = !quoted;
            }
        } else if (character === "," && !quoted) {
            cells.push(current);
            current = "";
        } else {
            current += character;
        }
    }
    cells.push(current);
    return cells;
}

function selectLocation(catalog, id, telemetry) {
    if (id) {
        const selected = catalog.find((entry) => entry.id === id);
        if (!selected) {
            throw new Error(`Unknown stop-sign catalog id ${id}`);
        }
        return selected;
    }
    const x = Number(telemetry?.positionX);
    const y = Number(telemetry?.positionY);
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
        return catalog[0];
    }
    return catalog.reduce((nearest, entry) => (
        distance2D(entry, {x, y}) < distance2D(nearest, {x, y}) ? entry : nearest
    ), catalog[0]);
}

async function ensureSetupCar(apiBase, initialState) {
    if (validSetupCar(initialState)) {
        return initialState;
    }
    const command = await postJSON(apiBase, "/control/command", {
        type: "startEgo",
        safetyEpoch: initialState.safetyEpoch,
    });
    console.log(`[probe] queued setup car command=${command.command.id}`);
    return waitForState(apiBase, "managed setup car", (state) => validSetupCar(state) ? state : null, 30_000);
}

async function calibrateLocation(apiBase, initialState, location, headingOffsetDeg) {
    const queuedAt = Date.now();
    const command = await postJSON(apiBase, "/control/command", {
        type: "probeStopSignTarget",
        safetyEpoch: initialState.safetyEpoch,
        stopSignProbe: {
            catalogId: location.id,
            catalogPosition: {x: location.x, y: location.y, z: location.z},
            headingOffsetDeg,
        },
    });
    console.log(`[probe] queued location probe command=${command.command.id}`);
    return waitForState(apiBase, `calibration ${location.id}`, (state) => {
        failOnRuntimeError(state, "stop-sign probe");
        const telemetry = state.telemetry;
        const pose = telemetry?.stopSignPose;
        if (
            telemetry?.stopSignTargetConfigured
            && Number(telemetry.receivedAtMs) >= queuedAt - 1000
            && distance2D(telemetryPosition(telemetry), location) <= 30
            && distance2D(pose, telemetryPosition(telemetry)) <= 1
            && Math.abs(Number(telemetry.pitchDeg)) <= 5
            && Math.abs(Number(telemetry.rollDeg)) <= 5
        ) {
            return state;
        }
        return null;
    }, 30_000);
}

async function queueProbeBatch(apiBase, state, location, profile, options) {
    if (profile !== "success" && profile !== "failure") {
        throw new Error("--profile must be calibrate, success, or failure");
    }
    const signPose = state.telemetry?.stopSignPose;
    if (!signPose) {
        throw new Error("Probe calibration did not publish stopSignPose");
    }
    const suffix = `${Date.now()}-${profile}`;
    const settings = profile === "success"
        ? {startDistanceM: 35, targetSpeedMps: 5, stopDistanceM: 4, egoCenterOffsetM: 2.5}
        : {startDistanceM: 5, targetSpeedMps: 8, stopDistanceM: 0.5, egoCenterOffsetM: 0.5};
    const request = {
        safetyEpoch: state.safetyEpoch,
        id: `probe-${location.id}-${suffix}`,
        seed: `probe:${location.id}:${suffix}`,
        entries: [{
            id: location.id,
            signPose,
            ...settings,
            exitDistanceM: 8,
            stopConfirmationMs: numberOption(options["stop-confirmation"], 250),
            attemptCount: 1,
            weather: "EXTRASUNNY",
            time: {hour: 12, minute: 0},
            vehicle: {model: "sultan", color: {r: 26, g: 86, b: 219}},
            variations: [{id: profile}],
        }],
    };
    const response = await postJSON(apiBase, "/control/stop-sign-batches", request);
    console.log(`[probe] queued batch=${response.planId} fingerprint=${response.planFingerprint}`);
    return response;
}

async function captureEvidence(apiBase, sourceId, runKey, label) {
    const result = await postJSON(apiBase, "/capture/snapshot", {
        sourceId,
        outputFile: `probe-evidence/${runKey}/${label}.png`,
        cropToWindow: false,
    });
    console.log(`[probe] snapshot=${result.outputFile}`);
    return result;
}

async function persistEvidence(snapshotPath, filename, value) {
    const localSnapshotPath = windowsPathToLocal(snapshotPath);
    const directory = path.dirname(localSnapshotPath);
    await mkdir(directory, {recursive: true});
    await writeFile(path.join(directory, filename), `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

async function collectProcessedEvidence(apiBase, batch, calibratedSnapshotPath, requireTrainingSamples) {
    const runFragment = sanitizeRunFragment(batch.planId);
    const run = await waitForValue(`processed run ${runFragment}`, async () => {
        const body = await requestJSON(`${apiBase}/data/runs`);
        const candidate = body.runs?.find((entry) => entry.runId?.includes(runFragment));
        const trip = candidate?.scenes?.[0]?.trips?.[0];
        return candidate && trip?.processingState === "completed" ? candidate : null;
    }, 30_000);
    const scene = run.scenes[0];
    const trip = scene.trips[0];
    const evidenceDir = path.dirname(windowsPathToLocal(calibratedSnapshotPath));
    const dataRoot = path.dirname(path.dirname(evidenceDir));
    const tripDir = path.join(dataRoot, "runs", run.runId, scene.sceneKey, trip.tripName);
    const processing = JSON.parse(await readFile(path.join(tripDir, "processing.json"), "utf8"));
    if (requireTrainingSamples && (!Number.isSafeInteger(processing.sampleCount) || processing.sampleCount <= 0)) {
        throw new Error(
            `Processed probe produced no training samples: ${JSON.stringify(processing.zeroSampleReasons ?? {})}`,
        );
    }

    const rows = (await readFile(path.join(tripDir, "dataset.jsonl"), "utf8"))
        .split(/\r?\n/)
        .filter(Boolean)
        .map((line) => JSON.parse(line));
    const phaseSnapshots = {};
    for (const phase of ["accelerate", "cruise_approach", "decelerate", "stop_hold", "release"]) {
        const matches = rows.filter((row) => row.phase === phase && row.frame_paths?.length);
        if (matches.length === 0) {
            continue;
        }
        const representative = matches[Math.floor(matches.length / 2)];
        const relativeFrame = representative.frame_paths.at(-1);
        const sourceFrame = path.resolve(tripDir, relativeFrame);
        if (!sourceFrame.startsWith(`${path.resolve(tripDir)}${path.sep}`)) {
            throw new Error(`Dataset frame escaped the trip directory: ${relativeFrame}`);
        }
        const destination = path.join(evidenceDir, `${phase}.jpg`);
        await copyFile(sourceFrame, destination);
        phaseSnapshots[phase] = destination;
    }
    await addReleaseTailEvidence(tripDir, trip.tripIndex, processing.frameCount, evidenceDir, phaseSnapshots);
    const metadata = JSON.parse(await readFile(path.join(tripDir, "metadata.json"), "utf8"));
    const evidence = {
        runId: run.runId,
        tripDir,
        videoFile: metadata.videoFile,
        sampleCount: processing.sampleCount,
        zeroSampleReasons: processing.zeroSampleReasons ?? {},
        phaseSnapshots,
    };
    await writeFile(path.join(evidenceDir, "sequence-state.json"), `${JSON.stringify(evidence, null, 2)}\n`, "utf8");
    return evidence;
}

async function addReleaseTailEvidence(tripDir, tripIndex, frameCount, evidenceDir, phaseSnapshots) {
    if (phaseSnapshots.release || !Number.isSafeInteger(frameCount) || frameCount <= 0) {
        return;
    }
    const records = (await readFile(path.join(path.dirname(tripDir), "run.jsonl"), "utf8"))
        .split(/\r?\n/)
        .filter(Boolean)
        .map((line) => JSON.parse(line));
    const record = records.find((entry) => entry.tripIndex === tripIndex);
    const lastPhase = record?.vehicleData?.at(-1)?.stopSignPhase;
    if (lastPhase !== "release") {
        return;
    }
    const frameIndex = Math.max(1, frameCount - 20);
    const source = path.join(tripDir, "frames", `${String(frameIndex).padStart(6, "0")}.jpg`);
    const destination = path.join(evidenceDir, "release.jpg");
    await copyFile(source, destination);
    phaseSnapshots.release = destination;
}

function sanitizeRunFragment(value) {
    return value.trim().replace(/[^a-zA-Z0-9_-]+/g, "-").slice(0, 40);
}

async function waitForValue(label, operation, timeoutMs) {
    const deadline = Date.now() + timeoutMs;
    let latestError;
    while (Date.now() < deadline) {
        try {
            const value = await operation();
            if (value) {
                return value;
            }
        } catch (error) {
            latestError = error;
        }
        await new Promise((resolve) => setTimeout(resolve, POLL_MS));
    }
    throw new Error(`Timed out after ${timeoutMs}ms waiting for ${label}: ${latestError?.message ?? "not found"}`);
}

function windowsPathToLocal(value) {
    const match = /^([a-zA-Z]):[\\/](.*)$/.exec(value);
    if (!match || process.platform === "win32") {
        return value;
    }
    return `/mnt/${match[1].toLowerCase()}/${match[2].replaceAll("\\", "/")}`;
}

async function fetchState(apiBase) {
    return requestJSON(`${apiBase}/control/state`);
}

async function postJSON(apiBase, pathname, body) {
    return requestJSON(`${apiBase}${pathname}`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body),
    });
}

async function requestJSON(url, init) {
    const response = await fetch(url, init);
    const text = await response.text();
    let body;
    try {
        body = text ? JSON.parse(text) : {};
    } catch {
        throw new Error(`${response.status} ${response.statusText}: ${text}`);
    }
    if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}: ${body.error ?? text}`);
    }
    return body;
}

async function waitForState(apiBase, label, predicate, timeoutMs) {
    const deadline = Date.now() + timeoutMs;
    let latest;
    while (Date.now() < deadline) {
        latest = await fetchState(apiBase);
        const accepted = predicate(latest);
        if (accepted) {
            return accepted;
        }
        await new Promise((resolve) => setTimeout(resolve, POLL_MS));
    }
    throw new Error(`Timed out after ${timeoutMs}ms waiting for ${label}; latest=${JSON.stringify(latest?.runtime ?? {})}`);
}

function assertControlReady(state) {
    if (!state.runtime?.fivemConnected) {
        throw new Error("FiveM is not connected");
    }
    if (state.runtime.appliedSafetyEpoch !== state.safetyEpoch || state.runtime.inFlightSafetyStarts !== 0) {
        throw new Error(`FiveM safety epoch is not synchronized: ${state.runtime.appliedSafetyEpoch}/${state.safetyEpoch}`);
    }
}

function validSetupCar(state) {
    return state.runtime?.fivemConnected
        && state.runtime.appliedSafetyEpoch === state.safetyEpoch
        && state.telemetry?.vehicleExists
        && state.telemetry?.isInVehicle
        && state.egoTelemetry?.valid;
}

function failOnRuntimeError(state, label) {
    if (state.runtime?.status === "error") {
        throw new Error(`${label} failed: ${state.runtime.lastError || "unknown FiveM error"}`);
    }
}

function telemetryPosition(telemetry) {
    return {x: Number(telemetry?.positionX), y: Number(telemetry?.positionY), z: Number(telemetry?.positionZ)};
}

function distance2D(a, b) {
    if (![a?.x, a?.y, b?.x, b?.y].every(Number.isFinite)) {
        return Number.POSITIVE_INFINITY;
    }
    return Math.hypot(a.x - b.x, a.y - b.y);
}

function numberOption(value, fallback) {
    if (value == null) {
        return fallback;
    }
    const number = Number(value);
    if (!Number.isFinite(number)) {
        throw new Error(`Expected a finite number, got ${value}`);
    }
    return number;
}
