import {lstat, readFile, readdir} from "node:fs/promises";
import path from "node:path";

export const STOP_SIGN_PHASES = ["accelerate", "cruise_approach", "decelerate", "stop_hold", "release"];
export const STOP_SIGN_CLIP_STAGES = ["approach", "brake_stop", "release"];
const STAGE_PHASES = {
    approach: new Set(["accelerate", "cruise_approach"]),
    brake_stop: new Set(["decelerate", "stop_hold"]),
    release: new Set(["release"]),
};
const CONTROL_FIELDS = [
    "expertDesiredSpeedMps",
    "expertThrottle",
    "expertBrake",
    "expertStopProbability",
    "expertGoProbability",
];

export async function auditStopSignTrip(tripDir) {
    const processing = await readJSON(path.join(tripDir, "processing.json"));
    const metadata = await readJSON(path.join(tripDir, "metadata.json"));
    if (!successfulStopSignMetadata(metadata)) {
        return null;
    }
    if (processing.state !== "completed" || !Number.isSafeInteger(processing.sampleCount) || processing.sampleCount <= 0) {
        return null;
    }
    const clipStage = metadata.stopSignGoal.clipStage;
    const allowedPhases = STAGE_PHASES[clipStage];
    const rows = await readJSONLines(path.join(tripDir, "dataset.jsonl"));
    const errors = [];
    const phaseCounts = Object.fromEntries(STOP_SIGN_PHASES.map((phase) => [phase, 0]));
    const labeledPhases = new Set();
    const variations = new Set();
    const locations = new Set();
    const anchorSkews = [];
    let previousVideoPTS = Number.NEGATIVE_INFINITY;
    let previousGameTime = Number.NEGATIVE_INFINITY;

    if (rows.length !== processing.sampleCount) {
        errors.push(`processing sampleCount=${processing.sampleCount}, dataset rows=${rows.length}`);
    }
    for (const [index, row] of rows.entries()) {
        const prefix = `row ${index + 1}`;
        if (row.task !== "stop-sign" || row.training_eligible !== true) {
            errors.push(`${prefix}: row is not marked as eligible stop-sign training data`);
        }
        if (row.clip_stage !== clipStage) {
            errors.push(`${prefix}: clip_stage ${String(row.clip_stage)} does not match metadata ${clipStage}`);
        }
        if (!STOP_SIGN_PHASES.includes(row.phase)) {
            errors.push(`${prefix}: invalid phase ${String(row.phase)}`);
            continue;
        }
        if (!allowedPhases.has(row.phase)) {
            errors.push(`${prefix}: phase ${row.phase} crossed the ${clipStage} clip boundary`);
        }
        phaseCounts[row.phase] += 1;
        labeledPhases.add(row.phase);
        for (const phase of row.label?.aux?.future_stop_sign_phases ?? []) {
            if (!STOP_SIGN_PHASES.includes(phase)) {
                errors.push(`${prefix}: invalid future phase ${String(phase)}`);
            } else if (!allowedPhases.has(phase)) {
                errors.push(`${prefix}: future phase ${phase} crossed the ${clipStage} clip boundary`);
            } else {
                labeledPhases.add(phase);
            }
        }
        if (!row.variation_id?.trim()) {
            errors.push(`${prefix}: variation_id is missing`);
        } else {
            variations.add(row.variation_id);
        }
        if (!row.scenario_split_group?.trim()) {
            errors.push(`${prefix}: scenario_split_group is missing`);
        } else {
            locations.add(row.scenario_split_group);
        }
        if (row.label?.aux?.stopSignPhase !== row.phase || row.telemetry_history?.at(-1)?.aux?.stopSignPhase !== row.phase) {
            errors.push(`${prefix}: anchor phase does not agree across row, label, and telemetry`);
        }
        validateMonotonicAnchor(row, prefix, errors, previousVideoPTS, previousGameTime);
        previousVideoPTS = row.anchor_video_pts;
        previousGameTime = row.anchor_game_time;
        anchorSkews.push(row.anchor_video_pts - row.anchor_game_time);
        await validateFrames(row, processing.imageOffsets, tripDir, prefix, errors);
        validateTelemetry(row, processing, prefix, errors);
        validateTargets(row, processing.futureOffsets, prefix, errors);
    }

    if (![...allowedPhases].some((phase) => phaseCounts[phase] > 0)) {
        errors.push(`${clipStage} clip contains no stage-appropriate anchor labels`);
    }
    if (locations.size !== 1) {
        errors.push(`trip must resolve to one physical location group, got ${locations.size}`);
    }
    const skewRangeMs = anchorSkews.length === 0
        ? Number.POSITIVE_INFINITY
        : (Math.max(...anchorSkews) - Math.min(...anchorSkews)) * 1000;
    if (skewRangeMs > 5) {
        errors.push(`RGB/game-time alignment offset drifted by ${skewRangeMs.toFixed(3)} ms`);
    }
    if (errors.length > 0) {
        throw new Error(`${tripDir}\n  ${errors.slice(0, 20).join("\n  ")}`);
    }
    return {
        tripDir,
        runId: metadata.runId,
        clipStage,
        sampleCount: rows.length,
        location: [...locations][0],
        variations: [...variations].sort(),
        phaseCounts,
        labeledPhases: STOP_SIGN_PHASES.filter((phase) => labeledPhases.has(phase)),
        alignmentOffsetMs: anchorSkews[0] * 1000,
        alignmentDriftMs: skewRangeMs,
    };
}

export async function findStopSignTripDirs(dataRoot, runIds = []) {
    const allowedRuns = new Set(runIds);
    const runsRoot = path.join(dataRoot, "runs");
    const tripDirs = [];
    for (const run of await directories(runsRoot)) {
        if (allowedRuns.size > 0 && !allowedRuns.has(run.name)) {
            continue;
        }
        const sceneDir = path.join(runsRoot, run.name, "stop-sign_temporal-v1");
        for (const trip of await directories(sceneDir, true)) {
            if (/^trip-\d+$/.test(trip.name)) {
                tripDirs.push(path.join(sceneDir, trip.name));
            }
        }
    }
    return tripDirs.sort();
}

export function resolveLocalDataRoot(raw = process.env.FSD_DATA_ROOT) {
    const value = raw?.trim().replace(/^["']|["']$/g, "");
    if (!value) {
        return process.platform === "win32" ? "S:\\fsd_fivem_data" : "/mnt/s/fsd_fivem_data";
    }
    const windows = /^([a-zA-Z]):[\\/](.*)$/.exec(value);
    if (windows && process.platform !== "win32") {
        return `/mnt/${windows[1].toLowerCase()}/${windows[2].replaceAll("\\", "/")}`;
    }
    return path.resolve(value);
}

function successfulStopSignMetadata(metadata) {
    return metadata.sceneId === "stop-sign"
        && metadata.sceneVariant === "temporal-v1"
        && metadata.stopSignGoal?.task === "stop-sign"
        && metadata.stopSignGoal?.contract === "stop-sign-goal.v2"
        && STOP_SIGN_CLIP_STAGES.includes(metadata.stopSignGoal?.clipStage)
        && metadata.stopSignOutcome?.success === true
        && metadata.stopSignOutcome?.status === "succeeded";
}

function validateMonotonicAnchor(row, prefix, errors, previousVideoPTS, previousGameTime) {
    if (!Number.isFinite(row.anchor_video_pts) || row.anchor_video_pts <= previousVideoPTS) {
        errors.push(`${prefix}: anchor_video_pts is not strictly increasing`);
    }
    if (!Number.isFinite(row.anchor_game_time) || row.anchor_game_time <= previousGameTime) {
        errors.push(`${prefix}: anchor_game_time is not strictly increasing`);
    }
}

async function validateFrames(row, imageOffsets, tripDir, prefix, errors) {
    if (!Array.isArray(row.frame_paths) || row.frame_paths.length !== imageOffsets?.length) {
        errors.push(`${prefix}: frame_paths does not match configured image offsets`);
        return;
    }
    const indices = [];
    for (const relative of row.frame_paths) {
        const match = /^frames\/(\d{6})\.jpg$/.exec(relative);
        if (!match) {
            errors.push(`${prefix}: invalid frame path ${String(relative)}`);
            continue;
        }
        const absolute = path.resolve(tripDir, relative);
        if (!absolute.startsWith(`${path.resolve(tripDir)}${path.sep}`)) {
            errors.push(`${prefix}: frame path escaped the trip directory`);
            continue;
        }
        try {
            const info = await lstat(absolute);
            if (!info.isFile() || info.size < 1) {
                errors.push(`${prefix}: frame is not a non-empty regular file: ${relative}`);
            }
        } catch {
            errors.push(`${prefix}: referenced frame is missing: ${relative}`);
        }
        indices.push(Number(match[1]));
    }
    for (let index = 1; index < indices.length; index += 1) {
        const actual = indices[index] - indices[index - 1];
        const expected = imageOffsets[index] - imageOffsets[index - 1];
        if (actual !== expected) {
            errors.push(`${prefix}: RGB frame spacing=${actual}, expected=${expected}`);
        }
    }
}

function validateTelemetry(row, processing, prefix, errors) {
    const history = row.telemetry_history;
    const future = row.telemetry_future;
    const requiredHistory = -Math.min(...processing.telemetryOffsets) + 1;
    const requiredFuture = Math.max(...processing.futureOffsets);
    if (!Array.isArray(history) || history.length < requiredHistory) {
        errors.push(`${prefix}: telemetry history is shorter than ${requiredHistory}`);
        return;
    }
    if (!Array.isArray(future) || future.length < requiredFuture) {
        errors.push(`${prefix}: telemetry future is shorter than ${requiredFuture}`);
        return;
    }
    const points = [...history, ...future];
    const expectedMs = processing.telemetrySampleIntervalMs;
    for (let index = 1; index < points.length; index += 1) {
        const delta = points[index]?.raw?.time - points[index - 1]?.raw?.time;
        if (!Number.isFinite(delta) || delta < expectedMs * 0.5 || delta > expectedMs * 1.5) {
            errors.push(`${prefix}: telemetry tick ${index} has ${String(delta)} ms spacing, expected about ${expectedMs}`);
            break;
        }
    }
}

function validateTargets(row, futureOffsets, prefix, errors) {
    const currentTarget = row.telemetry_history?.at(-1)?.control;
    for (const field of CONTROL_FIELDS) {
        if (!sameNumber(row.label?.control?.[field], currentTarget?.[field])) {
            errors.push(`${prefix}: label.control.${field} is not aligned to the anchor telemetry tick`);
        }
    }
    const arrays = {
        future_speed_targets_mps: "expertDesiredSpeedMps",
        future_stop_probabilities: "expertStopProbability",
        future_go_probabilities: "expertGoProbability",
    };
    for (const [labelName, telemetryName] of Object.entries(arrays)) {
        const values = row.label?.aux?.[labelName];
        if (!Array.isArray(values) || values.length !== futureOffsets.length) {
            errors.push(`${prefix}: ${labelName} does not match the configured horizon`);
            continue;
        }
        for (const [horizonIndex, offset] of futureOffsets.entries()) {
            const expected = row.telemetry_future?.[offset - 1]?.control?.[telemetryName];
            if (!sameNumber(values[horizonIndex], expected)) {
                errors.push(`${prefix}: ${labelName}[${horizonIndex}] is not aligned to telemetry offset ${offset}`);
                break;
            }
        }
    }
    const phases = row.label?.aux?.future_stop_sign_phases;
    if (!Array.isArray(phases) || phases.length !== futureOffsets.length) {
        errors.push(`${prefix}: future_stop_sign_phases does not match the configured horizon`);
    } else {
        for (const [horizonIndex, offset] of futureOffsets.entries()) {
            if (phases[horizonIndex] !== row.telemetry_future?.[offset - 1]?.aux?.stopSignPhase) {
                errors.push(`${prefix}: future phase ${horizonIndex} is not aligned to telemetry offset ${offset}`);
                break;
            }
        }
    }
    const throttle = Number(row.label?.control?.expertThrottle);
    const brake = Number(row.label?.control?.expertBrake);
    if (!Number.isFinite(throttle) || !Number.isFinite(brake) || throttle < 0 || brake < 0 || throttle * brake > 1e-8) {
        errors.push(`${prefix}: expert throttle and brake must be finite, non-negative, and mutually exclusive`);
    }
}

function sameNumber(left, right) {
    return Number.isFinite(left) && Number.isFinite(right) && Math.abs(left - right) <= 1e-6;
}

async function readJSON(file) {
    return JSON.parse(await readFile(file, "utf8"));
}

async function readJSONLines(file) {
    return (await readFile(file, "utf8")).split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
}

async function directories(parent, missingIsEmpty = false) {
    try {
        return (await readdir(parent, {withFileTypes: true})).filter((entry) => entry.isDirectory());
    } catch (error) {
        if (missingIsEmpty && error?.code === "ENOENT") {
            return [];
        }
        throw error;
    }
}
