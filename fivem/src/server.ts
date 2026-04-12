//srv/server.ts

import fetch from "node-fetch";
import {log} from "./helper";
import {defaultScene, defaultSceneId, getLocalScene} from "./datasets";
import {normalizeScenePayload, SceneType} from "./sceneManger";
import {WaypointCompleted} from "./egoService";

console.log("[server] loaded");

type AggregatedTrip = Omit<WaypointCompleted, "vehicleData" | "chunkIndex" | "isTripComplete"> & {
    vehicleData: WaypointCompleted["vehicleData"]
};

type CaptureRequest = {
    requestId: string
    runId: string
    tripIndex?: number
    sceneId?: string
    sceneVariant?: string
    sceneName?: string
}

type CaptureResponse = {
    requestId: string
    success: boolean
    error?: string
    outputFile?: string
    logFile?: string
}

type RunIdParts = {
    safeRunId: string
    runFolder: string
    humanTime: string
}

type RunStoragePaths = {
    runsDir: string
    sceneDir: string
    sceneDirRelative: string
    runFile: string
    runTiming: RunIdParts
}

type TripStoragePaths = {
    tripDir: string
    tripDirRelative: string
    videoFile: string
    videoFileRelative: string
    logFile: string
    logFileRelative: string
    metadataFile: string
}

type ControlCommandType = "startScene" | "runAllScenes" | "endScene" | "endAllScenes";

type ControlCommand = {
    id: string
    type: ControlCommandType
    sceneName?: string
    createdAt?: string
}

type ControlStatusUpdate = {
    status: "idle" | "runningScene" | "runningAllScenes" | "stopping" | "error"
    activeSceneName?: string
    lastError?: string
}

type AvailableScene = {
    name: string
    label: string
}

const activeTripByRun = new Map<string, string>();
const pendingTrips = new Map<string, AggregatedTrip>();
let controlPollInFlight = false;
let lastSeenControlCommandId = "";
let availableScenesSyncInFlight = false;
let sceneListRequestInFlight = false;
let activeControlPlayerSource: number | null = null;
let controlConnectInFlight = false;

onNet("capture:startRequest", async (request: CaptureRequest) => {
    const playerSource = (global as any).source;
    rememberControlPlayerSource(playerSource);
    const response: CaptureResponse = {
        requestId: String(request?.requestId ?? ""),
        success: false
    };

    try {
        const validated = validateCaptureRequest(request);
        const sceneInfo = resolveSceneInfo(validated.sceneId, validated.sceneVariant, validated.sceneName);
        const runStorage = buildRunStoragePaths(
            resolveDataRoot(),
            sceneInfo.sceneId,
            sceneInfo.sceneVariant,
            validated.runId
        );
        const tripStorage = buildTripStoragePaths(resolveDataRoot(), runStorage, validated.tripIndex);

        const result = await captureApiRequest("/capture/start", {
            sourceId: "monitor-2",
            cropToWindow: false,
            outputFile: tripStorage.videoFileRelative
        });

        response.success = true;
        response.outputFile = result.outputFile;
        response.logFile = result.logFile;
    } catch (err: any) {
        response.error = err?.message ?? "failed to start capture";
    }

    emitNet("capture:startResponse", playerSource, response);
});

onNet("capture:stopRequest", async (request: CaptureRequest) => {
    const playerSource = (global as any).source;
    rememberControlPlayerSource(playerSource);
    const response: CaptureResponse = {
        requestId: String(request?.requestId ?? ""),
        success: false
    };

    try {
        validateCaptureRequest(request);
        const result = await captureApiRequest("/capture/stop", {});
        response.success = true;
        response.outputFile = result.outputFile;
        response.logFile = result.logFile;
    } catch (err: any) {
        response.error = err?.message ?? "failed to stop capture";
    }

    emitNet("capture:stopResponse", playerSource, response);
});

onNet("control:statusUpdate", async (update: ControlStatusUpdate) => {
    rememberControlPlayerSource((global as any).source);
    try {
        await apiRequest("/control/status", "POST", {
            status: update?.status ?? "idle",
            activeSceneName: update?.activeSceneName ?? "",
            lastError: update?.lastError ?? ""
        });
    } catch (err: any) {
        console.error(`[control] failed to push status update: ${err?.message ?? err}`);
    }
});

onNet("control:availableScenesResponse", async (scenes: AvailableScene[]) => {
    rememberControlPlayerSource((global as any).source);
    try {
        sceneListRequestInFlight = false;
        await syncAvailableScenes(Array.isArray(scenes) ? scenes : []);
    } catch (err: any) {
        console.error(`[control] failed to handle available scenes response: ${err?.message ?? err}`);
    }
});

onNet("control:registerClient", async () => {
    rememberControlPlayerSource((global as any).source);
    lastSeenControlCommandId = "";
    await stopCaptureOnReconnect();
    await connectControlSession();
    requestAvailableScenesFromClient();
});

onNet("ego:vehicleData", async (data: WaypointCompleted) => {
    const fs = require("fs");
    const path = require("path");
    const dataRoot = resolveDataRoot();
    const runStorage = buildRunStoragePaths(
        dataRoot,
        data.sceneId,
        data.sceneVariant,
        data.runId
    );
    const runsDir = runStorage.runsDir;
    const runFile = runStorage.runFile;
    const manifestFile = resolveManifestFile(dataRoot);
    const manifestDir = path.dirname(manifestFile);
    const tripStorage = buildTripStoragePaths(dataRoot, runStorage, data.tripIndex);
    const tripKey = `${data.runId}:${data.tripIndex}`;
    const runKey = data.runId;

    console.log(`[server] received chunk ${data.chunkIndex} for trip ${data.tripIndex} run ${data.runId} with ${data.vehicleData.length} points complete=${data.isTripComplete}`);

    if (!fs.existsSync(runsDir)) {
        fs.mkdirSync(runsDir, { recursive: true });
        console.log("Directory created:", runsDir);
    }

    if (!fs.existsSync(runStorage.sceneDir)) {
        fs.mkdirSync(runStorage.sceneDir, { recursive: true });
        console.log("Directory created:", runStorage.sceneDir);
    }

    if (!fs.existsSync(tripStorage.tripDir)) {
        fs.mkdirSync(tripStorage.tripDir, { recursive: true });
        console.log("Directory created:", tripStorage.tripDir);
    }

    if (!fs.existsSync(manifestDir)) {
        fs.mkdirSync(manifestDir, { recursive: true });
        console.log("Directory created:", manifestDir);
    }

    if (!fs.existsSync(manifestFile)) {
        try {
            fs.writeFileSync(manifestFile, "");
            console.log("File created:", manifestFile);
        } catch (err) {
            console.error("Error creating file:", err);
            return;
        }
    }

    const previousTripKey = activeTripByRun.get(runKey);
    if (previousTripKey && previousTripKey !== tripKey) {
        flushTrip(previousTripKey, runStorage, manifestFile, fs, dataRoot);
    }

    activeTripByRun.set(runKey, tripKey);

    const existingTrip = pendingTrips.get(tripKey);
    if (existingTrip) {
        existingTrip.vehicleData.push(...data.vehicleData);
        existingTrip.endTime = data.endTime;
        existingTrip.chunkDurationMs = data.chunkDurationMs;
    } else {
        pendingTrips.set(tripKey, {
            runId: data.runId,
            sceneId: data.sceneId,
            sceneVariant: data.sceneVariant,
            tripIndex: data.tripIndex,
            chunkDurationMs: data.chunkDurationMs,
            syncTime: data.syncTime,
            endTime: data.endTime,
            fromDestination: data.fromDestination,
            toDestination: data.toDestination,
            vehicle: data.vehicle,
            vehicleData: [...data.vehicleData]
        });
    }

    if (data.isTripComplete) {
        flushTrip(tripKey, runStorage, manifestFile, fs, dataRoot);
        activeTripByRun.delete(runKey);
    }
});

function sanitizePathSegment(value: string): string {
    return value.replace(/[^a-zA-Z0-9._-]/g, "_");
}

function rememberControlPlayerSource(value: unknown) {
    const parsed = Number(value);
    if (Number.isFinite(parsed) && parsed > 0) {
        activeControlPlayerSource = parsed;
    }
}

function resolveProjectRoot(): string {
    return process.env.AWESOME_PROJECT_ROOT
        ?? (process.platform === "win32"
            ? "C:\\Users\\theki\\GolandProjects\\awesomeProject"
            : "/mnt/c/Users/theki/GolandProjects/awesomeProject");
}

function resolveDataRoot(): string {
    const path = require("path");
    return process.env.VEHICLE_DATA_DIR
        ?? path.join(resolveProjectRoot(), "backend", "data");
}

function resolveManifestFile(dataRoot: string): string {
    const path = require("path");
    return process.env.VEHICLE_DATA_FILE
        ?? path.join(dataRoot, "runs.jsonl");
}

function parseRunIdParts(runId: string): RunIdParts {
    const safeRunId = sanitizePathSegment(runId);
    const match = safeRunId.match(/^(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})(AM|PM)(?:_[a-z0-9]+)?$/i);
    if (!match) {
        return {
            safeRunId,
            runFolder: safeRunId || "legacy",
            humanTime: safeRunId
        };
    }

    const [, date, hour, minute, second, meridiem] = match;
    const amPm = meridiem.toUpperCase();
    const runFolder = safeRunId;
    return {
        safeRunId,
        runFolder,
        humanTime: `${date} ${hour}:${minute}:${second} ${amPm}`
    };
}

function buildRunStoragePaths(dataRoot: string, sceneId: string, sceneVariant: string, runId: string): RunStoragePaths {
    const path = require("path");
    const runTiming = parseRunIdParts(runId);
    const sceneFolder = sanitizePathSegment(`${sceneId}_${sceneVariant}`);

    const runsDir = path.join(
        dataRoot,
        "runs",
        runTiming.runFolder
    );
    const sceneDir = path.join(runsDir, sceneFolder);
    const sceneDirRelative = `runs/${runTiming.runFolder}/${sceneFolder}`;

    const runFile = path.join(sceneDir, "run.jsonl");

    return {
        runsDir,
        sceneDir,
        sceneDirRelative,
        runFile,
        runTiming
    };
}

function buildTripStoragePaths(dataRoot: string, runStorage: RunStoragePaths, tripIndex: number): TripStoragePaths {
    const path = require("path");
    const tripSuffix = String(tripIndex).padStart(3, "0");
    const tripDirRelative = `${runStorage.sceneDirRelative}/trip-${tripSuffix}`;
    const videoFileRelative = `${tripDirRelative}/video.mkv`;
    return {
        tripDir: path.join(runStorage.sceneDir, `trip-${tripSuffix}`),
        tripDirRelative,
        videoFile: path.join(runStorage.sceneDir, `trip-${tripSuffix}`, "video.mkv"),
        videoFileRelative,
        logFile: path.join(runStorage.sceneDir, `trip-${tripSuffix}`, "video.log"),
        logFileRelative: `${tripDirRelative}/video.log`,
        metadataFile: path.join(runStorage.sceneDir, `trip-${tripSuffix}`, "metadata.json")
    };
}

function validateCaptureRequest(request: CaptureRequest): CaptureRequest {
    const requestId = String(request?.requestId ?? "").trim();
    const runId = String(request?.runId ?? "").trim();
    const tripIndex = typeof request?.tripIndex === "number"
        ? request.tripIndex
        : Number(request?.tripIndex);

    if (!requestId) {
        throw new Error("missing requestId");
    }
    if (!runId) {
        throw new Error("missing runId");
    }
    if (!Number.isInteger(tripIndex) || tripIndex < 0) {
        throw new Error("missing tripIndex");
    }

    return {
        requestId,
        runId,
        tripIndex,
        sceneId: request?.sceneId,
        sceneVariant: request?.sceneVariant,
        sceneName: request?.sceneName
    };
}

function resolveSceneInfo(sceneId?: string, sceneVariant?: string, sceneName?: string) {
    const parsed = parseSceneName(sceneName);
    return {
        sceneId: sanitizePathSegment((sceneId || parsed.sceneId || "unknown-scene").trim()),
        sceneVariant: sanitizePathSegment((sceneVariant || parsed.sceneVariant || "default").trim())
    };
}

function parseSceneName(sceneName?: string): { sceneId: string; sceneVariant: string } {
    if (!sceneName || typeof sceneName !== "string") {
        return { sceneId: "unknown-scene", sceneVariant: "default" };
    }

    const [sceneId, sceneVariant] = sceneName.split(":");
    return {
        sceneId: sceneId || "unknown-scene",
        sceneVariant: sceneVariant || "default"
    };
}

async function captureApiRequest(endpoint: string, body: any): Promise<any> {
    return apiRequest(endpoint, "POST", body);
}

async function apiRequest(endpoint: string, method: "GET" | "POST", body?: any): Promise<any> {
    const baseUrl = (process.env.CAPTURE_API_URL || "http://127.0.0.1:8080").replace(/\/+$/, "");
    const url = `${baseUrl}${endpoint}`;
    const response = await fetch(url, method === "GET"
        ? {
            method
        }
        : {
            method,
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(body ?? {})
        });

    const text = await response.text();
    let parsed: any = {};
    if (text) {
        try {
            parsed = JSON.parse(text);
        } catch {
            parsed = { error: text };
        }
    }

    if (!response.ok) {
        const message = parsed?.error || `capture api ${endpoint} failed with status ${response.status}`;
        throw new Error(message);
    }

    return parsed;
}

async function syncAvailableScenes(scenes: AvailableScene[]) {
    if (availableScenesSyncInFlight) {
        return;
    }

    availableScenesSyncInFlight = true;
    try {
        await apiRequest("/control/scenes", "POST", {
            scenes
        });
    } catch (err: any) {
        console.error(`[control] failed to sync available scenes: ${err?.message ?? err}`);
    } finally {
        availableScenesSyncInFlight = false;
    }
}

async function connectControlSession() {
    if (controlConnectInFlight) {
        return;
    }

    controlConnectInFlight = true;
    try {
        const result = await apiRequest("/control/connect", "POST", {});
        lastSeenControlCommandId = "";
        console.log(`[control] reset session ${result?.sessionId ?? "unknown"}`);
    } catch (err: any) {
        console.error(`[control] failed to reset control session: ${err?.message ?? err}`);
    } finally {
        controlConnectInFlight = false;
    }
}

async function stopCaptureOnReconnect() {
    try {
        await captureApiRequest("/capture/stop", {});
        console.log("[control] stopped active capture during reconnect");
    } catch (err: any) {
        const message = String(err?.message ?? err ?? "");
        if (message.toLowerCase().includes("capture is not running")) {
            return;
        }
        console.error(`[control] failed to stop capture during reconnect: ${message}`);
    }
}

function requestAvailableScenesFromClient() {
    if (sceneListRequestInFlight) {
        return;
    }

    const playerSource = activeControlPlayerSource;
    if (playerSource === null) {
        return;
    }

    sceneListRequestInFlight = true;
    emitNet("control:requestAvailableScenes", playerSource);
    setTimeout(() => {
        sceneListRequestInFlight = false;
    }, 3000);
}

async function pollControlCommands() {
    if (controlPollInFlight) {
        return;
    }

    controlPollInFlight = true;
    try {
        const query = lastSeenControlCommandId
            ? `?lastSeenCommandId=${encodeURIComponent(lastSeenControlCommandId)}`
            : "";
        const result = await apiRequest(`/control/poll${query}`, "GET");
        const command = result?.command as ControlCommand | undefined;
        if (!command?.id) {
            return;
        }

        const playerSource = activeControlPlayerSource;
        if (playerSource === null) {
            return;
        }

        emitNet("control:executeCommand", playerSource, command);
        lastSeenControlCommandId = command.id;
    } catch (err: any) {
        console.error(`[control] poll failed: ${err?.message ?? err}`);
    } finally {
        controlPollInFlight = false;
    }
}

setImmediate(() => {
    requestAvailableScenesFromClient();
    void pollControlCommands();
});

setInterval(() => {
    void pollControlCommands();
}, 2000);

setInterval(() => {
    requestAvailableScenesFromClient();
}, 10000);

function flushTrip(tripKey: string, runStorage: RunStoragePaths, manifestFile: string, fs: any, dataRoot: string) {
    const trip = pendingTrips.get(tripKey);
    if (!trip) {
        return;
    }

    const tripStorage = buildTripStoragePaths(dataRoot, runStorage, trip.tripIndex);

    if (!fs.existsSync(tripStorage.tripDir)) {
        fs.mkdirSync(tripStorage.tripDir, { recursive: true });
    }

    const tripMetadata = {
        runId: trip.runId,
        runFolder: runStorage.runTiming.runFolder,
        sceneFolder: pathBasename(runStorage.sceneDir),
        runLocalTime: runStorage.runTiming.humanTime,
        sceneId: trip.sceneId,
        sceneVariant: trip.sceneVariant,
        tripIndex: trip.tripIndex,
        chunkDurationMs: trip.chunkDurationMs,
        syncTime: trip.syncTime,
        endTime: trip.endTime,
        fromDestination: trip.fromDestination,
        toDestination: trip.toDestination,
        vehicleDataPoints: trip.vehicleData.length,
        videoFile: tripStorage.videoFile,
        logFile: tripStorage.logFile
    };

    fs.writeFileSync(tripStorage.metadataFile, JSON.stringify(tripMetadata, null, 2) + "\n");

    const runFile = runStorage.runFile;
    fs.appendFileSync(runFile, JSON.stringify(trip) + "\n");

    const manifestLine = JSON.stringify({
        runId: trip.runId,
        runFolder: runStorage.runTiming.runFolder,
        sceneFolder: pathBasename(runStorage.sceneDir),
        runLocalTime: runStorage.runTiming.humanTime,
        sceneId: trip.sceneId,
        sceneVariant: trip.sceneVariant,
        tripIndex: trip.tripIndex,
        chunkDurationMs: trip.chunkDurationMs,
        syncTime: trip.syncTime,
        endTime: trip.endTime,
        fromDestination: trip.fromDestination,
        toDestination: trip.toDestination,
        vehicleDataPoints: trip.vehicleData.length,
        file: runFile,
        tripDir: tripStorage.tripDir,
        videoFile: tripStorage.videoFile,
        logFile: tripStorage.logFile,
        metadataFile: tripStorage.metadataFile
    }) + "\n";

    fs.appendFileSync(manifestFile, manifestLine);
    pendingTrips.delete(tripKey);
    console.log(`Stored trip ${trip.tripIndex} for run ${trip.runId} with ${trip.vehicleData.length} data points`);
}

function pathBasename(targetPath: string): string {
    const path = require("path");
    return path.basename(targetPath);
}
