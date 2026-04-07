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

const activeTripByRun = new Map<string, string>();
const pendingTrips = new Map<string, AggregatedTrip>();

onNet("ego:vehicleData", async (data: WaypointCompleted) => {
    const fs = require("fs");
    const path = require("path");
    const projectRoot = process.env.AWESOME_PROJECT_ROOT
        ?? (process.platform === "win32"
            ? "C:\\Users\\theki\\GolandProjects\\awesomeProject"
            : "/mnt/c/Users/theki/GolandProjects/awesomeProject");
    const dataRoot = process.env.VEHICLE_DATA_DIR
        ?? path.join(projectRoot, "backend", "data");
    const runsDir = path.join(
        dataRoot,
        "runs",
        sanitizePathSegment(data.sceneId),
        sanitizePathSegment(data.sceneVariant)
    );
    const runFile = path.join(runsDir, `${sanitizePathSegment(data.runId)}.jsonl`);
    const manifestFile = process.env.VEHICLE_DATA_FILE
        ?? path.join(dataRoot, "runs.jsonl");
    const manifestDir = path.dirname(manifestFile);
    const tripKey = `${data.runId}:${data.tripIndex}`;
    const runKey = data.runId;

    console.log(`[server] received chunk ${data.chunkIndex} for trip ${data.tripIndex} run ${data.runId} with ${data.vehicleData.length} points complete=${data.isTripComplete}`);

    if (!fs.existsSync(runsDir)) {
        fs.mkdirSync(runsDir, { recursive: true });
        console.log("Directory created:", runsDir);
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
        flushTrip(previousTripKey, runFile, manifestFile, fs);
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
        flushTrip(tripKey, runFile, manifestFile, fs);
        activeTripByRun.delete(runKey);
    }
});

function sanitizePathSegment(value: string): string {
    return value.replace(/[^a-zA-Z0-9._-]/g, "_");
}

function flushTrip(tripKey: string, runFile: string, manifestFile: string, fs: any) {
    const trip = pendingTrips.get(tripKey);
    if (!trip) {
        return;
    }

    fs.appendFileSync(runFile, JSON.stringify(trip) + "\n");

    const manifestLine = JSON.stringify({
        runId: trip.runId,
        sceneId: trip.sceneId,
        sceneVariant: trip.sceneVariant,
        tripIndex: trip.tripIndex,
        chunkDurationMs: trip.chunkDurationMs,
        syncTime: trip.syncTime,
        endTime: trip.endTime,
        fromDestination: trip.fromDestination,
        toDestination: trip.toDestination,
        vehicleDataPoints: trip.vehicleData.length,
        file: runFile
    }) + "\n";

    fs.appendFileSync(manifestFile, manifestLine);
    pendingTrips.delete(tripKey);
    console.log(`Stored trip ${trip.tripIndex} for run ${trip.runId} with ${trip.vehicleData.length} data points`);
}
