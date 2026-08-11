#!/usr/bin/env node

import process from "node:process";
import {auditStopSignTrip, findStopSignTripDirs, resolveLocalDataRoot} from "./lib/stop-sign-data-audit.mjs";

const DEFAULT_API = "http://127.0.0.1:8080";
const options = parseArgs(process.argv.slice(2));
const api = options.api ?? DEFAULT_API;
const epochs = integerOption(options.epochs, 150, 1, 10_000);

await requestJSON(`${api}/processing/reconcile`, {method: "POST"});
await waitForProcessing(api, integerOption(options.timeout, 180_000, 1_000, 900_000));
const readiness = await requestJSON(`${api}/processing/readiness`);
if (!readiness.trainingReady) {
    throw new Error(
        `Training is not ready: locations=${readiness.trainingLocationCount ?? 0}, `
        + `eligibleTrips=${readiness.trainingEligibleTripCount ?? 0}, samples=${readiness.trainingSampleCount ?? 0}. `
        + "Collect one successful clip at another physical stop-sign location, then rerun this command.",
    );
}
const trainRunIds = uniqueStrings(readiness.suggestedTrainRunIds);
const valRunIds = uniqueStrings(readiness.suggestedValRunIds);
if (trainRunIds.length === 0 || valRunIds.length === 0) {
    throw new Error("Readiness did not provide a location-disjoint train/validation split");
}
const selectedRunIds = [...trainRunIds, ...valRunIds];
const tripDirs = await findStopSignTripDirs(resolveLocalDataRoot(options.root), selectedRunIds);
const auditedTrips = [];
for (const tripDir of tripDirs) {
    const audited = await auditStopSignTrip(tripDir);
    if (audited) {
        auditedTrips.push(audited);
    }
}
const auditedRunIds = new Set(auditedTrips.map((trip) => trip.runId));
const missingAudits = selectedRunIds.filter((runId) => !auditedRunIds.has(runId));
if (missingAudits.length > 0) {
    throw new Error(`Training split contains runs that did not pass RGB/telemetry audit: ${missingAudits.join(", ")}`);
}
const audit = {
    tripCount: auditedTrips.length,
    sampleCount: auditedTrips.reduce((count, trip) => count + trip.sampleCount, 0),
    locationCount: new Set(auditedTrips.map((trip) => trip.location)).size,
    variations: [...new Set(auditedTrips.flatMap((trip) => trip.variations))].sort(),
};

const spec = {
    name: options.name ?? `stop-sign-${new Date().toISOString().replaceAll(":", "-").slice(0, 19)}`,
    notes: "RGB temporal stop-sign v1: phase-balanced launch, approach/brake, stop/dwell, scripted release.",
    epochs,
    trainRunIds,
    valRunIds,
};
if (options["dry-run"] === "true") {
    console.log(JSON.stringify({status: "ready", spec, audit, readiness}, null, 2));
    process.exit(0);
}

try {
    await requestJSON(`${api}/training/config`);
} catch (error) {
    throw new Error(`Training service is unavailable. Run npm start, then retry. ${error.message}`);
}
const queued = await requestJSON(`${api}/training/jobs`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(spec),
});
console.log(JSON.stringify({status: "queued", spec, audit, jobs: queued.jobs}, null, 2));

async function waitForProcessing(apiBase, timeoutMs) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        const state = await requestJSON(`${apiBase}/processing/state`);
        if ((state.queued?.length ?? 0) === 0 && (state.active?.length ?? 0) === 0) {
            return;
        }
        await new Promise((resolve) => setTimeout(resolve, 500));
    }
    throw new Error(`Timed out after ${timeoutMs}ms waiting for dataset processing`);
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

function parseArgs(args) {
    const parsed = {};
    for (let index = 0; index < args.length; index += 1) {
        const token = args[index];
        if (token === "--dry-run") {
            parsed["dry-run"] = "true";
            continue;
        }
        if (!token.startsWith("--")) {
            throw new Error(`Unexpected argument ${token}`);
        }
        const [key, inline] = token.slice(2).split("=", 2);
        const value = inline ?? args[++index];
        if (!value || value.startsWith("--")) {
            throw new Error(`Missing value for --${key}`);
        }
        parsed[key] = value;
    }
    return parsed;
}

function integerOption(value, fallback, min, max) {
    const number = value == null ? fallback : Number(value);
    if (!Number.isSafeInteger(number) || number < min || number > max) {
        throw new Error(`Expected a whole number from ${min} to ${max}, got ${String(value)}`);
    }
    return number;
}

function uniqueStrings(value) {
    if (!Array.isArray(value) || value.some((item) => typeof item !== "string" || !item.trim())) {
        throw new Error("Training split must be a list of non-empty run ids");
    }
    return [...new Set(value)].sort();
}
