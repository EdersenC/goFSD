#!/usr/bin/env node

import process from "node:process";
import {auditStopSignTrip, findStopSignTripDirs, resolveLocalDataRoot, STOP_SIGN_CLIP_STAGES} from "./lib/stop-sign-data-audit.mjs";

const options = parseArgs(process.argv.slice(2));
const dataRoot = resolveLocalDataRoot(options.root);
const runIds = options.runs ? options.runs.split(",").map((value) => value.trim()).filter(Boolean) : [];
const tripDirs = await findStopSignTripDirs(dataRoot, runIds);
const audited = [];
for (const tripDir of tripDirs) {
    const result = await auditStopSignTrip(tripDir);
    if (result) {
        audited.push(result);
    }
}
if (audited.length === 0) {
    throw new Error(`No successful, non-empty stop-sign datasets were found under ${dataRoot}`);
}

const phaseCounts = {};
for (const trip of audited) {
    for (const [phase, count] of Object.entries(trip.phaseCounts)) {
        phaseCounts[phase] = (phaseCounts[phase] ?? 0) + count;
    }
}
console.log(JSON.stringify({
    status: "aligned",
    dataRoot,
    tripCount: audited.length,
    sampleCount: audited.reduce((count, trip) => count + trip.sampleCount, 0),
    locationCount: new Set(audited.map((trip) => trip.location)).size,
    variations: [...new Set(audited.flatMap((trip) => trip.variations))].sort(),
    clipStageCounts: Object.fromEntries(STOP_SIGN_CLIP_STAGES.map((stage) => [stage, audited.filter((trip) => trip.clipStage === stage).length])),
    anchorPhaseCounts: phaseCounts,
    labeledPhases: [...new Set(audited.flatMap((trip) => trip.labeledPhases))],
    maxAlignmentDriftMs: Math.max(...audited.map((trip) => trip.alignmentDriftMs)),
    trips: audited,
}, null, 2));

function parseArgs(args) {
    const parsed = {};
    for (let index = 0; index < args.length; index += 1) {
        const token = args[index];
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
