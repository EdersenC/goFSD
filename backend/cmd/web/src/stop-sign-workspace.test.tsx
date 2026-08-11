import {ThemeProvider} from "@mui/material";
import {renderToStaticMarkup} from "react-dom/server";
import {createStopSignPlan} from "./stop-sign-plan";
import {PhaseRail, resolvePhaseIndex} from "./stop-sign/PhaseRail";
import {PlanEditor} from "./stop-sign/PlanEditor";
import {TelemetryPanel} from "./stop-sign/TelemetryPanel";
import {operatorTheme} from "./theme";
import type {ControlState} from "./types";

assertEqual(resolvePhaseIndex("launching"), 0);
assertEqual(resolvePhaseIndex("approach_braking"), 1);
assertEqual(resolvePhaseIndex("dwell"), 2);
assertEqual(resolvePhaseIndex("release"), 3);

const phaseMarkup = render(<PhaseRail phase="dwell" />);
for (const label of ["Launch", "Approach / Brake", "Stop / Dwell", "Go / Release", "scripted release"]) {
    assert(phaseMarkup.includes(label), `missing phase ${label}`);
}
assert(phaseMarkup.includes('data-phase="stop"'), "stop phase needs a stable render marker");
assert(phaseMarkup.includes('data-active="true"'), "current phase must be marked active");

const plan = createStopSignPlan();
plan.entries[0]!.signPose = {x: 100, y: 200, z: 30, heading: 90};
plan.entries[0]!.variations.push({id: "rain", weather: "RAIN", targetSpeedMps: 6});
const editorMarkup = render(
    <PlanEditor
        plan={plan}
        onChange={() => undefined}
        onCalibrate={() => undefined}
        onClearCalibration={() => undefined}
        onApplyCalibration={() => undefined}
        calibrationReady
        calibrationBusy={false}
    />,
);
for (const label of ["Stop-sign plan", "Calibrate current sign", "Sign pose", "Base run", "Variants", "Attempts"]) {
    assert(editorMarkup.includes(label), `plan editor missing ${label}`);
}
assert(!/experience picker|guided runbook|choose a workflow/i.test(editorMarkup), "obsolete workflow selection copy must not render");

const control: ControlState = {
    safetyEpoch: 2,
    runtime: {
        status: "runningAllScenes",
        fivemConnected: true,
        appliedSafetyEpoch: 2,
        inFlightSafetyStarts: 0,
    },
    telemetry: {
        currentSpeed: 4.25,
        currentYaw: 90,
        brakeApplied: .42,
        throttleApplied: .1,
        vehicleExists: true,
        isInVehicle: true,
        stopSignTargetConfigured: true,
        stopSignPhase: "approach_braking",
        stopSignDistanceM: 16.5,
        stopLineDistanceM: 12.5,
        stopSignEgoStopPose: {x: 98, y: 200, z: 30, heading: 90},
        stopSignDwellElapsedMs: 0,
        stopSignDwellTargetMs: 5000,
        stopSignAttemptIndex: 2,
        stopSignAttemptCount: 10,
    },
    pendingCommands: [],
};
const telemetryMarkup = render(<TelemetryPanel control={control} />);
for (const value of ["FiveM linked", "approach_braking", "4.25m/s", "16.50m", "Attempt 2/10"]) {
    assert(telemetryMarkup.includes(value), `telemetry panel missing ${value}`);
}

const unsynchronizedMarkup = render(<TelemetryPanel control={{
    ...control,
    runtime: {...control.runtime, appliedSafetyEpoch: 1},
}} />);
assert(unsynchronizedMarkup.includes("Restart FSD"), "linked transport with a stale safety epoch must request a resource restart");

console.log("stop-sign workspace render and phase-state tests passed");

function render(node: React.ReactNode): string {
    return renderToStaticMarkup(<ThemeProvider theme={operatorTheme}>{node}</ThemeProvider>);
}

function assert(condition: unknown, message = "assertion failed"): asserts condition {
    if (!condition) {
        throw new Error(message);
    }
}

function assertEqual(actual: unknown, expected: unknown, message = `expected ${String(expected)}, got ${String(actual)}`) {
    assert(Object.is(actual, expected), message);
}
