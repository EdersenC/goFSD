import {ThemeProvider} from "@mui/material";
import {renderToStaticMarkup} from "react-dom/server";
import {captureScenePose, createStopSignPlan, stageStopSignCatalogLocation} from "./stop-sign-plan";
import {PhaseRail, resolvePhaseIndex} from "./stop-sign/PhaseRail";
import {PlanEditor} from "./stop-sign/PlanEditor";
import {TelemetryPanel} from "./stop-sign/TelemetryPanel";
import {StopSignCatalog} from "./stop-sign/StopSignCatalog";
import {operatorTheme} from "./theme";
import type {ControlState} from "./types";

assertEqual(resolvePhaseIndex("launching"), 0);
assertEqual(resolvePhaseIndex("approach_braking"), 1);
assertEqual(resolvePhaseIndex("stop_hold"), 1);
assertEqual(resolvePhaseIndex("release"), 2);

const phaseMarkup = render(<PhaseRail phase="stop_hold" />);
for (const label of ["Approach", "Brake → Stop", "Release → End", "scripted release"]) {
    assert(phaseMarkup.includes(label), `missing phase ${label}`);
}
assert(phaseMarkup.includes('data-phase="brake_stop"'), "brake-to-stop phase needs a stable render marker");
assert(phaseMarkup.includes('data-active="true"'), "current phase must be marked active");

let plan = stageStopSignCatalogLocation(createStopSignPlan(), {
    id: "gta-v-sign-0023", x: -1830.7, y: 3206.4, z: 31.8,
}).plan;
plan = captureScenePose(plan, 0, "startPose", {x: 100, y: 140, z: 30, heading: 0});
plan = captureScenePose(plan, 0, "egoStopPose", {x: 100, y: 200, z: 30, heading: 0});
plan = captureScenePose(plan, 0, "exitPose", {x: 100, y: 212, z: 30, heading: 0});
const editorMarkup = render(
    <PlanEditor
        plan={plan}
        activeEntryIndex={0}
        onChange={() => undefined}
        onSelectEntry={() => undefined}
        onCapture={() => undefined}
        onRemove={() => undefined}
        captureReady
        captureBusy={false}
    />,
);
for (const label of ["Capture Start", "Start", "Stop", "End", "Automatic seeded variants", "50 continuous runs", "Motion variance"]) {
    assert(editorMarkup.includes(label), `plan editor missing ${label}`);
}
assert(!/experience picker|guided runbook|choose a workflow/i.test(editorMarkup), "obsolete workflow selection copy must not render");

const catalogMarkup = render(<StopSignCatalog
    connected
    candidate={{id: "gta-v-sign-0023", model: "prop_sign_road_01a", kind: "stop", x: 1, y: 2, z: 3, tilted: false, sourceYmap: "test"}}
    savedScenes={plan.entries}
    onTeleport={() => undefined}
    onUseCandidate={() => undefined}
    onOpenSaved={() => undefined}
/>);
for (const label of ["Choose a stop sign", "Saved signs", "gta-v-sign-0023", "Start", "Stop", "End", "Search all signs", "Teleport", "only previews", "Open saved scene", "Open + go", "has not been added"]) {
    assert(catalogMarkup.includes(label), `stop-sign catalog missing ${label}`);
}

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
        stopSignConfirmationElapsedMs: 0,
        stopSignConfirmationTargetMs: 250,
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
