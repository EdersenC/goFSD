import {
    buildForwardBayCurriculum,
    buildForwardBayEvaluationPlan,
    DEFAULT_PARKING_EVALUATION_SEED,
    resolveParkingAttemptCount,
} from "./curriculum";
import {
    gtaForwardVector,
    gtaHeadingFromVector,
    gtaRightVector,
    headingDeltaDegrees,
    isVehicleFootprintInsideBay,
    poseFromLocalOffset,
    relativePose,
} from "./geometry";
import {
    ParkingEvaluationOutcomeTracker,
    ParkingHazardLatch,
    ParkingObservation,
    ParkingOutcomeTracker,
    terminalParkingPhase,
} from "./outcome";
import {ParkingGoal, ParkingPose, VehicleFootprint} from "./types";
import {
    CONTROL_TELEMETRY_SAMPLE_INTERVAL_MS,
    createNoEgoControlTelemetry,
} from "../controlTelemetry";
import {
    PARKING_DISABLED_CONTROL_IDS,
    PARKING_EVALUATION_COLLISION_CLEAN_BUFFER_MS,
    PARKING_EVALUATION_SPAWN_SETTLE_MS,
    PARKING_POST_FLASH_CLEAN_FRAME_BUFFER_MS,
} from "./controls";
import {isActuatorStateNeutral, isInferenceCleanupComplete} from "../expertNeutral";
import {
    normalizeVehicleSteering,
    resolveSteeringFullLockRadians,
    SULTAN_STEERING_FULL_LOCK_RADIANS,
} from "../vehicleSteering";

const epsilon = 1e-9;

function test(name: string, body: () => void) {
    try {
        body();
        console.log(`ok - ${name}`);
    } catch (error) {
        console.error(`not ok - ${name}`);
        throw error;
    }
}

function assert(condition: boolean, message: string): asserts condition {
    if (!condition) {
        throw new Error(message);
    }
}

function assertApproximately(actual: number, expected: number, message: string) {
    assert(
        Math.abs(actual - expected) <= epsilon,
        `${message}: expected ${expected}, got ${actual}`
    );
}

function assertThrows(body: () => void, expectedMessage: string) {
    try {
        body();
    } catch (error) {
        assert(error instanceof Error, "expected an Error instance");
        assert(error.message.includes(expectedMessage), `unexpected error: ${error.message}`);
        return;
    }
    throw new Error("expected function to throw");
}

const target: ParkingPose = {coords: [120, -45, 8], heading: 90};
const footprint: VehicleFootprint = {halfWidthM: 0.9, halfLengthM: 2.2};

function buildGoal(): ParkingGoal {
    return {
        task: "parking",
        maneuver: "forward-bay",
        target: {coords: [0, 0, 0], heading: 0},
        bay: {widthM: 3.4, lengthM: 6.5},
        tolerances: {
            maxCenterDistanceM: 0.65,
            maxHeadingErrorDeg: 5,
            maxSpeedMps: 0.15,
            settledDurationMs: 1000,
        },
        attemptIndex: 0,
        attemptCount: 1,
        seed: "parking-test",
        startPose: {coords: [0, -10, 0], heading: 0},
        startOffset: {longitudinalM: -10, lateralM: 0, headingDeg: 0},
    };
}

function safeObservation(
    value: Pick<ParkingObservation, "nowMs" | "pose" | "speedMps"> & Partial<ParkingObservation>
): ParkingObservation {
    return {
        collision: false,
        reversing: false,
        onGround: true,
        pitchDeg: 0,
        rollDeg: 0,
        stopRequested: false,
        ...value,
    };
}

test("GTA cardinal headings map to native XY axes", () => {
    const north = gtaForwardVector(0);
    const west = gtaForwardVector(90);
    const south = gtaForwardVector(180);
    const east = gtaForwardVector(270);
    assertApproximately(north[0], 0, "heading 0 x");
    assertApproximately(north[1], 1, "heading 0 y");
    assertApproximately(west[0], -1, "heading 90 x");
    assertApproximately(west[1], 0, "heading 90 y");
    assertApproximately(south[1], -1, "heading 180 y");
    assertApproximately(east[0], 1, "heading 270 x");
    assertApproximately(gtaRightVector(0)[0], 1, "heading 0 right x");
});

test("heading/vector conversions round trip", () => {
    for (const heading of [0, 30, 90, 179, 270, 359]) {
        const roundTripped = gtaHeadingFromVector(gtaForwardVector(heading));
        assertApproximately(headingDeltaDegrees(roundTripped, heading), 0, `heading ${heading}`);
    }
});

test("negative longitudinal offsets are behind every cardinal target heading", () => {
    const origin: ParkingPose = {coords: [0, 0, 0], heading: 0};
    const expectedStarts: Array<[number, number, number]> = [
        [0, -10, 0],
        [10, 0, 0],
        [0, 10, 0],
        [-10, 0, 0],
    ];
    for (const [index, heading] of [0, 90, 180, 270].entries()) {
        const start = poseFromLocalOffset({...origin, heading}, -10, 0);
        const expected = expectedStarts[index];
        assert(expected !== undefined, `missing expected cardinal start ${index}`);
        assertApproximately(start.coords[0], expected[0], `heading ${heading} behind x`);
        assertApproximately(start.coords[1], expected[1], `heading ${heading} behind y`);
    }
});

test("local offsets and relative poses are inverse transformations", () => {
    for (const heading of [0, 90, 217]) {
        const pose = poseFromLocalOffset({...target, heading}, -11.25, 1.75, -7);
        const relative = relativePose(pose, {...target, heading});
        assertApproximately(relative.longitudinalM, -11.25, `heading ${heading} longitudinal`);
        assertApproximately(relative.lateralM, 1.75, `heading ${heading} lateral`);
        assertApproximately(relative.headingDeg, 7, `heading ${heading} heading delta`);
    }
});

test("vehicle footprint containment honors target rotation and bay edges", () => {
    const bay = {widthM: 4, lengthM: 6};
    assert(isVehicleFootprintInsideBay(target, target, bay, footprint), "centered vehicle should fit");
    const inside = poseFromLocalOffset(target, 0.79, 0.09);
    assert(isVehicleFootprintInsideBay(inside, target, bay, footprint), "near-edge vehicle should fit");
    const outsideLength = poseFromLocalOffset(target, 0.81, 0);
    assert(!isVehicleFootprintInsideBay(outsideLength, target, bay, footprint), "length edge should fail");
    const outsideWidth = poseFromLocalOffset(target, 0, 1.11);
    assert(!isVehicleFootprintInsideBay(outsideWidth, target, bay, footprint), "width edge should fail");
});

test("curriculum is deterministic, bounded, and zero based", () => {
    const first = buildForwardBayCurriculum(target, 12, "same-seed");
    const second = buildForwardBayCurriculum(target, 12, "same-seed");
    assert(JSON.stringify(first) === JSON.stringify(second), "same seed should reproduce curriculum");
    assert(first.length === 12, "curriculum count should match request");
    assert(first.every((plan, index) => plan.attemptIndex === index), "attempt indices should be zero based");
    for (const plan of first) {
        const relative = relativePose(plan.startPose, target);
        assert(relative.longitudinalM < 0, "negative longitudinal offset must place the car behind the target");
        assertApproximately(
            relative.longitudinalM,
            plan.startOffset.longitudinalM,
            `attempt ${plan.attemptIndex} longitudinal offset`
        );
    }
    assert(
        JSON.stringify(first) !== JSON.stringify(buildForwardBayCurriculum(target, 12, "other-seed")),
        "different seeds should change curriculum"
    );
    assertThrows(() => resolveParkingAttemptCount(0), "1 through 50");
    assertThrows(() => resolveParkingAttemptCount(51), "1 through 50");
    assertThrows(() => resolveParkingAttemptCount(1.5), "1 through 50");
});

test("parking evaluation selects one deterministic curriculum start", () => {
    const first = buildForwardBayEvaluationPlan(target, "evaluation-seed");
    const repeated = buildForwardBayEvaluationPlan(target, "evaluation-seed");
    assert(JSON.stringify(first) === JSON.stringify(repeated), "evaluation plan must reproduce by seed");
    assert(first.plan.attemptIndex === 0, "evaluation must use the zero-based first attempt");

    const defaulted = buildForwardBayEvaluationPlan(target, "  ");
    assert(defaulted.seed === DEFAULT_PARKING_EVALUATION_SEED, "empty seed must use stable evaluation default");
});

test("parking succeeds only after continuously settled for the tolerance", () => {
    const tracker = new ParkingOutcomeTracker(buildGoal(), footprint, 100);
    const stablePose: ParkingPose = {coords: [0, 0, 0], heading: 0};
    const first = tracker.observe(safeObservation({
        nowMs: 100,
        pose: stablePose,
        speedMps: 0,
    }));
    assert(first.outcome === null && !first.state.parked, "first stable sample must not succeed");
    const early = tracker.observe(safeObservation({
        nowMs: 1099,
        pose: stablePose,
        speedMps: 0,
    }));
    assert(early.outcome === null && !early.state.parked, "999 ms must not succeed");
    const complete = tracker.observe(safeObservation({
        nowMs: 1100,
        pose: stablePose,
        speedMps: 0,
    }));
    assert(complete.outcome?.status === "succeeded", "1000 ms should succeed");
    assert(complete.outcome.success, "success outcome flag should be true");
});

test("parking evaluation waits for motion before timeout and settlement", () => {
    const tracker = new ParkingEvaluationOutcomeTracker(buildGoal(), footprint, 1000);
    const farPose: ParkingPose = {coords: [0, -10, 0], heading: 0};
    const waiting = tracker.observe(safeObservation({
        nowMs: 60_000,
        pose: farPose,
        speedMps: 0,
    }));
    assert(waiting.outcome === null && !tracker.isArmed(), "waiting at the start must not arm or timeout");

    const armed = tracker.observe(safeObservation({
        nowMs: 60_001,
        pose: farPose,
        speedMps: 0.05,
    }));
    assert(armed.outcome === null && tracker.isArmed(), "first meaningful motion must arm evaluation");

    const timedOut = tracker.observe(safeObservation({
        nowMs: 61_002,
        pose: farPose,
        speedMps: 0.05,
    }));
    assert(timedOut.outcome?.status === "timeout", "timeout must be relative to first motion");
});

test("collision, stop, invalid vehicle, and timeout are strict failures", () => {
    const goal = buildGoal();
    const farPose: ParkingPose = {coords: [0, -10, 0], heading: 0};
    const collision = new ParkingOutcomeTracker(goal, footprint, 0).observe(safeObservation({
        nowMs: 1,
        pose: farPose,
        speedMps: 1,
        collision: true,
    })).outcome;
    assert(collision?.status === "collision" && !collision.success, "collision must fail");

    const stopped = new ParkingOutcomeTracker(goal, footprint, 0).observe(safeObservation({
        nowMs: 1,
        pose: farPose,
        speedMps: 0,
        stopRequested: true,
    })).outcome;
    assert(stopped?.status === "stopped" && !stopped.success, "stop must fail");
    assert(terminalParkingPhase(stopped) === "failed", "stopped outcome must expose failed terminal phase");

    const invalid = new ParkingOutcomeTracker(goal, footprint, 0).observe(safeObservation({
        nowMs: 1,
        pose: null,
        speedMps: 0,
    })).outcome;
    assert(invalid?.status === "invalid_vehicle" && !invalid.success, "invalid vehicle must fail");

    const timeout = new ParkingOutcomeTracker(goal, footprint, 0).observe(safeObservation({
        nowMs: 45_000,
        pose: farPose,
        speedMps: 0,
    })).outcome;
    assert(timeout?.status === "timeout" && !timeout.success, "timeout must fail");
    assert(terminalParkingPhase(timeout) === "failed", "timeout must expose failed terminal phase");
});

test("unsafe expert motion and pose are strict failures", () => {
    const goal = buildGoal();
    const farPose: ParkingPose = {coords: [0, -10, 0], heading: 0};
    const reversing = new ParkingOutcomeTracker(goal, footprint, 0).observe(safeObservation({
        nowMs: 1,
        pose: farPose,
        speedMps: 0.5,
        reversing: true,
    })).outcome;
    assert(reversing?.status === "reversing" && !reversing.success, "reverse motion must fail");

    const airborne = new ParkingOutcomeTracker(goal, footprint, 0).observe(safeObservation({
        nowMs: 1,
        pose: farPose,
        speedMps: 0,
        onGround: false,
    })).outcome;
    assert(airborne?.status === "off_ground" && !airborne.success, "off-ground motion must fail");

    const tilted = new ParkingOutcomeTracker(goal, footprint, 0).observe(safeObservation({
        nowMs: 1,
        pose: farPose,
        speedMps: 0,
        rollDeg: 5.01,
    })).outcome;
    assert(tilted?.status === "not_upright" && !tilted.success, "unsafe tilt must fail");
});

test("per-frame collision and reverse hazards remain latched", () => {
    const hazards = new ParkingHazardLatch(-0.1);
    hazards.observe(true, -0.2);
    hazards.observe(false, 1);
    assert(hazards.collisionDetected(), "one collision frame must remain latched");
    assert(hazards.reversingDetected(), "one reversing frame must remain latched");
});

test("expert input guards cover driving controls and require applied neutral state", () => {
    assert(
        CONTROL_TELEMETRY_SAMPLE_INTERVAL_MS === 50,
        "expert capture and live-control history must use the declared 50 ms cadence"
    );
    assert(
        PARKING_POST_FLASH_CLEAN_FRAME_BUFFER_MS >= 300,
        "post-flash clean-frame buffer must cover the eight-frame training history"
    );
    assert(PARKING_EVALUATION_SPAWN_SETTLE_MS >= 1000, "evaluation spawn must settle before scoring");
    assert(
        PARKING_EVALUATION_COLLISION_CLEAN_BUFFER_MS >= 300,
        "evaluation spawn must have a clean collision-history buffer before scoring"
    );
    for (const controlId of [59, 60, 63, 64, 71, 72, 75, 76]) {
        assert(
            (PARKING_DISABLED_CONTROL_IDS as readonly number[]).includes(controlId),
            `control ${controlId} must be disabled`
        );
    }
    assert(isActuatorStateNeutral({
        applied: {enabled: false, steer: 0, throttle: 0, brakePressureAvg: 0, handbrake: false},
    }), "disabled zero actuator should be neutral");
    assert(!isActuatorStateNeutral({
        applied: {enabled: true, steer: 0, throttle: 0, brakePressureAvg: 0, handbrake: false},
    }), "enabled actuator should not be accepted as neutral");
    assert(!isActuatorStateNeutral({
        applied: {enabled: false, steer: 0.1, throttle: 0, brakePressureAvg: 0, handbrake: false},
    }), "non-zero actuator should not be accepted as neutral");
    assert(
        !isInferenceCleanupComplete({state: "succeeded", active: true}),
        "terminal inference state must not bypass process cleanup"
    );
    assert(
        isInferenceCleanupComplete({state: "succeeded", active: false}),
        "inactive terminal inference state should be safe to hand off"
    );
    assert(
        isInferenceCleanupComplete({state: "idle", active: false}),
        "inactive idle inference should be safe to hand off"
    );
    assert(
        !isInferenceCleanupComplete({state: "running", active: false}),
        "a non-terminal inference state must not be treated as cleanup complete"
    );
});

test("physical wheel steering normalizes to the parking setpoint contract", () => {
    const fullLock = resolveSteeringFullLockRadians(42);
    const halfLeft = normalizeVehicleSteering(fullLock / 2, 42);
    assertApproximately(halfLeft.normalized, 0.5, "half-lock steering should normalize to 0.5");
    assertApproximately(halfLeft.wheelAngleRadians, fullLock / 2, "raw wheel angle should be retained");
    assertApproximately(halfLeft.fullLockRadians, fullLock, "handling full lock should be retained");

    const beyondRightLock = normalizeVehicleSteering(-2 * fullLock, 42);
    assert(beyondRightLock.normalized === -1, "normalized steering must clamp at full lock");
});

test("invalid handling steering lock uses the validated Sultan fallback", () => {
    for (const invalidLock of [Number.NaN, 0, 4.99, 90.01]) {
        assert(
            resolveSteeringFullLockRadians(invalidLock) === SULTAN_STEERING_FULL_LOCK_RADIANS,
            `handling lock ${String(invalidLock)} should use the Sultan fallback`
        );
    }
    const fallbackSample = normalizeVehicleSteering(SULTAN_STEERING_FULL_LOCK_RADIANS, Number.NaN);
    assert(fallbackSample.normalized === 1, "Sultan fallback full lock should normalize to 1");
});

test("no-ego heartbeat clears stale readiness and route state", () => {
    const telemetry = createNoEgoControlTelemetry(1234);
    assert(!telemetry.vehicleExists, "no-ego heartbeat must clear vehicle existence");
    assert(!telemetry.isInVehicle, "no-ego heartbeat must clear player readiness");
    assert(telemetry.currentSpeed === 0, "no-ego heartbeat must clear speed");
    assert(telemetry.vehicleModelHash === 0, "no-ego heartbeat must clear vehicle identity");
    assert(telemetry.routeDirectionUnknown === 1, "no-ego route should be explicitly unknown");
    assert(telemetry.routeForwardDelta === 0, "no-ego heartbeat must clear route forward delta");
    assert(telemetry.routeHeadingError === 0, "no-ego heartbeat must clear route heading error");
    assert(telemetry.routeDistance === 0, "no-ego heartbeat must clear route distance");
    assert(telemetry.gameTimeMs === 1234, "heartbeat should preserve sample time");
});

console.log("parking unit tests passed");
