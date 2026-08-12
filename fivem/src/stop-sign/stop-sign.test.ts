import assert from "node:assert/strict";
import {gtaForwardVector, poseAhead, poseBehind, relativeStopLinePose} from "./geometry";
import {
    classifyStopSignPhase,
    planStopSignExpert,
} from "./expert";
import {parseStopSignJobs} from "./batch";
import {logicalClipStageForPhase, STOP_SIGN_LOGICAL_CLIP_STAGES} from "./clip-plan";
import {
    advanceEarlyStopProgress,
    nextFixedIntervalDeadlineMs,
    validateStopSignAttemptVehicleState,
} from "./attempt-progress";
import {
    deriveStopSignRouteWaypoint,
    STOP_SIGN_ROUTE_WAYPOINT_MAX_FORWARD_M,
    STOP_SIGN_ROUTE_WAYPOINT_MAX_LATERAL_M,
    STOP_SIGN_ROUTE_WAYPOINT_MIN_FORWARD_M,
} from "./route-waypoint";

function testGeometryUsesGtaHeadingConvention() {
    const north = gtaForwardVector(0);
    assert(Math.abs(north[0]) < 1e-9 && Math.abs(north[1] - 1) < 1e-9);
    const west = gtaForwardVector(90);
    assert(Math.abs(west[0] + 1) < 1e-9 && Math.abs(west[1]) < 1e-9);
    assert.deepEqual(poseBehind({x: 10, y: 20, z: 3, heading: 0}, 4), {
        x: 10,
        y: 16,
        z: 3,
        heading: 0,
    });
    const relative = relativeStopLinePose(
        {x: 10, y: 8, z: 3, heading: 0},
        {x: 10, y: 10, z: 3, heading: 0},
    );
    assert.equal(relative.longitudinalM, -2);
    assert.equal(relative.lateralM, 0);
}

function testPhaseClassificationIsTemporalButNotSequenceDependent() {
    const common = {targetSpeedMps: 8, dtSeconds: 0.05};
    assert.equal(classifyStopSignPhase({
        ...common,
        remainingDistanceM: 30,
        measuredSpeedMps: 1,
        previousDesiredSpeedMps: 1,
    }), "accelerate");
    assert.equal(classifyStopSignPhase({
        ...common,
        remainingDistanceM: 30,
        measuredSpeedMps: 8,
        previousDesiredSpeedMps: 8,
    }), "cruise_approach");
    const brakingInput = {
        ...common,
        remainingDistanceM: 5,
        measuredSpeedMps: 8,
        previousDesiredSpeedMps: 8,
    };
    const phase = classifyStopSignPhase(brakingInput);
    const command = planStopSignExpert({
        ...brakingInput,
        phase,
        lateralErrorM: 0,
        headingErrorDeg: 0,
    });
    assert.equal(phase, "decelerate");
    assert(command.desiredSpeedMps < brakingInput.previousDesiredSpeedMps);
    assert(command.brake > 0);
}

function testApproachTransitionsFromThrottleToBrake() {
    const far = planStopSignExpert({
        phase: "accelerate",
        remainingDistanceM: 30,
        measuredSpeedMps: 1,
        targetSpeedMps: 8,
        previousDesiredSpeedMps: 1,
        dtSeconds: 0.05,
        lateralErrorM: 0,
        headingErrorDeg: 0,
    });
    const near = planStopSignExpert({
        phase: "decelerate",
        remainingDistanceM: 0.8,
        measuredSpeedMps: 5,
        targetSpeedMps: 8,
        previousDesiredSpeedMps: 5,
        dtSeconds: 0.05,
        lateralErrorM: 0,
        headingErrorDeg: 0,
    });
    assert(far.throttle > 0 && far.brake === 0);
    assert(near.brake > 0 && near.throttle === 0);
    assert(near.stopProbability > far.stopProbability);
}

function testStopAndGoLabelsAreExplicit() {
    const common = {
        remainingDistanceM: 0,
        measuredSpeedMps: 0,
        targetSpeedMps: 7,
        previousDesiredSpeedMps: 0,
        dtSeconds: 0.05,
        lateralErrorM: 0,
        headingErrorDeg: 0,
    };
    const stop = planStopSignExpert({...common, phase: "stop_hold"});
    const go = planStopSignExpert({...common, phase: "release"});
    assert.equal(stop.brake, 1);
    assert.equal(stop.stopProbability, 1);
    assert.equal(go.goProbability, 1);
    assert(go.throttle > 0);
}

function testEarlyStopScoringWaitsForActualDeparture() {
    const resetFrame = advanceEarlyStopProgress({
        departureObserved: false,
        speedMps: 0,
        distanceFromStartM: 0,
        stopStartedAtMs: null,
        remainingDistanceM: 40,
    });
    assert.deepEqual(resetFrame, {departureObserved: false, stoppedTooEarly: false});

    const launchFrame = advanceEarlyStopProgress({
        ...resetFrame,
        speedMps: 0.5,
        distanceFromStartM: 0.8,
        stopStartedAtMs: null,
        remainingDistanceM: 39.2,
    });
    assert.deepEqual(launchFrame, {departureObserved: true, stoppedTooEarly: false});

    const actualEarlyStop = advanceEarlyStopProgress({
        departureObserved: launchFrame.departureObserved,
        speedMps: 0,
        distanceFromStartM: 8,
        stopStartedAtMs: null,
        remainingDistanceM: 32,
    });
    assert.deepEqual(actualEarlyStop, {departureObserved: true, stoppedTooEarly: true});

    const validStop = advanceEarlyStopProgress({
        departureObserved: true,
        speedMps: 0,
        distanceFromStartM: 40,
        stopStartedAtMs: 1000,
        remainingDistanceM: 0,
    });
    assert.deepEqual(validStop, {departureObserved: true, stoppedTooEarly: false});
}

function testFixedIntervalSchedulerDoesNotAccumulateWorkTime() {
    assert.equal(nextFixedIntervalDeadlineMs(1000, 1020, 50), 1050);
    assert.equal(nextFixedIntervalDeadlineMs(1050, 1070, 50), 1100);
    assert.equal(nextFixedIntervalDeadlineMs(1100, 1190, 50), 1240);
}

function testAttemptPreflightRejectsUnsafeStarts() {
    const readyVehicle = {
        exists: true,
        playerIsDriver: true,
        engineRunning: true,
        onAllWheels: true,
        speedMps: 0,
        pitchDeg: 0,
        rollDeg: 0,
        collided: false,
    };
    assert.equal(validateStopSignAttemptVehicleState(readyVehicle), null);
    assert.equal(validateStopSignAttemptVehicleState({...readyVehicle, pitchDeg: 8}), null);
    assert.match(
        validateStopSignAttemptVehicleState({...readyVehicle, playerIsDriver: false}) ?? "",
        /driver seat/,
    );
    assert.match(
        validateStopSignAttemptVehicleState({...readyVehicle, collided: true}) ?? "",
        /spawned in contact/,
    );
    assert.match(
        validateStopSignAttemptVehicleState({...readyVehicle, rollDeg: 6}) ?? "",
        /not level/,
    );
}

function testExpandedJobsKeepBackendAndFiveMGeometryCoherent() {
    const signPose = {x: 100, y: 200, z: 8, heading: 90};
    const stopDistanceM = 4;
    const egoCenterOffsetM = 2.5;
    const startDistanceM = 40;
    const stopLinePose = poseBehind(signPose, stopDistanceM);
    const egoStopPose = poseBehind(stopLinePose, egoCenterOffsetM);
    const startPose = poseBehind(egoStopPose, startDistanceM);
    const exitDistanceM = 8;
    const exitPose = poseAhead(signPose, exitDistanceM);
    const job = {
        id: "alta:base",
        entryId: "alta",
        variationId: "base",
        signPose,
        stopLinePose,
        egoStopPose,
        startPose,
        exitPose,
        stopDistanceM,
        egoCenterOffsetM,
        startDistanceM,
        exitDistanceM,
        targetSpeedMps: 15,
        stopConfirmationMs: 250,
        attemptCount: 2,
        weather: "EXTRASUNNY",
        time: {hour: 12, minute: 0},
        vehicle: {model: "sultan"},
        seed: "fresh:alta:base",
    };
    assert.equal(parseStopSignJobs([job])[0]?.id, job.id);
    assert.equal(parseStopSignJobs([job])[0]?.targetSpeedMps, 15);
    assert.throws(() => parseStopSignJobs([{...job, targetSpeedMps: 15.01}]), /targetSpeedMps/);
    assert.throws(
        () => parseStopSignJobs([{...job, signPose: {x: 0, y: 0, z: 0, heading: 0}}]),
        /uncalibrated origin placeholder/,
    );
    assert.throws(
        () => parseStopSignJobs([{...job, egoStopPose: {...egoStopPose, y: egoStopPose.y + 1}}]),
        /contradicts the sign-relative distance contract/,
    );
    assert.throws(
        () => parseStopSignJobs([{...job, exitPose: {...exitPose, x: exitPose.x + 1}}]),
        /contradicts the sign-relative distance contract/,
    );
}

function testCapturedSceneJobsPreserveExactStartStopEndGeometry() {
    const parsed = capturedSceneJobFixture();
    assert.equal(parsed.catalogId, "gta-v-sign-0023");
    assert.deepEqual(parsed.startPose, {x: 100, y: 180, z: 8, heading: 0});
    assert.deepEqual(parsed.egoStopPose, {x: 100, y: 200, z: 8, heading: 0});
    assert.deepEqual(parsed.exitPose, {x: 100, y: 212, z: 8, heading: 0});
    assert.equal(parsed.stopConfirmationMs, 250);
}

function testLogicalClipPlanLabelsAnchorsWithoutSplittingPhysicalCapture() {
    assert.deepEqual(STOP_SIGN_LOGICAL_CLIP_STAGES, ["approach", "brake_stop", "release"]);
    assert.equal(logicalClipStageForPhase("accelerate"), "approach");
    assert.equal(logicalClipStageForPhase("cruise_approach"), "approach");
    assert.equal(logicalClipStageForPhase("decelerate"), "brake_stop");
    assert.equal(logicalClipStageForPhase("stop_hold"), "brake_stop");
    assert.equal(logicalClipStageForPhase("release"), "release");
}

function testRouteWaypointVariesReproduciblyBeyondEnd() {
    const exitPose = {x: 100, y: 200, z: 8, heading: 90};
    const first = deriveStopSignRouteWaypoint(exitPose, "scene:auto-001:attempt:1");
    const repeated = deriveStopSignRouteWaypoint(exitPose, "scene:auto-001:attempt:1");
    const different = deriveStopSignRouteWaypoint(exitPose, "scene:auto-002:attempt:1");
    assert.deepEqual(repeated, first);
    assert.notDeepEqual(different, first);
    assert(first.forwardM >= STOP_SIGN_ROUTE_WAYPOINT_MIN_FORWARD_M);
    assert(first.forwardM <= STOP_SIGN_ROUTE_WAYPOINT_MAX_FORWARD_M);
    assert(Math.abs(first.lateralM) <= STOP_SIGN_ROUTE_WAYPOINT_MAX_LATERAL_M);
    const relative = relativeStopLinePose(first.pose, exitPose);
    assert(Math.abs(relative.longitudinalM - first.forwardM) < 1e-9);
    assert(Math.abs(relative.lateralM - first.lateralM) < 1e-9);
}

function capturedSceneJobFixture() {
    const egoStopPose = {x: 100, y: 200, z: 8, heading: 0};
    const stopLinePose = {x: 100, y: 202.5, z: 8, heading: 0};
    const signPose = {x: 100, y: 206.5, z: 8, heading: 0};
    const startPose = {x: 100, y: 180, z: 8, heading: 0};
    const exitPose = {x: 100, y: 212, z: 8, heading: 0};
    return parseStopSignJobs([{
        id: "catalog-23:auto-001",
        entryId: "catalog-23",
        variationId: "auto-001",
        catalogId: "gta-v-sign-0023",
        catalogPosition: {x: 100, y: 206.5, z: 8},
        signPose,
        stopLinePose,
        egoStopPose,
        startPose,
        exitPose,
        stopDistanceM: 4,
        egoCenterOffsetM: 2.5,
        startDistanceM: 20,
        exitDistanceM: 12,
        targetSpeedMps: 5,
        stopConfirmationMs: 250,
        attemptCount: 1,
        weather: "EXTRASUNNY",
        time: {hour: 12, minute: 0},
        vehicle: {model: "sultan"},
        seed: "fresh:catalog-23:auto-001",
    }])[0]!;
}

testGeometryUsesGtaHeadingConvention();
testPhaseClassificationIsTemporalButNotSequenceDependent();
testApproachTransitionsFromThrottleToBrake();
testStopAndGoLabelsAreExplicit();
testEarlyStopScoringWaitsForActualDeparture();
testFixedIntervalSchedulerDoesNotAccumulateWorkTime();
testAttemptPreflightRejectsUnsafeStarts();
testExpandedJobsKeepBackendAndFiveMGeometryCoherent();
testCapturedSceneJobsPreserveExactStartStopEndGeometry();
testLogicalClipPlanLabelsAnchorsWithoutSplittingPhysicalCapture();
testRouteWaypointVariesReproduciblyBeyondEnd();
console.log("stop-sign expert tests passed");
