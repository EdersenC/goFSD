import assert from "node:assert/strict";
import {
    teleportEntityToWaypoint,
    type TeleportPosition,
    type WaypointTeleportOperations,
} from "./waypoint-teleport";

type OperationLog = {
    freezes: boolean[]
    positions: TeleportPosition[]
    headings: number[]
    focusCount: number
    clearFocusCount: number
    vehicleGroundCount: number
};

function createOperations(overrides: Partial<WaypointTeleportOperations> = {}) {
    const log: OperationLog = {
        freezes: [],
        positions: [],
        headings: [],
        focusCount: 0,
        clearFocusCount: 0,
        vehicleGroundCount: 0,
    };
    const operations: WaypointTeleportOperations = {
        getEntityCoords: () => [10, 20, 30],
        getEntityHeading: () => 135,
        freezeEntity: (_entity, frozen) => log.freezes.push(frozen),
        setEntityCoords: (_entity, position) => log.positions.push(position),
        setEntityHeading: (_entity, heading) => log.headings.push(heading),
        setFocus: () => { log.focusCount += 1; },
        clearFocus: () => { log.clearFocusCount += 1; },
        requestCollision: () => undefined,
        getGroundZ: ([, , z]) => z === 800 ? [true, 42] : [false, 0],
        getClosestVehicleNode: () => [false, [0, 0, 0], 0],
        hasCollisionLoadedAroundEntity: () => true,
        setVehicleOnGround: () => {
            log.vehicleGroundCount += 1;
            return true;
        },
        wait: async () => undefined,
        ...overrides,
    };
    return {operations, log};
}

async function testGroundedVehicleTeleport() {
    const {operations, log} = createOperations();
    const result = await teleportEntityToWaypoint(55, true, [100, 200, 0], operations);

    assert.equal(result.success, true);
    assert.equal(result.usedVehicleNodeFallback, false);
    assert.equal(result.collisionLoaded, true);
    assert.deepEqual(log.positions[log.positions.length - 1], [100, 200, 43]);
    assert.deepEqual(log.headings, [135]);
    assert.deepEqual(log.freezes, [true, false]);
    assert.equal(log.focusCount, 1);
    assert.equal(log.clearFocusCount, 1);
    assert.equal(log.vehicleGroundCount, 2);
}

async function testVehicleNodeFallback() {
    const {operations, log} = createOperations({
        getGroundZ: () => [false, 0],
        getClosestVehicleNode: () => [true, [110, 205, 7], 270],
    });
    const result = await teleportEntityToWaypoint(55, false, [100, 200, 0], operations);

    assert.equal(result.success, true);
    assert.equal(result.usedVehicleNodeFallback, true);
    assert.deepEqual(log.positions[log.positions.length - 1], [110, 205, 8]);
    assert.equal(log.vehicleGroundCount, 0);
}

async function testFailureRestoresOriginalPosition() {
    const {operations, log} = createOperations({
        getGroundZ: () => [false, 0],
        getClosestVehicleNode: () => [false, [0, 0, 0], 0],
    });
    const result = await teleportEntityToWaypoint(55, true, [100, 200, 0], operations);

    assert.equal(result.success, false);
    assert.match(result.reason ?? "", /safe ground/i);
    assert.deepEqual(log.positions[log.positions.length - 1], [10, 20, 30]);
    assert.deepEqual(log.headings, [135]);
    assert.deepEqual(log.freezes, [true, false]);
    assert.equal(log.clearFocusCount, 1);
    assert.equal(log.positions.some(([, , z]) => z === 100), false);
}

async function run() {
    await testGroundedVehicleTeleport();
    await testVehicleNodeFallback();
    await testFailureRestoresOriginalPosition();
    console.log("waypoint teleport tests passed");
}

void run();
