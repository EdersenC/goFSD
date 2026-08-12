import assert from "node:assert/strict";
import {
    parseStopSignLocationProbeRequest,
    placeVehicleAtStopSignLocation,
    StopSignLocationProbeOperations,
} from "./location-probe";

function operations(overrides: Partial<StopSignLocationProbeOperations> = {}): StopSignLocationProbeOperations {
    let coords: [number, number, number] = [0, 0, 0];
    let heading = 0;
    let frozen = false;
    let settled = false;
    return {
        closestVehicleNode: () => [true, [102, 201, 8], 90],
        entityCoords: () => coords,
        entityHeading: () => heading,
        entityPitch: () => 1,
        entityRoll: () => -2,
        entitySpeed: () => 0,
        entityOnGround: () => true,
        freezeEntity: (_entity, value) => { frozen = value; },
        setEntityCoords: (_entity, value) => { coords = value; },
        setEntityHeading: (_entity, value) => { heading = value; },
        setVehicleForwardSpeed: () => undefined,
        setVehicleOnGround: () => true,
        setFocus: () => undefined,
        clearFocus: () => undefined,
        startSceneLoad: () => undefined,
        stopSceneLoad: () => undefined,
        requestPaths: () => undefined,
        requestCollision: () => undefined,
        collisionLoaded: () => true,
        wait: async () => {
            if (!frozen && !settled) {
                coords = [coords[0], coords[1], coords[2] - 1];
                settled = true;
            }
        },
        ...overrides,
    };
}

async function main() {
    const parsed = parseStopSignLocationProbeRequest({
        catalogId: "gta-v-sign-0022",
        catalogPosition: {x: 100, y: 200, z: 8},
        headingOffsetDeg: 180,
    });
    assert.equal(parsed.headingOffsetDeg, 180);

    const result = await placeVehicleAtStopSignLocation(12, parsed, operations());
    assert.equal(result.catalogId, "gta-v-sign-0022");
    assert.equal(result.roadNodePose.heading, 270);
    assert.equal(result.settledPose.z, 8);
    assert(result.distanceFromCatalogM < 3);

    let nodeAttempt = 0;
    let sceneLoadStarted = false;
    let sceneLoadStopped = false;
    const streamed = await placeVehicleAtStopSignLocation(12, parsed, operations({
        closestVehicleNode: () => {
            nodeAttempt += 1;
            return nodeAttempt < 3
                ? [true, [1800, 200, 8], 0]
                : [true, [102, 201, 8], 90];
        },
        startSceneLoad: () => { sceneLoadStarted = true; },
        stopSceneLoad: () => { sceneLoadStopped = true; },
    }));
    assert.equal(streamed.distanceFromCatalogM < 3, true);
    assert.equal(nodeAttempt, 3);
    assert.equal(sceneLoadStarted, true);
    assert.equal(sceneLoadStopped, true);

    await assert.rejects(
        () => placeVehicleAtStopSignLocation(12, {...parsed, headingOffsetDeg: 45}, operations()),
        /either 0 or 180/,
    );
    await assert.rejects(
        () => placeVehicleAtStopSignLocation(12, parsed, operations({entityRoll: () => 8})),
        /not level/,
    );
    await assert.rejects(
        () => placeVehicleAtStopSignLocation(12, parsed, operations({closestVehicleNode: () => [true, [200, 300, 8], 90]})),
        /within 30m.*nearest GTA result/,
    );

    console.log("stop-sign location probe tests passed");
}

void main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
