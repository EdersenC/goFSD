import assert from "node:assert/strict";
import {
    shouldRefreshWorldIsolationZone,
    WORLD_ISOLATION_RADIUS_M,
    WORLD_ISOLATION_ZONE_REFRESH_DISTANCE_M,
    worldIsolationBounds,
} from "./world-isolation";

const bounds = worldIsolationBounds([120, -45, 8]);
assert.equal(WORLD_ISOLATION_RADIUS_M, 1000);
assert.deepEqual(bounds.min, [-880, -1045, -992]);
assert.deepEqual(bounds.max, [1120, 955, 1008]);
assert(shouldRefreshWorldIsolationZone(null, [0, 0, 0]));
assert(!shouldRefreshWorldIsolationZone([0, 0, 0], [99, 0, 0]));
assert(shouldRefreshWorldIsolationZone(
    [0, 0, 0],
    [WORLD_ISOLATION_ZONE_REFRESH_DISTANCE_M, 0, 0],
));

console.log("world isolation tests passed");
