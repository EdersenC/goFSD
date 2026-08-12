import {
    captureScenePose,
    createStopSignPlan,
    currentSceneStep,
    isStopSignSceneCalibrated,
    migrateImplicitCatalogDrafts,
    parseStoredCollectionEntryIds,
    parseStoredStopSignPlan,
    planForEntries,
    planForScene,
    readyStopSignEntryIds,
    reconcileCollectionEntryIds,
    requiredRollingStartDistanceM,
    stageStopSignCatalogLocation,
    stopSignPlanStats,
    validateStopSignPlan,
} from "./stop-sign-plan";
import {
    deriveRouteWaypoint,
    ROUTE_WAYPOINT_MAX_FORWARD_M,
    ROUTE_WAYPOINT_MAX_LATERAL_M,
    ROUTE_WAYPOINT_MIN_FORWARD_M,
} from "./stop-sign/route-waypoint";

const location = {id: "gta-v-sign-0023", x: -1830.7697, y: 3206.3914, z: 31.846758};
const staged = stageStopSignCatalogLocation(createStopSignPlan(), location);
assertEqual(staged.entryIndex, 0);
assertEqual(staged.plan.entries[0]?.catalogId, location.id);
assertEqual(staged.plan.entries[0]?.autoVariations?.count, 50);
assertEqual(currentSceneStep(staged.plan.entries[0]), 0);

let plan = captureScenePose(staged.plan, 0, "startPose", {x: 0, y: -40, z: 30, heading: 0});
assertEqual(currentSceneStep(plan.entries[0]), 1);
plan = captureScenePose(plan, 0, "egoStopPose", {x: 0, y: 0, z: 30, heading: 0});
assertEqual(currentSceneStep(plan.entries[0]), 2);
plan = captureScenePose(plan, 0, "exitPose", {x: 0, y: 12, z: 30, heading: 0});
assertEqual(currentSceneStep(plan.entries[0]), 3);
assert(isStopSignSceneCalibrated(plan.entries[0]!));
assertEqual(validateStopSignPlan(plan).length, 0);
assertEqual(JSON.stringify(stopSignPlanStats(plan)), JSON.stringify({signCount: 1, variationCount: 50, jobCount: 50, attemptCount: 50}));

const queued = planForScene(plan, 0);
assertEqual(queued.entries.length, 1);
assertEqual(queued.entries[0]?.startPose?.y, -40);
assertEqual(queued.entries[0]?.egoStopPose?.y, 0);
assertEqual(queued.entries[0]?.exitPose?.y, 12);
assertEqual(parseStoredStopSignPlan(JSON.stringify(plan))?.entries[0]?.catalogId, location.id);

const secondLocation = {id: "gta-v-sign-0042", x: -1800, y: 3220, z: 31};
const secondStaged = stageStopSignCatalogLocation(plan, secondLocation);
let twoScenePlan = captureScenePose(secondStaged.plan, secondStaged.entryIndex, "startPose", {x: 20, y: -50, z: 30, heading: 0});
twoScenePlan = captureScenePose(twoScenePlan, secondStaged.entryIndex, "egoStopPose", {x: 20, y: 0, z: 30, heading: 0});
twoScenePlan = captureScenePose(twoScenePlan, secondStaged.entryIndex, "exitPose", {x: 20, y: 12, z: 30, heading: 0});
const orderedQueue = planForEntries(twoScenePlan, [twoScenePlan.entries[1]!.id, twoScenePlan.entries[0]!.id]);
assertEqual(orderedQueue.entries.length, 2);
assertEqual(orderedQueue.entries[0]?.catalogId, secondLocation.id);
assertEqual(orderedQueue.entries[1]?.catalogId, location.id);
assertEqual(stopSignPlanStats(orderedQueue).jobCount, 100);
assertEqual(JSON.stringify(readyStopSignEntryIds(twoScenePlan)), JSON.stringify([location.id, secondLocation.id]));
assertEqual(JSON.stringify(reconcileCollectionEntryIds(twoScenePlan, [secondLocation.id, "missing", secondLocation.id, location.id])), JSON.stringify([secondLocation.id, location.id]));
assertEqual(JSON.stringify(parseStoredCollectionEntryIds(JSON.stringify([secondLocation.id, location.id]))), JSON.stringify([secondLocation.id, location.id]));
assertEqual(parseStoredCollectionEntryIds(JSON.stringify([secondLocation.id, 3])), null);
assertThrows(() => planForEntries(twoScenePlan, []), "at least one");
assertThrows(() => planForEntries(twoScenePlan, [location.id, location.id]), "duplicate");
assert(requiredRollingStartDistanceM(15) > 125 && requiredRollingStartDistanceM(15) < 130);
const highSpeedShort = {...plan, entries: [{...plan.entries[0]!, targetSpeedMps: 15}]};
assert(validateStopSignPlan(highSpeedShort).some((error) => error.includes("record stable cruise")));
const highSpeedLong = captureScenePose(highSpeedShort, 0, "startPose", {x: 0, y: -140, z: 30, heading: 0});
assertEqual(validateStopSignPlan(highSpeedLong).length, 0);

const reopened = stageStopSignCatalogLocation(plan, {...location, x: location.x + .1});
assertEqual(reopened.plan.entries.length, 1);
assertEqual(reopened.entryIndex, 0);
assertEqual(reopened.plan.entries[0]?.startPose?.y, -40);
assertEqual(reopened.plan.entries[0]?.egoStopPose?.y, 0);
assertEqual(reopened.plan.entries[0]?.exitPose?.y, 12);
assertEqual(reopened.plan.entries[0]?.catalogPosition?.x, location.x + .1);

const implicitDraft = stageStopSignCatalogLocation(plan, {
    id: "gta-v-sign-0024",
    x: -1820,
    y: 3210,
    z: 31,
}).plan;
const migrated = migrateImplicitCatalogDrafts(implicitDraft);
assertEqual(migrated.entries.length, 1);
assertEqual(migrated.entries[0]?.catalogId, location.id);

const backward = captureScenePose(plan, 0, "exitPose", {x: 0, y: -10, z: 30, heading: 0});
assert(validateStopSignPlan(backward).some((error) => error.includes("End must be")));
assertEqual(parseStoredStopSignPlan(JSON.stringify({...plan, version: "stop-sign-plan.v2"})), null);

const waypoint = deriveRouteWaypoint(plan.entries[0]!.exitPose!, `${plan.seed}:${location.id}:saved-scene-waypoint`);
const repeatedWaypoint = deriveRouteWaypoint(plan.entries[0]!.exitPose!, `${plan.seed}:${location.id}:saved-scene-waypoint`);
assertEqual(JSON.stringify(waypoint), JSON.stringify(repeatedWaypoint));
assert(waypoint.forwardM >= ROUTE_WAYPOINT_MIN_FORWARD_M && waypoint.forwardM <= ROUTE_WAYPOINT_MAX_FORWARD_M);
assert(Math.abs(waypoint.lateralM) <= ROUTE_WAYPOINT_MAX_LATERAL_M);

console.log("stop-sign scene plan tests passed");

function assert(condition: unknown, message = "assertion failed"): asserts condition {
    if (!condition) throw new Error(message);
}

function assertEqual(actual: unknown, expected: unknown, message = `expected ${String(expected)}, got ${String(actual)}`) {
    assert(Object.is(actual, expected), message);
}

function assertThrows(work: () => void, expectedMessage: string) {
    try {
        work();
    } catch (error) {
        assert(error instanceof Error && error.message.includes(expectedMessage), `expected error containing ${expectedMessage}`);
        return;
    }
    throw new Error(`expected error containing ${expectedMessage}`);
}
