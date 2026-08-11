import {
    captureScenePose,
    createStopSignPlan,
    currentSceneStep,
    migrateImplicitCatalogDrafts,
    parseStoredStopSignPlan,
    planForScene,
    stageStopSignCatalogLocation,
    stopSignPlanStats,
    validateStopSignPlan,
} from "./stop-sign-plan";

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
assertEqual(validateStopSignPlan(plan).length, 0);
assertEqual(JSON.stringify(stopSignPlanStats(plan)), JSON.stringify({signCount: 1, variationCount: 50, jobCount: 50, attemptCount: 50}));

const queued = planForScene(plan, 0);
assertEqual(queued.entries.length, 1);
assertEqual(queued.entries[0]?.startPose?.y, -40);
assertEqual(queued.entries[0]?.egoStopPose?.y, 0);
assertEqual(queued.entries[0]?.exitPose?.y, 12);
assertEqual(parseStoredStopSignPlan(JSON.stringify(plan))?.entries[0]?.catalogId, location.id);

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

console.log("stop-sign scene plan tests passed");

function assert(condition: unknown, message = "assertion failed"): asserts condition {
    if (!condition) throw new Error(message);
}

function assertEqual(actual: unknown, expected: unknown, message = `expected ${String(expected)}, got ${String(actual)}`) {
    assert(Object.is(actual, expected), message);
}
