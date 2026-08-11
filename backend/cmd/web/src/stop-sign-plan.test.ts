import {
    createStopSignEntry,
    createStopSignPlan,
    distanceBetweenPoses,
    parseStoredStopSignPlan,
    stopSignPlanStats,
    stageStopSignCatalogLocation,
    validateStopSignPlan,
} from "./stop-sign-plan";

const plan = createStopSignPlan();
assert(/sign pose/i.test(validateStopSignPlan(plan).join(" ")), "a placeholder pose must block collection");
const stagedPlaceholder = stageStopSignCatalogLocation(plan, "gta-v-sign-0001");
assertEqual(stagedPlaceholder.entries[0]?.id, "gta-v-sign-0001");
assertEqual(stagedPlaceholder.entries[0]?.signPose.x, 0, "catalog staging must not copy a roadside prop pose");
assertEqual(plan.entries[0]?.id, "sign-01", "catalog staging must not mutate the current plan");
plan.entries[0]!.signPose = {x: 120, y: -40, z: 30, heading: 90};
plan.entries[0]!.variations.push({id: "rain", weather: "RAIN", targetSpeedMps: 6, attemptCount: 4});

assertEqual(JSON.stringify(stopSignPlanStats(plan)), JSON.stringify({signCount: 1, variationCount: 2, jobCount: 2, attemptCount: 14}));
assertEqual(JSON.stringify(validateStopSignPlan(plan)), "[]");

const stagedAdditional = stageStopSignCatalogLocation(plan, "gta-v-sign-0002");
assertEqual(stagedAdditional.entries.length, 2);
assertEqual(stagedAdditional.entries[1]?.id, "gta-v-sign-0002");
assert(/uncalibrated/i.test(validateStopSignPlan(stagedAdditional).join(" ")));
assertEqual(stageStopSignCatalogLocation(stagedAdditional, "gta-v-sign-0002").entries.length, 2, "restaging must not duplicate a catalog sign");

const parsed = parseStoredStopSignPlan(JSON.stringify(plan));
assert(parsed, "v2 plan should restore");
parsed.entries[0]!.signPose.x = 999;
assertEqual(plan.entries[0]!.signPose.x, 120, "restored plan must not alias input poses");
assertEqual(parseStoredStopSignPlan('{"version":"legacy"}'), null);

const invalid = createStopSignPlan();
invalid.entries = [createStopSignEntry(1), createStopSignEntry(1)];
invalid.entries[0]!.startDistanceM = 1;
invalid.entries[0]!.exitDistanceM = 1;
const errors = validateStopSignPlan(invalid).join(" ");
assert(/duplicated/i.test(errors));
assert(/start distance must be from/i.test(errors));
assert(/exit distance must be from/i.test(errors));
assertEqual(distanceBetweenPoses(
    {x: 0, y: 0, z: 0, heading: 0},
    {x: 3, y: 4, z: 0, heading: 90},
), 5);

console.log("stop-sign plan tests passed");

function assert(condition: unknown, message = "assertion failed"): asserts condition {
    if (!condition) {
        throw new Error(message);
    }
}

function assertEqual(actual: unknown, expected: unknown, message = `expected ${String(expected)}, got ${String(actual)}`) {
    assert(Object.is(actual, expected), message);
}
