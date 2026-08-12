import {
    Pose,
    STOP_SIGN_PLAN_VERSION,
    StopSignPlan,
    StopSignPlanEntry,
} from "./types";

export const STOP_SIGN_PLAN_STORAGE_KEY = "fsd.stop-sign-scenes.v5";
export const STOP_SIGN_COLLECTION_QUEUE_STORAGE_KEY = "fsd.stop-sign-collection-queue.v1";
export const LEGACY_IMPLICIT_STOP_SIGN_PLAN_STORAGE_KEY = "fsd.stop-sign-scenes.v4";
export const DEFAULT_VARIANT_COUNT = 50;
export const DEFAULT_MOTION_VARIANCE_PCT = 20;
export const MAXIMUM_MOTION_VARIANCE_PCT = 25;
export const MAXIMUM_TARGET_SPEED_MPS = 15;
export const MINIMUM_ROLLING_CRUISE_SECONDS = 1.25;
const STOP_SIGN_LAUNCH_ACCELERATION_MPS2 = 1.8;
const STOP_SIGN_BRAKING_DECELERATION_MPS2 = 2.4;

export type ScenePoseField = "startPose" | "egoStopPose" | "exitPose";
export type CalibratedStopSignPlanEntry = StopSignPlanEntry & {
    catalogId: string
    startPose: Pose
    egoStopPose: Pose
    exitPose: Pose
};

export type StopSignPlanStats = {
    signCount: number
    variationCount: number
    jobCount: number
    attemptCount: number
};

type CatalogLocation = {id: string, x: number, y: number, z: number};

export function createStopSignPlan(): StopSignPlan {
    return {
        version: STOP_SIGN_PLAN_VERSION,
        id: "stop-sign-scenes",
        seed: createStopSignSeed(),
        entries: [],
    };
}

export function createStopSignEntry(index: number, location?: CatalogLocation): StopSignPlanEntry {
    return {
        id: location?.id ?? `scene-${String(index).padStart(2, "0")}`,
        catalogId: location?.id,
        catalogPosition: location ? {x: location.x, y: location.y, z: location.z} : undefined,
        signPose: {x: 0, y: 0, z: 0, heading: 0},
        autoVariations: {count: DEFAULT_VARIANT_COUNT, motionVariancePct: DEFAULT_MOTION_VARIANCE_PCT},
        stopDistanceM: 4,
        egoCenterOffsetM: 2.5,
        startDistanceM: 35,
        exitDistanceM: 8,
        targetSpeedMps: 5,
        stopConfirmationMs: 250,
        attemptCount: 1,
        weather: "EXTRASUNNY",
        time: {hour: 12, minute: 0},
        vehicle: {model: "sultan", color: {r: 26, g: 86, b: 219}},
        variations: [],
    };
}

export function cloneStopSignPlan(plan: StopSignPlan): StopSignPlan {
    return {
        version: STOP_SIGN_PLAN_VERSION,
        id: plan.id,
        seed: plan.seed,
        entries: plan.entries.map(cloneEntry),
    };
}

export function stopSignPlanStats(plan: StopSignPlan): StopSignPlanStats {
    return plan.entries.reduce<StopSignPlanStats>((stats, entry) => {
        const variations = entry.autoVariations?.count ?? Math.max(1, entry.variations.length);
        return {
            signCount: stats.signCount + 1,
            variationCount: stats.variationCount + variations,
            jobCount: stats.jobCount + variations,
            attemptCount: stats.attemptCount + variations * entry.attemptCount,
        };
    }, {signCount: 0, variationCount: 0, jobCount: 0, attemptCount: 0});
}

export function stageStopSignCatalogLocation(plan: StopSignPlan, location: CatalogLocation): {plan: StopSignPlan, entryIndex: number} {
    const next = cloneStopSignPlan(plan);
    const existingIndex = next.entries.findIndex((entry) => entry.catalogId === location.id);
    if (existingIndex >= 0) {
        next.entries[existingIndex] = {
            ...next.entries[existingIndex]!,
            catalogPosition: {x: location.x, y: location.y, z: location.z},
        };
        return {plan: next, entryIndex: existingIndex};
    }
    next.entries.push(createStopSignEntry(next.entries.length + 1, location));
    return {plan: next, entryIndex: next.entries.length - 1};
}

export function captureScenePose(
    plan: StopSignPlan,
    entryIndex: number,
    field: ScenePoseField,
    pose: Pose,
): StopSignPlan {
    if (!validPose(pose)) {
        throw new Error("The setup car did not report a valid world pose");
    }
    const next = cloneStopSignPlan(plan);
    const entry = next.entries[entryIndex];
    if (!entry) {
        throw new Error("Select a stop-sign scene before capturing a point");
    }
    next.entries[entryIndex] = {...entry, [field]: {...pose}};
    return next;
}

export function planForScene(plan: StopSignPlan, entryIndex: number): StopSignPlan {
    const entry = plan.entries[entryIndex];
    if (!entry) {
        throw new Error("Select a saved scene before collecting");
    }
    return planForEntries(plan, [entry.id]);
}

export function planForEntries(plan: StopSignPlan, entryIds: readonly string[]): StopSignPlan {
    if (entryIds.length === 0) {
        throw new Error("Choose at least one saved scene before collecting");
    }
    const entriesById = new Map(plan.entries.map((entry) => [entry.id, entry]));
    const selected = new Set<string>();
    const entries = entryIds.map((entryId) => {
        if (selected.has(entryId)) {
            throw new Error(`Collection queue contains duplicate scene ${entryId}`);
        }
        selected.add(entryId);
        const entry = entriesById.get(entryId);
        if (!entry || !isStopSignSceneCalibrated(entry)) {
            throw new Error(`Collection scene ${entryId} is missing a saved Start, Stop, or End pose`);
        }
        return cloneEntry(entry);
    });
    return {
        version: STOP_SIGN_PLAN_VERSION,
        id: `${plan.id}-collection`.slice(0, 120),
        seed: plan.seed,
        entries,
    };
}

export function readyStopSignEntryIds(plan: StopSignPlan): string[] {
    return plan.entries.filter(isStopSignSceneCalibrated).map((entry) => entry.id);
}

export function reconcileCollectionEntryIds(plan: StopSignPlan, entryIds: readonly string[]): string[] {
    const ready = new Set(readyStopSignEntryIds(plan));
    const seen = new Set<string>();
    return entryIds.filter((entryId) => {
        if (!ready.has(entryId) || seen.has(entryId)) {
            return false;
        }
        seen.add(entryId);
        return true;
    });
}

export function parseStoredCollectionEntryIds(raw: string | null): string[] | null {
    if (!raw) {
        return null;
    }
    try {
        const candidate = JSON.parse(raw) as unknown;
        if (!Array.isArray(candidate) || candidate.some((entryId) => typeof entryId !== "string" || !entryId.trim())) {
            return null;
        }
        return [...candidate];
    } catch {
        return null;
    }
}

export function validateStopSignPlan(plan: StopSignPlan): string[] {
    const errors: string[] = [];
    if (plan.version !== STOP_SIGN_PLAN_VERSION) {
        errors.push(`Scene library version must be ${STOP_SIGN_PLAN_VERSION}.`);
    }
    if (!plan.id.trim()) {
        errors.push("Scene library id is required.");
    }
    if (!plan.seed.trim()) {
        errors.push("Seed is required.");
    }
    if (plan.entries.length === 0) {
        errors.push("Teleport to a stop sign to create the first scene.");
    }
    if (plan.entries.length > 100) {
        errors.push("A scene library can contain at most 100 stop signs.");
    }
    if (stopSignPlanStats(plan).jobCount > 5000) {
        errors.push("The saved scenes can contain at most 5,000 generated variants.");
    }
    const ids = new Set<string>();
    plan.entries.forEach((entry) => {
        const label = entry.catalogId || entry.id || "unnamed scene";
        if (!entry.id.trim() || ids.has(entry.id.trim())) {
            errors.push(`${label}: scene id must be present and unique.`);
        }
        ids.add(entry.id.trim());
        errors.push(...validateStopSignEntry(entry).map((error) => `${label}: ${error}`));
    });
    return errors;
}

export function validateStopSignEntry(entry: StopSignPlanEntry): string[] {
    const errors: string[] = [];
    if (!entry.catalogId?.trim() || !entry.catalogPosition || !validPosition(entry.catalogPosition)) {
        errors.push("choose a physical stop sign from the catalog.");
    }
    for (const [label, pose] of [
        ["Start", entry.startPose],
        ["Stop", entry.egoStopPose],
        ["End", entry.exitPose],
    ] as const) {
        if (!pose || !validPose(pose)) {
            errors.push(`capture the ${label} pose from the setup car.`);
        }
    }
    if (entry.startPose && entry.egoStopPose && entry.exitPose) {
        const start = relativeTo(entry.startPose, entry.egoStopPose);
        const end = relativeTo(entry.exitPose, entry.egoStopPose);
        const minimumRollingStartM = requiredRollingStartDistanceM(entry.targetSpeedMps);
        if (start.longitudinal > -minimumRollingStartM) {
            errors.push(`Start must be at least ${minimumRollingStartM.toFixed(0)} m before Stop to reach ${entry.targetSpeedMps.toFixed(1)} m/s and record stable cruise.`);
        }
        if (end.longitudinal < 8) {
            errors.push("End must be at least 8 m beyond Stop so the release clip has enough temporal context.");
        }
        if (distance(entry.startPose, entry.egoStopPose) > 250) {
            errors.push("Start must be within 250 m of Stop.");
        }
        if (distance(entry.exitPose, entry.egoStopPose) > 50) {
            errors.push("End must be within 50 m of Stop.");
        }
        if (Math.abs(start.lateral) > 8 || Math.abs(end.lateral) > 8) {
            errors.push("Start, Stop, and End must stay within one 8 m lane corridor.");
        }
    }
    const generated = entry.autoVariations;
    if (!generated || !Number.isInteger(generated.count) || generated.count < 1 || generated.count > 100) {
        errors.push("generated variant count must be a whole number from 1 to 100.");
    }
    if (!generated || !Number.isFinite(generated.motionVariancePct) || generated.motionVariancePct < 0 || generated.motionVariancePct > MAXIMUM_MOTION_VARIANCE_PCT) {
        errors.push(`motion variance must be from 0 to ${MAXIMUM_MOTION_VARIANCE_PCT}%.`);
    }
    if (!Number.isFinite(entry.targetSpeedMps) || entry.targetSpeedMps < .5 || entry.targetSpeedMps > MAXIMUM_TARGET_SPEED_MPS) {
        errors.push(`target speed must be from 0.5 to ${MAXIMUM_TARGET_SPEED_MPS} m/s.`);
    }
    if (!Number.isInteger(entry.stopConfirmationMs) || entry.stopConfirmationMs < 100 || entry.stopConfirmationMs > 1000) {
        errors.push("stop confirmation must be a whole number from 100 to 1,000 ms.");
    }
    return errors;
}

export function requiredRollingStartDistanceM(targetSpeedMps: number): number {
    if (!Number.isFinite(targetSpeedMps) || targetSpeedMps <= 0) {
        return Number.POSITIVE_INFINITY;
    }
    const accelerationDistanceM = targetSpeedMps ** 2 / (2 * STOP_SIGN_LAUNCH_ACCELERATION_MPS2);
    const cruiseDistanceM = targetSpeedMps * MINIMUM_ROLLING_CRUISE_SECONDS;
    const brakingDistanceM = targetSpeedMps ** 2 / (2 * STOP_SIGN_BRAKING_DECELERATION_MPS2);
    return accelerationDistanceM + cruiseDistanceM + brakingDistanceM;
}

export function parseStoredStopSignPlan(raw: string | null): StopSignPlan | null {
    if (!raw) {
        return null;
    }
    try {
        const candidate = JSON.parse(raw) as Partial<StopSignPlan>;
        if (candidate.version !== STOP_SIGN_PLAN_VERSION || !Array.isArray(candidate.entries)) {
            return null;
        }
        return cloneStopSignPlan(candidate as StopSignPlan);
    } catch {
        return null;
    }
}

export function migrateImplicitCatalogDrafts(plan: StopSignPlan): StopSignPlan {
    const migrated = cloneStopSignPlan(plan);
    migrated.entries = migrated.entries.filter(hasCapturedScenePose);
    return migrated;
}

export function currentSceneStep(entry?: StopSignPlanEntry): 0 | 1 | 2 | 3 {
    if (!entry?.startPose) return 0;
    if (!entry.egoStopPose) return 1;
    if (!entry.exitPose) return 2;
    return 3;
}

export function isStopSignSceneCalibrated(entry: StopSignPlanEntry): entry is CalibratedStopSignPlanEntry {
    return Boolean(
        entry.catalogId?.trim()
        && entry.startPose && validPose(entry.startPose)
        && entry.egoStopPose && validPose(entry.egoStopPose)
        && entry.exitPose && validPose(entry.exitPose),
    );
}

export function poseSummary(pose?: Pose): string {
    return pose
        ? `${pose.x.toFixed(1)}, ${pose.y.toFixed(1)} · ${pose.heading.toFixed(0)}°`
        : "Not captured";
}

function cloneEntry(entry: StopSignPlanEntry): StopSignPlanEntry {
    return {
        ...entry,
        signPose: {...entry.signPose},
        catalogPosition: entry.catalogPosition ? {...entry.catalogPosition} : undefined,
        startPose: entry.startPose ? {...entry.startPose} : undefined,
        egoStopPose: entry.egoStopPose ? {...entry.egoStopPose} : undefined,
        exitPose: entry.exitPose ? {...entry.exitPose} : undefined,
        autoVariations: entry.autoVariations ? {...entry.autoVariations} : undefined,
        time: {...entry.time},
        vehicle: {...entry.vehicle, color: entry.vehicle.color ? {...entry.vehicle.color} : undefined},
        variations: entry.variations.map((variation) => ({
            ...variation,
            time: variation.time ? {...variation.time} : undefined,
            vehicle: variation.vehicle ? {
                ...variation.vehicle,
                color: variation.vehicle.color ? {...variation.vehicle.color} : undefined,
            } : undefined,
        })),
    };
}

function hasCapturedScenePose(entry: StopSignPlanEntry): boolean {
    return Boolean(entry.startPose || entry.egoStopPose || entry.exitPose);
}

function validPose(pose: Pose): boolean {
    return validPosition(pose) && Number.isFinite(pose.heading) && pose.heading >= 0 && pose.heading < 360;
}

function validPosition(position: {x: number, y: number, z: number}): boolean {
    return [position.x, position.y, position.z].every(Number.isFinite)
        && Math.abs(position.x) <= 10_000
        && Math.abs(position.y) <= 10_000
        && position.z >= -1_000
        && position.z <= 3_000;
}

function relativeTo(pose: Pose, origin: Pose): {longitudinal: number, lateral: number} {
    const radians = origin.heading * Math.PI / 180;
    const forward = [-Math.sin(radians), Math.cos(radians)];
    const right = [Math.cos(radians), Math.sin(radians)];
    const x = pose.x - origin.x;
    const y = pose.y - origin.y;
    return {
        longitudinal: x * forward[0]! + y * forward[1]!,
        lateral: x * right[0]! + y * right[1]!,
    };
}

function distance(first: Pose, second: Pose): number {
    return Math.hypot(first.x - second.x, first.y - second.y, first.z - second.z);
}

export function createStopSignSeed(): string {
    return `scene-seed-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}
