import {
    Pose,
    STOP_SIGN_PLAN_VERSION,
    StopSignPlan,
    StopSignPlanEntry,
    StopSignVariation,
} from "./types";

export const STOP_SIGN_PLAN_STORAGE_KEY = "fsd.stop-sign-plan.v1";
const VALID_WEATHER = new Set([
    "BLIZZARD", "CLEAR", "CLEARING", "CLOUDS", "EXTRASUNNY", "FOGGY", "HALLOWEEN", "NEUTRAL",
    "OVERCAST", "RAIN", "SMOG", "SNOW", "SNOWLIGHT", "THUNDER", "XMAS",
]);

export type StopSignPlanStats = {
    signCount: number
    variationCount: number
    jobCount: number
    attemptCount: number
};

export function createStopSignPlan(): StopSignPlan {
    return {
        version: STOP_SIGN_PLAN_VERSION,
        id: "stop-sign-baseline",
        seed: "stop-sign-001",
        entries: [createStopSignEntry(1)],
    };
}

export function createStopSignEntry(index: number): StopSignPlanEntry {
    return {
        id: `sign-${String(index).padStart(2, "0")}`,
        signPose: {x: 0, y: 0, z: 0, heading: 0},
        stopDistanceM: 4,
        egoCenterOffsetM: 2.5,
        startDistanceM: 45,
        targetSpeedMps: 8,
        dwellMs: 5000,
        attemptCount: 10,
        weather: "CLEAR",
        time: {hour: 12, minute: 0},
        vehicle: {model: "adder", color: {r: 26, g: 86, b: 219}},
        variations: [{id: "base"}],
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
        const variations = effectiveVariations(entry);
        return {
            signCount: stats.signCount + 1,
            variationCount: stats.variationCount + variations.length,
            jobCount: stats.jobCount + variations.length,
            attemptCount: stats.attemptCount + variations.reduce((count, variation) => count + (variation.attemptCount ?? entry.attemptCount), 0),
        };
    }, {signCount: 0, variationCount: 0, jobCount: 0, attemptCount: 0});
}

export function validateStopSignPlan(plan: StopSignPlan): string[] {
    const errors: string[] = [];
    if (plan.version !== STOP_SIGN_PLAN_VERSION) {
        errors.push(`Plan version must be ${STOP_SIGN_PLAN_VERSION}.`);
    }
    if (!plan.id.trim()) {
        errors.push("Plan id is required.");
    }
    if (!plan.seed.trim()) {
        errors.push("Seed is required.");
    }
    if (plan.entries.length === 0) {
        errors.push("Add at least one stop sign.");
    }
    if (plan.entries.length > 100) {
        errors.push("A plan can contain at most 100 stop signs.");
    }
    if (stopSignPlanStats(plan).jobCount > 100) {
        errors.push("A plan can expand to at most 100 jobs.");
    }
    const entryIds = new Set<string>();
    for (const entry of plan.entries) {
        const prefix = entry.id.trim() || "unnamed sign";
        if (!entry.id.trim()) {
            errors.push("Every stop sign needs an id.");
        } else if (entryIds.has(entry.id.trim())) {
            errors.push(`Stop sign id ${entry.id} is duplicated.`);
        }
        entryIds.add(entry.id.trim());
        errors.push(...validateStopSignEntry(entry).map((error) => `${prefix}: ${error}`));
    }
    return errors;
}

export function validateStopSignEntry(entry: StopSignPlanEntry): string[] {
    const errors: string[] = [];
    if (!validPose(entry.signPose)) {
        errors.push("sign pose must contain finite coordinates and a heading from 0 to 359.999 degrees.");
    }
    positive(errors, "stop distance", entry.stopDistanceM, 0.5, 15);
    positive(errors, "ego-center offset", entry.egoCenterOffsetM, 0.5, 8);
    positive(errors, "start distance", entry.startDistanceM, 5, 250);
    positive(errors, "target speed", entry.targetSpeedMps, 0.5, 8);
    integer(errors, "dwell", entry.dwellMs, 500, 30000);
    integer(errors, "attempt count", entry.attemptCount, 1, 50);
    integer(errors, "hour", entry.time.hour, 0, 23);
    integer(errors, "minute", entry.time.minute, 0, 59);
    if (!VALID_WEATHER.has(entry.weather.trim().toUpperCase())) {
        errors.push("weather is not a supported GTA weather id.");
    }
    errors.push(...validateVehicle(entry.vehicle));
    const variationIds = new Set<string>();
    for (const variation of effectiveVariations(entry)) {
        if (!variation.id.trim()) {
            errors.push("every variation needs an id.");
        } else if (variationIds.has(variation.id.trim())) {
            errors.push(`variation ${variation.id} is duplicated.`);
        }
        variationIds.add(variation.id.trim());
        errors.push(...validateVariation(variation).map((error) => `${variation.id || "unnamed variation"}: ${error}`));
    }
    return errors;
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

export function distanceBetweenPoses(a: Pose, b: Pose): number {
    return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

function validateVariation(variation: StopSignVariation): string[] {
    const errors: string[] = [];
    optionalRange(errors, "stop distance", variation.stopDistanceM, 0.5, 15);
    optionalRange(errors, "ego-center offset", variation.egoCenterOffsetM, 0.5, 8);
    optionalRange(errors, "start distance", variation.startDistanceM, 5, 250);
    optionalRange(errors, "target speed", variation.targetSpeedMps, 0.5, 8);
    optionalInteger(errors, "dwell", variation.dwellMs, 500, 30000);
    optionalInteger(errors, "attempt count", variation.attemptCount, 1, 50);
    if (variation.weather !== undefined && !VALID_WEATHER.has(variation.weather.trim().toUpperCase())) {
        errors.push("weather is not a supported GTA weather id.");
    }
    if (variation.time) {
        integer(errors, "hour", variation.time.hour, 0, 23);
        integer(errors, "minute", variation.time.minute, 0, 59);
    }
    if (variation.vehicle) {
        errors.push(...validateVehicle(variation.vehicle));
    }
    return errors;
}

function validateVehicle(vehicle: {model?: string, color?: {r: number, g: number, b: number}}): string[] {
    const errors: string[] = [];
    const model = vehicle.model?.trim() ?? "";
    if (model.length > 64 || !/^[a-z0-9_-]*$/i.test(model)) {
        errors.push("vehicle model must be at most 64 letters, numbers, underscores, or hyphens.");
    }
    if (vehicle.color) {
        integer(errors, "vehicle red", vehicle.color.r, 0, 255);
        integer(errors, "vehicle green", vehicle.color.g, 0, 255);
        integer(errors, "vehicle blue", vehicle.color.b, 0, 255);
    }
    return errors;
}

function effectiveVariations(entry: StopSignPlanEntry): StopSignVariation[] {
    return entry.variations.length > 0 ? entry.variations : [{id: "base"}];
}

function cloneEntry(entry: StopSignPlanEntry): StopSignPlanEntry {
    return {
        ...entry,
        signPose: {...entry.signPose},
        time: {...entry.time},
        vehicle: {
            ...entry.vehicle,
            color: entry.vehicle.color ? {...entry.vehicle.color} : undefined,
        },
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

function validPose(pose: Pose): boolean {
    return [pose.x, pose.y, pose.z, pose.heading].every(Number.isFinite)
        && pose.heading >= 0
        && pose.heading < 360;
}

function positive(errors: string[], label: string, value: number, min: number, max: number) {
    if (!Number.isFinite(value) || value < min || value > max) {
        errors.push(`${label} must be from ${min} to ${max}.`);
    }
}

function integer(errors: string[], label: string, value: number, min: number, max: number) {
    if (!Number.isInteger(value) || value < min || value > max) {
        errors.push(`${label} must be a whole number from ${min} to ${max}.`);
    }
}

function optionalRange(errors: string[], label: string, value: number | undefined, min: number, max: number) {
    if (value !== undefined) {
        positive(errors, label, value, min, max);
    }
}

function optionalInteger(errors: string[], label: string, value: number | undefined, min: number, max: number) {
    if (value !== undefined) {
        integer(errors, label, value, min, max);
    }
}
