import {poseFromLocalOffset} from "./geometry";
import {
    ParkingAttemptPlan,
    ParkingPose,
    ParkingStartOffset,
} from "./types";

export const MIN_PARKING_ATTEMPTS = 1;
export const MAX_PARKING_ATTEMPTS = 50;
export const DEFAULT_PARKING_EVALUATION_SEED = "parking-forward-bay:evaluation";

export type ParkingEvaluationPlan = {
    seed: string
    plan: ParkingAttemptPlan
};

const canonicalOffsets: readonly ParkingStartOffset[] = [
    {longitudinalM: -9.5, lateralM: 0, headingDeg: 0},
    {longitudinalM: -10.5, lateralM: 0.75, headingDeg: 4},
    {longitudinalM: -10.5, lateralM: -0.75, headingDeg: -4},
    {longitudinalM: -12, lateralM: 1.25, headingDeg: 7},
    {longitudinalM: -12, lateralM: -1.25, headingDeg: -7},
    {longitudinalM: -13.5, lateralM: 1.75, headingDeg: 11},
    {longitudinalM: -13.5, lateralM: -1.75, headingDeg: -11},
    {longitudinalM: -15, lateralM: 2.25, headingDeg: 14},
    {longitudinalM: -15, lateralM: -2.25, headingDeg: -14},
];

class SeededRandom {
    private state: number;

    constructor(seed: string) {
        this.state = hashString(seed) || 0x6d2b79f5;
    }

    next(): number {
        this.state += 0x6d2b79f5;
        let value = this.state;
        value = Math.imul(value ^ (value >>> 15), value | 1);
        value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
        return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    }
}

export function resolveParkingAttemptCount(value: unknown): number {
    const parsed = typeof value === "number" ? value : Number(value);
    if (!Number.isInteger(parsed) || parsed < MIN_PARKING_ATTEMPTS || parsed > MAX_PARKING_ATTEMPTS) {
        throw new Error(
            `attemptCount must be an integer from ${MIN_PARKING_ATTEMPTS} through ${MAX_PARKING_ATTEMPTS}`
        );
    }
    return parsed;
}

export function buildForwardBayCurriculum(
    target: ParkingPose,
    attemptCount: number,
    seed: string
): ParkingAttemptPlan[] {
    const count = resolveParkingAttemptCount(attemptCount);
    const rng = new SeededRandom(seed);
    const orderedOffsets = seededShuffle([...canonicalOffsets], rng);
    const plans: ParkingAttemptPlan[] = [];

    for (let attemptIndex = 0; attemptIndex < count; attemptIndex += 1) {
        const cycle = Math.floor(attemptIndex / orderedOffsets.length);
        const base = orderedOffsets[attemptIndex % orderedOffsets.length];
        if (!base) {
            throw new Error("parking curriculum offset invariant violated");
        }
        const startOffset = jitterOffset(base, rng, cycle);
        plans.push({
            attemptIndex,
            startOffset,
            startPose: poseFromLocalOffset(
                target,
                startOffset.longitudinalM,
                startOffset.lateralM,
                startOffset.headingDeg
            ),
        });
    }
    return plans;
}

export function buildForwardBayEvaluationPlan(
    target: ParkingPose,
    requestedSeed?: string
): ParkingEvaluationPlan {
    const seed = requestedSeed?.trim() || DEFAULT_PARKING_EVALUATION_SEED;
    const plan = buildForwardBayCurriculum(target, 1, seed)[0];
    if (!plan) {
        throw new Error("parking evaluation plan invariant violated");
    }
    return {seed, plan};
}

function jitterOffset(base: ParkingStartOffset, rng: SeededRandom, cycle: number): ParkingStartOffset {
    if (cycle === 0) {
        return {...base};
    }
    const scale = Math.min(cycle, 4);
    return {
        longitudinalM: round(base.longitudinalM - (rng.next() * 0.5 * scale), 3),
        lateralM: round(base.lateralM + ((rng.next() - 0.5) * 0.25 * scale), 3),
        headingDeg: round(base.headingDeg + ((rng.next() - 0.5) * 1.5 * scale), 3),
    };
}

function seededShuffle<T>(items: T[], rng: SeededRandom): T[] {
    for (let index = items.length - 1; index > 0; index -= 1) {
        const swapIndex = Math.floor(rng.next() * (index + 1));
        const current = items[index];
        const replacement = items[swapIndex];
        if (current === undefined || replacement === undefined) {
            throw new Error("parking curriculum shuffle invariant violated");
        }
        items[index] = replacement;
        items[swapIndex] = current;
    }
    return items;
}

function hashString(value: string): number {
    let hash = 2166136261;
    for (let index = 0; index < value.length; index += 1) {
        hash ^= value.charCodeAt(index);
        hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
}

function round(value: number, decimals: number): number {
    const scale = 10 ** decimals;
    return Math.round(value * scale) / scale;
}
