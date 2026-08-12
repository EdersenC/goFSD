import {gtaForwardVector, gtaRightVector} from "./geometry";
import {StopSignPose} from "./types";

export const STOP_SIGN_ROUTE_WAYPOINT_MIN_FORWARD_M = 120;
export const STOP_SIGN_ROUTE_WAYPOINT_MAX_FORWARD_M = 300;
export const STOP_SIGN_ROUTE_WAYPOINT_MAX_LATERAL_M = 75;

export type StopSignRouteWaypoint = {
    pose: StopSignPose
    forwardM: number
    lateralM: number
};

/**
 * Places the visual route marker well beyond End without changing the route
 * controller's End pose. The seed keeps every collected attempt reproducible.
 */
export function deriveStopSignRouteWaypoint(exitPose: StopSignPose, seed: string): StopSignRouteWaypoint {
    if (!seed.trim()) {
        throw new Error("stop-sign route waypoint seed is required");
    }
    const rng = new SeededRouteWaypointRandom(seed);
    const forwardM = interpolate(
        STOP_SIGN_ROUTE_WAYPOINT_MIN_FORWARD_M,
        STOP_SIGN_ROUTE_WAYPOINT_MAX_FORWARD_M,
        rng.next(),
    );
    const lateralM = interpolate(
        -STOP_SIGN_ROUTE_WAYPOINT_MAX_LATERAL_M,
        STOP_SIGN_ROUTE_WAYPOINT_MAX_LATERAL_M,
        rng.next(),
    );
    const [forwardX, forwardY] = gtaForwardVector(exitPose.heading);
    const [rightX, rightY] = gtaRightVector(exitPose.heading);
    return {
        pose: {
            x: exitPose.x + forwardX * forwardM + rightX * lateralM,
            y: exitPose.y + forwardY * forwardM + rightY * lateralM,
            z: exitPose.z,
            heading: exitPose.heading,
        },
        forwardM,
        lateralM,
    };
}

class SeededRouteWaypointRandom {
    private state: number;

    constructor(seed: string) {
        this.state = hashRouteWaypointSeed(seed) || 0x6d2b79f5;
    }

    next(): number {
        this.state += 0x6d2b79f5;
        let value = this.state;
        value = Math.imul(value ^ (value >>> 15), value | 1);
        value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
        return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    }
}

function hashRouteWaypointSeed(value: string): number {
    let hash = 2166136261;
    for (let index = 0; index < value.length; index += 1) {
        hash ^= value.charCodeAt(index);
        hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
}

function interpolate(minimum: number, maximum: number, ratio: number): number {
    return minimum + (maximum - minimum) * ratio;
}
