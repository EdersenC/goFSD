import type {Pose} from "../types";

export const ROUTE_WAYPOINT_MIN_FORWARD_M = 120;
export const ROUTE_WAYPOINT_MAX_FORWARD_M = 300;
export const ROUTE_WAYPOINT_MAX_LATERAL_M = 75;

export type RouteWaypointVariation = {
    pose: Pose
    forwardM: number
    lateralM: number
};

export function deriveRouteWaypoint(exitPose: Pose, seed: string): RouteWaypointVariation {
    if (!seed.trim()) {
        throw new Error("route waypoint seed is required");
    }
    const rng = new SeededWaypointRandom(seed);
    const forwardM = interpolate(ROUTE_WAYPOINT_MIN_FORWARD_M, ROUTE_WAYPOINT_MAX_FORWARD_M, rng.next());
    const lateralM = interpolate(-ROUTE_WAYPOINT_MAX_LATERAL_M, ROUTE_WAYPOINT_MAX_LATERAL_M, rng.next());
    const radians = exitPose.heading * Math.PI / 180;
    const forward = [-Math.sin(radians), Math.cos(radians)];
    const right = [Math.cos(radians), Math.sin(radians)];
    return {
        pose: {
            x: exitPose.x + forward[0]! * forwardM + right[0]! * lateralM,
            y: exitPose.y + forward[1]! * forwardM + right[1]! * lateralM,
            z: exitPose.z,
            heading: exitPose.heading,
        },
        forwardM,
        lateralM,
    };
}

class SeededWaypointRandom {
    private state: number;

    constructor(seed: string) {
        this.state = hashSeed(seed) || 0x6d2b79f5;
    }

    next(): number {
        this.state += 0x6d2b79f5;
        let value = this.state;
        value = Math.imul(value ^ (value >>> 15), value | 1);
        value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
        return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    }
}

function hashSeed(value: string): number {
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
