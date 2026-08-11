export type SpatialVector3 = [number, number, number];

export function gtaForwardVector(heading: number): SpatialVector3 {
    const radians = heading * Math.PI / 180;
    return [-Math.sin(radians), Math.cos(radians), 0];
}

export function gtaRightVector(heading: number): SpatialVector3 {
    const radians = heading * Math.PI / 180;
    return [Math.cos(radians), Math.sin(radians), 0];
}

export function gtaHeadingFromVector(vector: SpatialVector3): number {
    return normalizeHeading(Math.atan2(-vector[0], vector[1]) * 180 / Math.PI);
}

export function headingDeltaDegrees(targetHeading: number, sourceHeading: number): number {
    return ((targetHeading - sourceHeading + 540) % 360) - 180;
}

export function normalizeHeading(heading: number): number {
    return ((heading % 360) + 360) % 360;
}
