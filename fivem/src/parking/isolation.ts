import {isValidEntity} from "../helper";
import {ParkingVector3} from "./types";

export const PARKING_ISOLATION_RADIUS_M = 1000;
export const PARKING_ISOLATION_SWEEP_INTERVAL_MS = 1000;
export const PARKING_ISOLATION_ZONE_REFRESH_DISTANCE_M = 100;

export type ParkingIsolationSweepResult = {
    removedVehicles: number
    removedPeds: number
};

export type ParkingIsolationBounds = {
    min: ParkingVector3
    max: ParkingVector3
};

export function parkingIsolationBounds(center: ParkingVector3): ParkingIsolationBounds {
    const radius = PARKING_ISOLATION_RADIUS_M;
    return {
        min: [center[0] - radius, center[1] - radius, center[2] - radius],
        max: [center[0] + radius, center[1] + radius, center[2] + radius],
    };
}

/**
 * Owns the persistent scenario blocker used by parking isolation.
 * The blocker is rebuilt only after meaningful player movement so its 1000m box follows the workspace.
 */
export class PersistentParkingIsolationZone {
    private scenarioBlockingAreaId: number | null = null;
    private center: ParkingVector3 | null = null;

    sync(center: ParkingVector3): boolean {
        if (!shouldRefreshParkingIsolationZone(this.center, center)) {
            return false;
        }

        this.clear();
        const bounds = parkingIsolationBounds(center);
        const scenarioBlockingAreaId = AddScenarioBlockingArea(
            bounds.min[0],
            bounds.min[1],
            bounds.min[2],
            bounds.max[0],
            bounds.max[1],
            bounds.max[2],
            false,
            true,
            true,
            true
        );
        if (!Number.isFinite(scenarioBlockingAreaId) || scenarioBlockingAreaId < 0) {
            console.warn("[parking] native scenario-blocking area could not be created; retrying");
            return false;
        }

        this.scenarioBlockingAreaId = scenarioBlockingAreaId;
        this.center = [...center] as ParkingVector3;
        return true;
    }

    clear() {
        if (this.scenarioBlockingAreaId !== null) {
            RemoveScenarioBlockingArea(this.scenarioBlockingAreaId, false);
        }
        this.scenarioBlockingAreaId = null;
        this.center = null;
    }
}

export function shouldRefreshParkingIsolationZone(
    previousCenter: ParkingVector3 | null,
    currentCenter: ParkingVector3
): boolean {
    if (!previousCenter) {
        return true;
    }
    const dx = currentCenter[0] - previousCenter[0];
    const dy = currentCenter[1] - previousCenter[1];
    const dz = currentCenter[2] - previousCenter[2];
    const refreshDistance = PARKING_ISOLATION_ZONE_REFRESH_DISTANCE_M;
    return ((dx * dx) + (dy * dy) + (dz * dz)) >= (refreshDistance * refreshDistance);
}

/** Suppresses ambient spawning. GTA requires these multipliers to be applied every frame. */
export function enforceParkingIsolationThisFrame(center: ParkingVector3) {
    SetPedPopulationBudget(0);
    SetVehiclePopulationBudget(0);
    SetAllLowPriorityVehicleGeneratorsActive(false);
    SetNumberOfParkedVehicles(0);
    SetGarbageTrucks(false);
    SetRandomBoats(false);
    SetRandomBoatsInMp(false);
    SetRandomTrains(false);
    SetCreateRandomCops(false);
    SetCreateRandomCopsNotOnScenarios(false);
    SetCreateRandomCopsOnScenarios(false);
    SetPedDensityMultiplierThisFrame(0);
    SetScenarioPedDensityMultiplierThisFrame(0, 0);
    SetVehicleDensityMultiplierThisFrame(0);
    SetRandomVehicleDensityMultiplierThisFrame(0);
    SetParkedVehicleDensityMultiplierThisFrame(0);

    const bounds = parkingIsolationBounds(center);
    SetPedNonCreationArea(
        bounds.min[0],
        bounds.min[1],
        bounds.min[2],
        bounds.max[0],
        bounds.max[1],
        bounds.max[2]
    );
}

/** Releases the persistent native non-creation box when parking isolation ends. */
export function clearParkingIsolationNativeZone() {
    ClearPedNonCreationArea();
}

/** Removes local traffic actors while preserving the player and project-owned ego vehicles. */
export function sweepParkingIsolation(
    center: ParkingVector3,
    ownedVehicleIds: ReadonlySet<number>
): ParkingIsolationSweepResult {
    const playerPed = PlayerPedId();
    const protectedVehicles = collectProtectedVehicles(playerPed, ownedVehicleIds);
    removeVehiclesFromGenerators(center);

    let removedVehicles = 0;
    for (const vehicle of gamePool("CVehicle")) {
        if (
            !isValidEntity(vehicle)
            || protectedVehicles.has(vehicle)
            || !isInsideIsolationRadius(vehicle, center)
        ) {
            continue;
        }
        NetworkRequestControlOfEntity(vehicle);
        SetEntityAsMissionEntity(vehicle, true, true);
        DeleteVehicle(vehicle);
        if (!isValidEntity(vehicle)) {
            removedVehicles += 1;
        }
    }

    let removedPeds = 0;
    for (const ped of gamePool("CPed")) {
        if (
            ped === playerPed
            || IsPedAPlayer(ped)
            || !isValidEntity(ped)
            || !isInsideIsolationRadius(ped, center)
        ) {
            continue;
        }
        NetworkRequestControlOfEntity(ped);
        SetEntityAsMissionEntity(ped, true, true);
        DeletePed(ped);
        if (!isValidEntity(ped)) {
            removedPeds += 1;
        }
    }
    return {removedVehicles, removedPeds};
}

function collectProtectedVehicles(
    playerPed: number,
    ownedVehicleIds: ReadonlySet<number>
): Set<number> {
    const protectedVehicles = new Set<number>();
    for (const vehicle of ownedVehicleIds) {
        if (isValidEntity(vehicle)) {
            protectedVehicles.add(vehicle);
        }
    }
    if (IsPedInAnyVehicle(playerPed, false)) {
        const currentVehicle = GetVehiclePedIsIn(playerPed, false);
        if (isValidEntity(currentVehicle)) {
            protectedVehicles.add(currentVehicle);
        }
    }
    for (const ped of gamePool("CPed")) {
        if (!isValidEntity(ped) || !IsPedAPlayer(ped) || !IsPedInAnyVehicle(ped, false)) {
            continue;
        }
        const playerVehicle = GetVehiclePedIsIn(ped, false);
        if (isValidEntity(playerVehicle)) {
            protectedVehicles.add(playerVehicle);
        }
    }
    for (const vehicle of gamePool("CVehicle")) {
        if (isValidEntity(vehicle) && normalizedPlate(vehicle) === "EGO") {
            protectedVehicles.add(vehicle);
        }
    }
    return protectedVehicles;
}

function removeVehiclesFromGenerators(center: ParkingVector3) {
    const bounds = parkingIsolationBounds(center);
    SetAllVehicleGeneratorsActiveInArea(
        bounds.min[0],
        bounds.min[1],
        bounds.min[2],
        bounds.max[0],
        bounds.max[1],
        bounds.max[2],
        false,
        false
    );
    RemoveVehiclesFromGeneratorsInArea(
        bounds.min[0],
        bounds.min[1],
        bounds.min[2],
        bounds.max[0],
        bounds.max[1],
        bounds.max[2],
        0
    );
}

function isInsideIsolationRadius(entity: number, center: ParkingVector3): boolean {
    const coords = GetEntityCoords(entity, false) as [number, number, number];
    const dx = coords[0] - center[0];
    const dy = coords[1] - center[1];
    const dz = coords[2] - center[2];
    return ((dx * dx) + (dy * dy) + (dz * dz))
        <= (PARKING_ISOLATION_RADIUS_M * PARKING_ISOLATION_RADIUS_M);
}

function normalizedPlate(vehicle: number): string {
    return String(GetVehicleNumberPlateText(vehicle) ?? "").trim().toUpperCase();
}

function gamePool(poolName: "CVehicle" | "CPed"): number[] {
    const pool = GetGamePool(poolName) as number[] | undefined;
    return Array.isArray(pool) ? pool : [];
}
