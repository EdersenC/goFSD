import {isValidEntity} from "./helper";

export type WorldPosition = [number, number, number];

export const WORLD_ISOLATION_RADIUS_M = 1000;
export const WORLD_ISOLATION_SWEEP_INTERVAL_MS = 1000;
export const WORLD_ISOLATION_ZONE_REFRESH_DISTANCE_M = 100;

export type WorldIsolationSweepResult = {
    removedVehicles: number
    removedPeds: number
};

export type WorldIsolationBounds = {
    min: WorldPosition
    max: WorldPosition
};

export function worldIsolationBounds(center: WorldPosition): WorldIsolationBounds {
    const radius = WORLD_ISOLATION_RADIUS_M;
    return {
        min: [center[0] - radius, center[1] - radius, center[2] - radius],
        max: [center[0] + radius, center[1] + radius, center[2] + radius],
    };
}

export function shouldRefreshWorldIsolationZone(
    previousCenter: WorldPosition | null,
    currentCenter: WorldPosition,
): boolean {
    if (!previousCenter) {
        return true;
    }
    const dx = currentCenter[0] - previousCenter[0];
    const dy = currentCenter[1] - previousCenter[1];
    const dz = currentCenter[2] - previousCenter[2];
    const refreshDistance = WORLD_ISOLATION_ZONE_REFRESH_DISTANCE_M;
    return ((dx * dx) + (dy * dy) + (dz * dz)) >= (refreshDistance * refreshDistance);
}

/** Maintains the native blocker and repeated actor sweeps required for an empty collection world. */
export class WorldIsolation {
    private tickId: number | null = null;
    private scenarioBlockingAreaId: number | null = null;
    private center: WorldPosition | null = null;
    private nextSweepAtMs = 0;

    public start(): WorldIsolationSweepResult {
        if (this.tickId === null) {
            this.tickId = setTick(() => this.enforce());
        }
        return this.enforce(true);
    }

    public stop() {
        if (this.tickId !== null) {
            clearTick(this.tickId);
            this.tickId = null;
        }
        this.nextSweepAtMs = 0;
        this.clearScenarioBlockingArea();
        ClearPedNonCreationArea();
    }

    private enforce(forceSweep = false): WorldIsolationSweepResult {
        const center = playerPosition();
        if (!center) {
            return {removedVehicles: 0, removedPeds: 0};
        }
        enforceWorldIsolationThisFrame(center);
        this.syncScenarioBlockingArea(center);

        const nowMs = GetGameTimer();
        if (!forceSweep && nowMs < this.nextSweepAtMs) {
            return {removedVehicles: 0, removedPeds: 0};
        }
        this.nextSweepAtMs = nowMs + WORLD_ISOLATION_SWEEP_INTERVAL_MS;
        return sweepWorldIsolation(center);
    }

    private syncScenarioBlockingArea(center: WorldPosition) {
        if (!shouldRefreshWorldIsolationZone(this.center, center)) {
            return;
        }
        this.clearScenarioBlockingArea();
        const bounds = worldIsolationBounds(center);
        const id = AddScenarioBlockingArea(
            bounds.min[0],
            bounds.min[1],
            bounds.min[2],
            bounds.max[0],
            bounds.max[1],
            bounds.max[2],
            false,
            true,
            true,
            true,
        );
        if (!Number.isFinite(id) || id < 0) {
            console.warn("[world-isolation] native scenario-blocking area could not be created; retrying");
            return;
        }
        this.scenarioBlockingAreaId = id;
        this.center = [...center] as WorldPosition;
    }

    private clearScenarioBlockingArea() {
        if (this.scenarioBlockingAreaId !== null) {
            RemoveScenarioBlockingArea(this.scenarioBlockingAreaId, false);
        }
        this.scenarioBlockingAreaId = null;
        this.center = null;
    }
}

/** GTA population multipliers are frame-scoped and must be reapplied continuously. */
export function enforceWorldIsolationThisFrame(center: WorldPosition) {
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

    const bounds = worldIsolationBounds(center);
    SetPedNonCreationArea(
        bounds.min[0],
        bounds.min[1],
        bounds.min[2],
        bounds.max[0],
        bounds.max[1],
        bounds.max[2],
    );
}

/** Removes ambient actors while protecting players and project-owned EGO vehicles. */
export function sweepWorldIsolation(center: WorldPosition): WorldIsolationSweepResult {
    const playerPed = PlayerPedId();
    const protectedVehicles = collectProtectedVehicles(playerPed);
    removeVehiclesFromGenerators(center);

    let removedVehicles = 0;
    for (const vehicle of gamePool("CVehicle")) {
        if (!isValidEntity(vehicle) || protectedVehicles.has(vehicle) || !isInsideRadius(vehicle, center)) {
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
        if (ped === playerPed || IsPedAPlayer(ped) || !isValidEntity(ped) || !isInsideRadius(ped, center)) {
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

function collectProtectedVehicles(playerPed: number): Set<number> {
    const protectedVehicles = new Set<number>();
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

function removeVehiclesFromGenerators(center: WorldPosition) {
    const bounds = worldIsolationBounds(center);
    SetAllVehicleGeneratorsActiveInArea(
        bounds.min[0], bounds.min[1], bounds.min[2],
        bounds.max[0], bounds.max[1], bounds.max[2],
        false, false,
    );
    RemoveVehiclesFromGeneratorsInArea(
        bounds.min[0], bounds.min[1], bounds.min[2],
        bounds.max[0], bounds.max[1], bounds.max[2],
        0,
    );
}

function isInsideRadius(entity: number, center: WorldPosition): boolean {
    const coords = GetEntityCoords(entity, false) as WorldPosition;
    const dx = coords[0] - center[0];
    const dy = coords[1] - center[1];
    const dz = coords[2] - center[2];
    return ((dx * dx) + (dy * dy) + (dz * dz))
        <= (WORLD_ISOLATION_RADIUS_M * WORLD_ISOLATION_RADIUS_M);
}

function playerPosition(): WorldPosition | null {
    const player = PlayerPedId();
    if (!isValidEntity(player)) {
        return null;
    }
    const coords = GetEntityCoords(player, false) as WorldPosition;
    return Array.isArray(coords) && coords.length >= 3 && coords.every(Number.isFinite)
        ? [coords[0], coords[1], coords[2]]
        : null;
}

function normalizedPlate(vehicle: number): string {
    return String(GetVehicleNumberPlateText(vehicle) ?? "").trim().toUpperCase();
}

function gamePool(poolName: "CVehicle" | "CPed"): number[] {
    const pool = GetGamePool(poolName) as number[] | undefined;
    return Array.isArray(pool) ? pool : [];
}
