import {isValidEntity} from "../helper";
import {poseFromLocalOffset} from "./geometry";
import {ParkingPose, ParkingTarget} from "./types";

export type ParkingFixturePlan = {
    model: "buffalo" | "futo"
    color: 13 | 39
    pose: ParkingPose
};

const fixtureTemplates: readonly Omit<ParkingFixturePlan, "pose">[] = [
    {model: "buffalo", color: 13},
    {model: "futo", color: 39},
];

type ParkingFixtureDependencies = {
    spawn: (plan: ParkingFixturePlan) => Promise<number>
    protect: (vehicle: number) => void
};

export function buildParkingFixturePlans(target: ParkingTarget): ParkingFixturePlan[] {
    const adjacentBayCenterOffsetM = target.bay.widthM;
    return fixtureTemplates.map((template, index) => ({
        ...template,
        pose: poseFromLocalOffset(
            target.pose,
            0,
            index === 0 ? -adjacentBayCenterOffsetM : adjacentBayCenterOffsetM
        ),
    }));
}

/** Owns the deterministic neighboring cars used by the straight-stop scene. */
export class ParkingFixtureSet {
    private vehicleIds: number[] = [];

    constructor(private readonly dependencies: ParkingFixtureDependencies) {}

    async replace(target: ParkingTarget) {
        this.clear();
        try {
            for (const plan of buildParkingFixturePlans(target)) {
                const vehicle = await this.dependencies.spawn(plan);
                if (!isValidEntity(vehicle)) {
                    throw new Error(`Failed to spawn parking fixture model=${plan.model}`);
                }
                this.vehicleIds.push(vehicle);
                configureParkedFixture(vehicle);
                this.dependencies.protect(vehicle);
            }
        } catch (error) {
            this.clear();
            throw error;
        }
    }

    clear() {
        for (const vehicle of this.vehicleIds) {
            deleteOwnedVehicle(vehicle);
        }
        this.vehicleIds = [];
    }
}

function configureParkedFixture(vehicle: number) {
    SetVehicleEngineOn(vehicle, false, true, true);
    SetVehicleUndriveable(vehicle, true);
    SetVehicleHandbrake(vehicle, true);
    SetEntityCollision(vehicle, true, true);
    FreezeEntityPosition(vehicle, true);
}

function deleteOwnedVehicle(vehicle: number) {
    if (!isValidEntity(vehicle)) {
        return;
    }
    NetworkRequestControlOfEntity(vehicle);
    SetEntityAsMissionEntity(vehicle, true, true);
    DeleteVehicle(vehicle);
}
