import {isValidEntity, wait} from "../helper";
import {StopSignPose} from "./types";

const defaultSettleMs = 1000;

/** Places an existing managed vehicle at an exact saved lane pose and leaves it ready to drive. */
export async function positionVehicleAtStopSignPose(
    vehicle: number,
    pose: StopSignPose,
    assertCurrent: () => void = () => undefined,
) {
    if (!isValidEntity(vehicle)) {
        throw new Error("Cannot position an invalid stop-sign vehicle");
    }

    assertCurrent();
    SetEntityRecordsCollisions(vehicle, false);
    FreezeEntityPosition(vehicle, true);
    holdVehicle(vehicle);
    try {
        RequestCollisionAtCoord(pose.x, pose.y, pose.z);
        SetEntityCoordsNoOffset(vehicle, pose.x, pose.y, pose.z + 0.75, false, false, true);
        SetEntityHeading(vehicle, pose.heading);
        SetEntityVelocity(vehicle, 0, 0, 0);
        SetVehicleEngineOn(vehicle, true, true, false);
        SetVehicleUndriveable(vehicle, false);
        SetVehicleOnGroundProperly(vehicle);
        await wait(defaultSettleMs);
        assertCurrent();
        SetVehicleOnGroundProperly(vehicle);
    } finally {
        if (isValidEntity(vehicle)) {
            SetEntityRecordsCollisions(vehicle, true);
            FreezeEntityPosition(vehicle, false);
        }
    }

    assertCurrent();
    SetVehicleBrake(vehicle, false);
    SetVehicleHandbrake(vehicle, false);
    await wait(150);
    assertCurrent();
}

function holdVehicle(vehicle: number) {
    SetVehicleForwardSpeed(vehicle, 0);
    SetVehicleBrake(vehicle, true);
    SetVehicleHandbrake(vehicle, true);
    SetVehicleSteerBias(vehicle, 0);
}
