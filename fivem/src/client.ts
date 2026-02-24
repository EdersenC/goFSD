// src/client.ts
import {SceneManager, SceneType} from "./sceneManger";
import {WeatherType} from "./environment";
import {DrivingStyle, Route, VehicleColor, VehicleModel} from "./egoService";
import {log} from "./helper";

log("[client] loaded");



const sceneManager = new SceneManager();
RegisterCommand("startScene", async () => {
    emitNet('demo:requestScenes');
}, false);

onNet("demo:responseScenes", async (scene: SceneType) => {
    sceneManager.addScene("SunnyDay", scene);
    console.log(scene)
    // sceneManager.shuffleWaypoints("SunnyDay");
    await sceneManager.executeAllScenes()
});
















async function findTeleportZ(x: number, y: number): Promise<number> {
    SetFocusPosAndVel(x, y, 1000.0, 0.0, 0.0, 0.0);

    for (let zProbe = 1000.0; zProbe >= 0.0; zProbe -= 50.0) {
        RequestCollisionAtCoord(x, y, zProbe);
        await wait(15);
        const [found, groundZ] = GetGroundZFor_3dCoord(x, y, zProbe, false);
        if (found) {
            ClearFocus();
            return groundZ;
        }
    }

    ClearFocus();
    return 100.0;
}
RegisterCommand("tpwaypoint", async () => {
    const waypointBlip = GetFirstBlipInfoId(8);
    if (!waypointBlip || !DoesBlipExist(waypointBlip)) {
        emit("chat:addMessage", {
            args: ["Teleport", "Set a waypoint first"],
        });
        return;
    }

    const ped = PlayerPedId();
    const inVehicle = IsPedInAnyVehicle(ped, false);
    const vehicle = inVehicle ? GetVehiclePedIsIn(ped, false) : 0;
    const isDriver = vehicle !== 0 && GetPedInVehicleSeat(vehicle, -1) === ped;
    const entityToMove = isDriver ? vehicle : ped;

    const [x, y] = GetBlipInfoIdCoord(waypointBlip) as unknown as [number, number, number];
    const z = await findTeleportZ(x, y);

    FreezeEntityPosition(entityToMove, true);
    RequestCollisionAtCoord(x, y, z);
    SetEntityCoordsNoOffset(entityToMove, x, y, z, false, false, true);

    if (isDriver) {
        SetVehicleOnGroundProperly(entityToMove);
    }

    await wait(60);
    FreezeEntityPosition(entityToMove, false);

    emit("chat:addMessage", {
        args: ["Teleport", "Teleported to waypoint"],
    });
}, false);

function wait(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}
