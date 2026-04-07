// src/client.ts
import {SceneManager, SceneType} from "./sceneManger";
import {log} from "./helper";
import {
    innerCityDrivingScenes,
    type InnerCityDrivingSceneVariant
} from "./datasets/inner-city-driving";

log("[client] loaded");



const sceneManager = new SceneManager();
const innerCitySceneNames = Object.keys(innerCityDrivingScenes) as InnerCityDrivingSceneVariant[];

function registerInnerCityScenes() {
    for (const variant of innerCitySceneNames) {
        sceneManager.addScene(`inner-city-driving:${variant}`, innerCityDrivingScenes[variant]);
    }
}

registerInnerCityScenes();

RegisterCommand("startScene", async (_source: number, args: string[]) => {
    const requestedVariant = args[0] as InnerCityDrivingSceneVariant | undefined;
    const variant = requestedVariant && requestedVariant in innerCityDrivingScenes
        ? requestedVariant
        : "default";
    const sceneName = `inner-city-driving:${variant}`;

    await sceneManager.executeScene(sceneName);
}, false);

RegisterCommand("listSceneVariants", () => {
    const variants = innerCitySceneNames.join(", ");
    emit("chat:addMessage", {
        args: ["Scenes", variants],
    });
}, false);

onNet("demo:responseScenes", async (scene: SceneType) => {
    sceneManager.addScene("remote-scene", scene);
    console.log(scene)
    // sceneManager.shuffleWaypoints("remote-scene");
    await sceneManager.executeAllScenes()
});

RegisterCommand("coords", () => {
    const [x, y, z] = GetEntityCoords(PlayerPedId(), true) as [number, number, number];
    const formatted = `${x.toFixed(3)}, ${y.toFixed(3)}, ${z.toFixed(3)}`;

    console.log(`[coords] ${formatted}`);
    emit("chat:addMessage", {
        args: ["Coords", formatted],
    });
}, false);
















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
