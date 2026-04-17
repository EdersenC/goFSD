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

type ControlCommandType = "startScene" | "runAllScenes" | "endScene" | "endAllScenes" | "startEgo" | "stopEgo";
type ControlRuntimeStatus = "idle" | "runningScene" | "runningAllScenes" | "stopping" | "error";

type ControlCommand = {
    id: string
    type: ControlCommandType
    sceneName?: string
}

type AvailableScene = {
    name: string
    label: string
}

function registerInnerCityScenes() {
    for (const variant of innerCitySceneNames) {
        sceneManager.addScene(`inner-city-driving:${variant}`, innerCityDrivingScenes[variant]);
    }
}

function humanizeSceneVariant(variant: string): string {
    return variant
        .replace(/([a-z])([A-Z])/g, "$1 $2")
        .replace(/^\w/, (match) => match.toUpperCase());
}

function buildAvailableScenes(): AvailableScene[] {
    return innerCitySceneNames.map((variant) => ({
        name: `inner-city-driving:${variant}`,
        label: `Inner City Driving - ${humanizeSceneVariant(variant)}`
    }));
}

function publishAvailableScenes() {
    emitNet("control:availableScenesResponse", buildAvailableScenes());
}

function reportControlStatus(status: ControlRuntimeStatus, activeSceneName = "", lastError = "") {
    emitNet("control:statusUpdate", {
        status,
        activeSceneName,
        lastError
    });
}

async function executeSceneByName(sceneName: string) {
    reportControlStatus("runningScene", sceneName);
    try {
        await sceneManager.executeScene(sceneName);
        reportControlStatus("idle");
    } catch (error: any) {
        const message = error?.message ?? `Failed to execute scene "${sceneName}"`;
        reportControlStatus("error", sceneName, message);
        throw error;
    }
}

async function executeAllScenesControlled() {
    reportControlStatus("runningAllScenes");
    try {
        await sceneManager.executeAllScenes();
        reportControlStatus("idle");
    } catch (error: any) {
        const message = error?.message ?? "Failed to execute all scenes";
        reportControlStatus("error", "", message);
        throw error;
    }
}

async function executeEgoControl() {
    reportControlStatus("runningScene", "ego-control");
    try {
        await sceneManager.startEgoControl();
    } catch (error: any) {
        const message = error?.message ?? "Failed to start ego control";
        reportControlStatus("error", "ego-control", message);
        throw error;
    }
}

function stopEgoControl() {
    sceneManager.stopEgoControl();
    reportControlStatus("idle");
}

function requestEndScene() {
    reportControlStatus("stopping");
    sceneManager.endScene();
}

function requestEndAllScenes() {
    reportControlStatus("stopping");
    sceneManager.endAllScenes();
}

registerInnerCityScenes();
emitNet("control:registerClient");
reportControlStatus("idle");
publishAvailableScenes();

RegisterCommand("startScene", async (_source: number, args: string[]) => {
    const requestedVariant = args[0] as InnerCityDrivingSceneVariant | undefined;
    const variant = requestedVariant && requestedVariant in innerCityDrivingScenes
        ? requestedVariant
        : "default";
    const sceneName = `inner-city-driving:${variant}`;

    await executeSceneByName(sceneName);
}, false);

RegisterCommand("runAllScenes", async () => {
    await executeAllScenesControlled();
}, false);

RegisterCommand("endScene", () => {
    requestEndScene();
}, false);

RegisterCommand("endAllScenes", () => {
    requestEndAllScenes();
}, false);

RegisterCommand("startEgo", async (_source: number, args: string[]) => {
    await executeEgoControl();
}, false);

RegisterCommand("stopEgo", () => {
    stopEgoControl();
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
    await executeAllScenesControlled()
});

onNet("control:executeCommand", async (command: ControlCommand) => {
    try {
        switch (command?.type) {
            case "startScene":
                if (!command.sceneName) {
                    throw new Error("startScene command missing sceneName");
                }
                await executeSceneByName(command.sceneName);
                break;
            case "runAllScenes":
                await executeAllScenesControlled();
                break;
            case "endScene":
                requestEndScene();
                break;
            case "endAllScenes":
                requestEndAllScenes();
                break;
            case "startEgo":
                await executeEgoControl();
                break;
            case "stopEgo":
                stopEgoControl();
                break;
            default:
                throw new Error(`Unsupported control command: ${command?.type ?? "unknown"}`);
        }
    } catch (error: any) {
        const message = error?.message ?? "Failed to execute control command";
        reportControlStatus("error", command?.sceneName ?? "", message);
        log(message);
    }
});







onNet("control:requestAvailableScenes", () => {
    publishAvailableScenes();
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
