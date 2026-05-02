// src/client.ts
import {SceneManager, SceneType} from "./sceneManger";
import {log} from "./helper";
import {
    innerCityDrivingScenes,
    type InnerCityDrivingSceneVariant
} from "./datasets/inner-city-driving";

const CLIENT_BUILD_ID = "2026-04-21-capture-failfast-v1";
log(`[client] loaded build=${CLIENT_BUILD_ID}`);



const sceneManager = new SceneManager();
const innerCitySceneNames = Object.keys(innerCityDrivingScenes) as InnerCityDrivingSceneVariant[];
const canonicalInnerCitySceneName = "inner-city-driving:default";
const innerCitySceneBaseLabel = "Inner City Driving";

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

type ControlTelemetryUpdate = {
    currentSpeed: number
    currentYaw: number
    yawRate: number
    steering: number
    acceleration: number
    brakePressureAvg: number
    vehicleExists: boolean
    isInVehicle: boolean
    positionX?: number
    positionY?: number
    positionZ?: number
    velocityX?: number
    velocityY?: number
    velocityZ?: number
    pitchDeg?: number
    rollDeg?: number
    gear?: number
    rpm?: number
    wheelAngle?: number
    onGround?: boolean
    collisionState?: string
    routeDirectionCode: number
    routeDirectionDistanceM: number
    routeDirectionUnknown: number
    routeDirectionKeepStraight: number
    routeDirectionTurnLeft: number
    routeDirectionTurnRight: number
    routeDirectionRerouteWrongWay: number
    routeForwardDelta: number
    routeHeadingError: number
    routeDistance: number
    leadVehicleDistance: number
    hasLeadVehicle: boolean
    timestampMs: number
    gameTimeMs: number
}

const TELEMETRY_INTERVAL_MS = 67;
const CONTROL_REGISTER_INTERVAL_MS = 5000;

function registerInnerCityScenes() {
    for (const variant of innerCitySceneNames) {
        const sceneName = variant === "default"
            ? canonicalInnerCitySceneName
            : `inner-city-driving:${variant}`;
        sceneManager.addScene(sceneName, innerCityDrivingScenes[variant]);
    }
}

function humanizeSceneVariant(variant: string): string {
    return variant
        .replace(/([a-z])([A-Z])/g, "$1 $2")
        .replace(/^\w/, (match) => match.toUpperCase());
}

function buildAvailableScenes(): AvailableScene[] {
    return innerCitySceneNames.map((variant) => ({
        name: variant === "default"
            ? canonicalInnerCitySceneName
            : `inner-city-driving:${variant}`,
        label: `${innerCitySceneBaseLabel} - ${humanizeSceneVariant(variant)}`
    }));
}

function publishAvailableScenes() {
    emitNet("control:availableScenesResponse", buildAvailableScenes());
}

function registerControlClient(reason: string) {
    log(`[client] registering control client session (${reason})`);
    emitNet("control:registerClient");
}

function sendControlClientHeartbeat() {
    emitNet("control:clientHeartbeat");
}

function publishTelemetry(update: ControlTelemetryUpdate) {
    emitNet("control:telemetryUpdate", update);
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
    log("[client] starting ego control");
    try {
        await sceneManager.startEgoControl();
        log("[client] ego control started");
    } catch (error: any) {
        const message = error?.message ?? "Failed to start ego control";
        reportControlStatus("error", "ego-control", message);
        log(`[client] ego control failed: ${message}`);
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
registerControlClient("startup");
reportControlStatus("idle");
publishAvailableScenes();

setInterval(() => {
    sendControlClientHeartbeat();
}, CONTROL_REGISTER_INTERVAL_MS);

let lastTelemetrySentAt = 0;
let lastRouteForwardDelta = 0;
let lastRouteHeadingError = 0;
let lastRouteDistance = 0;
let lastLeadVehicleDistance = 100;
let lastTelemetryMissingLogAt = 0;
let lastTelemetryDebugLogAt = 0;
setTick(() => {
    const now = GetGameTimer();
    if (now-lastTelemetrySentAt < TELEMETRY_INTERVAL_MS) {
        return;
    }
    const telemetry = sceneManager.currentEgoControlTelemetry();
    if (!telemetry) {
        if (now - lastTelemetryMissingLogAt >= 1000) {
            lastTelemetryMissingLogAt = now;
            log(`[client] telemetry unavailable egoActive=${sceneManager.currentEgoSpeed() !== null}`);
        }
        return;
    }
    const routeForwardDelta = telemetry.routeForwardDelta ?? lastRouteForwardDelta;
    const routeHeadingError = telemetry.routeHeadingError ?? lastRouteHeadingError;
    const routeDistance = telemetry.routeDistance ?? lastRouteDistance;
    const leadVehicleDistance = telemetry.hasLeadVehicle
        ? (telemetry.leadVehicleDistance ?? lastLeadVehicleDistance)
        : 100;
    if (telemetry.routeForwardDelta !== null) {
        lastRouteForwardDelta = telemetry.routeForwardDelta;
    }
    if (telemetry.routeHeadingError !== null) {
        lastRouteHeadingError = telemetry.routeHeadingError;
    }
    if (telemetry.routeDistance !== null) {
        lastRouteDistance = telemetry.routeDistance;
    }
    if (telemetry.hasLeadVehicle && telemetry.leadVehicleDistance !== null) {
        lastLeadVehicleDistance = telemetry.leadVehicleDistance;
    }
    lastTelemetrySentAt = now;
    publishTelemetry({
        currentSpeed: telemetry.currentSpeed,
        currentYaw: telemetry.currentYaw,
        yawRate: telemetry.yawRate,
        steering: telemetry.steering,
        acceleration: telemetry.acceleration,
        brakePressureAvg: telemetry.brakePressureAvg,
        vehicleExists: telemetry.vehicleExists,
        isInVehicle: telemetry.isInVehicle,
        positionX: telemetry.positionX,
        positionY: telemetry.positionY,
        positionZ: telemetry.positionZ,
        velocityX: telemetry.velocityX,
        velocityY: telemetry.velocityY,
        velocityZ: telemetry.velocityZ,
        pitchDeg: telemetry.pitchDeg,
        rollDeg: telemetry.rollDeg,
        gear: telemetry.gear,
        rpm: telemetry.rpm,
        wheelAngle: telemetry.wheelAngle,
        onGround: telemetry.onGround,
        collisionState: telemetry.collisionState,
        routeDirectionCode: telemetry.routeDirectionCode,
        routeDirectionDistanceM: telemetry.routeDirectionDistanceM,
        routeDirectionUnknown: telemetry.routeDirectionUnknown,
        routeDirectionKeepStraight: telemetry.routeDirectionKeepStraight,
        routeDirectionTurnLeft: telemetry.routeDirectionTurnLeft,
        routeDirectionTurnRight: telemetry.routeDirectionTurnRight,
        routeDirectionRerouteWrongWay: telemetry.routeDirectionRerouteWrongWay,
        routeForwardDelta,
        routeHeadingError,
        routeDistance,
        leadVehicleDistance,
        hasLeadVehicle: telemetry.hasLeadVehicle,
        timestampMs: Date.now(),
        gameTimeMs: telemetry.gameTimeMs,
    });
    if (now - lastTelemetryDebugLogAt >= 2000) {
        lastTelemetryDebugLogAt = now;
        log(
            `[client] telemetry publish speed=${telemetry.currentSpeed.toFixed(2)} ` +
            `yawRate=${telemetry.yawRate.toFixed(2)} ` +
            `steer=${telemetry.steering.toFixed(2)} ` +
            `accel=${telemetry.acceleration.toFixed(2)} ` +
            `brakeAvg=${telemetry.brakePressureAvg.toFixed(2)} ` +
            `navCode=${telemetry.routeDirectionCode} ` +
            `navOneHot=[u=${Number(telemetry.routeDirectionUnknown || 0).toFixed(0)}, ` +
            `s=${Number(telemetry.routeDirectionKeepStraight || 0).toFixed(0)}, ` +
            `l=${Number(telemetry.routeDirectionTurnLeft || 0).toFixed(0)}, ` +
            `r=${Number(telemetry.routeDirectionTurnRight || 0).toFixed(0)}, ` +
            `w=${Number(telemetry.routeDirectionRerouteWrongWay || 0).toFixed(0)}] ` +
            `routeFwd=${routeForwardDelta.toFixed(2)} ` +
            `routeHeading=${routeHeadingError.toFixed(2)} ` +
            `routeDistance=${routeDistance.toFixed(2)} ` +
            `hasLead=${String(telemetry.hasLeadVehicle)} ` +
            `leadDistance=${leadVehicleDistance.toFixed(2)}`
        );
    }
});

RegisterCommand("startScene", async (_source: number, args: string[]) => {
    const requestedVariant = args[0] as InnerCityDrivingSceneVariant | undefined;
    const variant = requestedVariant && requestedVariant in innerCityDrivingScenes
        ? requestedVariant
        : "default";
    const sceneName = variant === "default"
        ? canonicalInnerCitySceneName
        : `inner-city-driving:${variant}`;

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
        log(`[client] received control command id=${command?.id ?? "unknown"} type=${command?.type ?? "unknown"} scene=${command?.sceneName ?? ""}`);
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
        log(`[client] command failed: ${message}`);
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
