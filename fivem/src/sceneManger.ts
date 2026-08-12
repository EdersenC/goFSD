
import {Environment, EnvironmentService, WeatherType} from "./environment";
import {log, shuffleArray, wait} from "./helper";
import {
    DrivingStyle,
    Ego,
    EgoService,
    Route,
    SceneStoppedErrorCode,
    VehicleColor,
    VehicleModel
} from "./egoService";
import {defaultScene} from "./datasets";
import {createNoEgoControlTelemetry, EgoControlTelemetry} from "./controlTelemetry";
import {completeSuccessfulScene} from "./scene-completion";
import {
    placeVehicleAtStopSignLocation,
    StopSignLocationProbeOperations,
} from "./stop-sign/location-probe";
import {StopSignBatchHooks, StopSignRunner, STOP_SIGN_SCENE_NAME} from "./stop-sign/runner";
import {StopSignJob, StopSignPose, StopSignTelemetry} from "./stop-sign/types";
import {parseStopSignPose} from "./stop-sign/batch";
import {positionVehicleAtStopSignPose} from "./stop-sign/vehicle-positioning";
import {WorldIsolation} from "./world-isolation";

export {syncFlash} from "./syncFlash";

const egoService = new EgoService();
const envService = new EnvironmentService();
const stopSignRunner = new StopSignRunner(egoService);
const worldIsolation = new WorldIsolation(() => egoService.oldEgo?.vehicle.id ?? 0);
const stopSignLocationProbeOperations: StopSignLocationProbeOperations = {
    closestVehicleNode: ({x, y, z}) => {
        const [found, position, heading] = GetClosestVehicleNodeWithHeading(x, y, z, 1, 3, 0);
        return [found, position as [number, number, number], heading];
    },
    entityCoords: (entity) => GetEntityCoords(entity, false) as [number, number, number],
    entityHeading: (entity) => GetEntityHeading(entity),
    entityPitch: (entity) => GetEntityPitch(entity),
    entityRoll: (entity) => GetEntityRoll(entity),
    entitySpeed: (entity) => GetEntitySpeed(entity),
    entityOnGround: (entity) => IsVehicleOnAllWheels(entity),
    freezeEntity: (entity, frozen) => FreezeEntityPosition(entity, frozen),
    setEntityCoords: (entity, [x, y, z]) => SetEntityCoordsNoOffset(entity, x, y, z, false, false, true),
    setEntityHeading: (entity, heading) => SetEntityHeading(entity, heading),
    setVehicleForwardSpeed: (vehicle, speedMps) => SetVehicleForwardSpeed(vehicle, speedMps),
    setVehicleOnGround: (vehicle) => SetVehicleOnGroundProperly(vehicle),
    setFocus: ([x, y, z]) => SetFocusPosAndVel(x, y, z, 0, 0, 0),
    clearFocus: () => ClearFocus(),
    requestCollision: ([x, y, z]) => RequestCollisionAtCoord(x, y, z),
    collisionLoaded: (entity) => HasCollisionLoadedAroundEntity(entity),
    wait,
};
export const newScene = defaultScene;
const canonicalInnerCitySceneName = "inner-city-driving:default";

export type SceneType = {
    environment: Environment
    ego:Ego
}

type SceneExecutionOptions = {
    isCanceled: () => boolean
    shuffleWaypoints?: boolean
    runId?: string
}

function parseSceneName(sceneName: string): { sceneId: string; sceneVariant: string } {
    const [sceneId, sceneVariant] = sceneName.split(":");
    return {
        sceneId: sceneId || "unknown-scene",
        sceneVariant: sceneVariant || "default"
    };
}

function createRunId(): string {
    const now = new Date();
    const year = now.getFullYear();
    const month = pad2(now.getMonth() + 1);
    const day = pad2(now.getDate());
    const hours24 = now.getHours();
    const meridiem = hours24 >= 12 ? "PM" : "AM";
    const hours12 = hours24 % 12 || 12;
    const minutes = pad2(now.getMinutes());
    const seconds = pad2(now.getSeconds());
    const suffix = Math.random().toString(36).slice(2, 8);

    return `${year}-${month}-${day}_${pad2(hours12)}-${minutes}-${seconds}${meridiem}_${suffix}`;
}

function pad2(value: number): string {
    return String(value).padStart(2, "0");
}

function toNumber(value: unknown): number | null {
    if (typeof value === "number" && Number.isFinite(value)) {
        return value;
    }
    if (typeof value === "string" && value.trim() !== "") {
        const parsed = Number(value);
        return Number.isFinite(parsed) ? parsed : null;
    }
    return null;
}

function resolveNumericEnumValue(
    enumObj: Record<string, string | number>,
    value: unknown,
    fallback: number
): number {
    const fromNumber = toNumber(value);
    if (fromNumber !== null) {
        return fromNumber;
    }

    if (typeof value !== "string") {
        return fallback;
    }

    const target = value.trim().toLowerCase();
    if (!target) {
        return fallback;
    }

    for (const key of Object.keys(enumObj)) {
        if (Number.isFinite(Number(key))) {
            continue;
        }
        if (key.toLowerCase() === target) {
            const resolved = enumObj[key];
            return typeof resolved === "number" ? resolved : fallback;
        }
    }
    return fallback;
}

function resolveWeatherType(value: unknown): WeatherType {
    if (typeof value !== "string") {
        return WeatherType.CLEARING;
    }
    const target = value.trim().toUpperCase();
    for (const key of Object.keys(WeatherType)) {
        const candidate = (WeatherType as any)[key];
        if (typeof candidate === "string" && candidate.toUpperCase() === target) {
            return candidate as WeatherType;
        }
    }
    return WeatherType.CLEARING;
}

function resolveVehicleModel(value: unknown): string {
    if (typeof value !== "string") {
        return VehicleModel.Random;
    }
    const target = value.trim().toLowerCase();
    if (!target) {
        return VehicleModel.Random;
    }
    return target;
}

export function normalizeScenePayload(raw: unknown): SceneType {
    const parsed = typeof raw === "string" ? JSON.parse(raw) : raw;
    const scene = (parsed ?? {}) as any;

    const waypoints = Array.isArray(scene?.ego?.waypoints)
        ? scene.ego.waypoints
            .map((wp: any) => {
                const dst = Array.isArray(wp?.destination) ? wp.destination : [];
                return {
                    destination: [
                        toNumber(dst[0]) ?? 0,
                        toNumber(dst[1]) ?? 0,
                        toNumber(dst[2]) ?? 0
                    ] as [number, number, number]
                };
            })
            .filter((wp: any) => wp.destination.length === 3)
        : [];

    return {
        environment: {
            weatherType: {
                type: resolveWeatherType(scene?.environment?.weatherType?.type),
                persistent: Boolean(scene?.environment?.weatherType?.persistent)
            },
            Time: {
                hour: toNumber(scene?.environment?.Time?.hour) ?? 12,
                minute: toNumber(scene?.environment?.Time?.minute) ?? 0,
                second: toNumber(scene?.environment?.Time?.second) ?? 0,
                persistent: Boolean(scene?.environment?.Time?.persistent)
            }
        },
        ego: {
            vehicle: {
                id: toNumber(scene?.ego?.vehicle?.id) ?? 0,
                model: resolveVehicleModel(scene?.ego?.vehicle?.model),
                color: resolveNumericEnumValue(
                    VehicleColor as unknown as Record<string, string | number>,
                    scene?.ego?.vehicle?.color,
                    VehicleColor.Random
                ) as VehicleColor,
                // Keep string styles supported by converting known names -> enum value.
                drivingStyle: resolveNumericEnumValue(
                    DrivingStyle as unknown as Record<string, string | number>,
                    scene?.ego?.vehicle?.drivingStyle,
                    DrivingStyle.Cautious
                ) as DrivingStyle,
                maxSpeed: toNumber(scene?.ego?.vehicle?.maxSpeed) ?? 22
            },
            waypoints
        }
    };
}

export class SceneManager {

    private  Scenes = new Map<string, SceneType>();
    private activeSceneName: string | null = null;
    private stopCurrentSceneRequested = false;
    private stopAllScenesRequested = false;
    private runningAllScenes = false;
    private egoControlActive = false;

    public constructor() {
        worldIsolation.start();
    }

    public async executeScene(name: string, options: SceneExecutionOptions) {
        const resolvedName = this.resolveSceneName(name);
        const scene = this.Scenes.get(resolvedName);
        if (scene) {
            if (options.isCanceled()) {
                throw new Error(SceneStoppedErrorCode);
            }
            if (this.activeSceneName) {
                throw new Error(`Scene "${this.activeSceneName}" is already running`);
            }

            const runId = options?.runId?.trim() ? options.runId.trim() : createRunId();
            const preparedScene = cloneScene(scene);
            const shouldShuffleWaypoints = options?.shuffleWaypoints ?? true;
            if (shouldShuffleWaypoints && Array.isArray(preparedScene.ego?.waypoints) && preparedScene.ego.waypoints.length > 1) {
                preparedScene.ego.waypoints = shuffleArray([...preparedScene.ego.waypoints]) as Route[];
                log(`Shuffled ${preparedScene.ego.waypoints.length} waypoints for scene "${resolvedName}".`);
            }
            this.activeSceneName = resolvedName;
            this.stopCurrentSceneRequested = false;
            const isCanceled = () => options.isCanceled()
                || this.stopCurrentSceneRequested
                || this.stopAllScenesRequested;

            try {
                if (isCanceled()) {
                    throw new Error(SceneStoppedErrorCode);
                }

                await egoService.execute(preparedScene.ego, resolvedName, isCanceled, runId, preparedScene.environment);
                completeSuccessfulScene(resolvedName, egoService);
            } catch (error: any) {
                if (this.isSceneStoppedError(error)) {
                    log(`Scene "${resolvedName}" stopped by command.`);
                    return;
                }
                throw error;
            } finally {
                this.activeSceneName = null;
                this.stopCurrentSceneRequested = false;
            }

            log(`Scene "${resolvedName}" executed.`);
        } else {
            log(`Scene "${name}" not found.`);
        }
    }

    public async executeAllScenes(isCanceled: () => boolean) {
        console.log("Executing all scenes...", this.Scenes);
        this.runningAllScenes = true;
        this.stopAllScenesRequested = false;
        const runId = createRunId();
        try {
            for (const [name] of this.Scenes.entries()) {
                if (this.stopAllScenesRequested || isCanceled()) {
                    log("endAllScenes requested; stopping queued scenes.");
                    break;
                }
                await this.executeScene(name, {isCanceled, shuffleWaypoints: true, runId})
                if (this.stopAllScenesRequested || isCanceled()) {
                    log("endAllScenes requested; scene queue stopped.");
                    break;
                }
            }
        } finally {
            this.runningAllScenes = false;
            this.stopAllScenesRequested = false;
        }
    }

    public endScene(): boolean {
        if (!this.activeSceneName) {
            if (this.disposeInactiveManagedEgo("endScene requested for retained managed ego")) {
                return true;
            }
            log("No active scene to end.");
            return !this.runningAllScenes;
        }

        if (stopSignRunner.isRunning()) {
            this.stopCurrentSceneRequested = true;
            stopSignRunner.requestStop();
            console.log(`Stopping stop-sign run "${this.activeSceneName}"...`);
            return false;
        }

        if (this.egoControlActive) {
            this.stopEgoControl();
            return true;
        }

        this.stopCurrentSceneRequested = true;
        egoService.requestStop(`endScene requested for "${this.activeSceneName}"`);
        log(`Stopping scene "${this.activeSceneName}"...`);
        return false;
    }

    public endAllScenes(): boolean {
        this.stopAllScenesRequested = true;
        if (stopSignRunner.isRunning()) {
            this.stopCurrentSceneRequested = true;
            stopSignRunner.requestStop();
            console.log("Stopping active stop-sign batch...");
            return false;
        }
        if (this.egoControlActive) {
            this.stopEgoControl();
            this.stopAllScenesRequested = false;
            return true;
        }
        if (this.activeSceneName) {
            this.stopCurrentSceneRequested = true;
            egoService.requestStop(`endAllScenes requested for "${this.activeSceneName}"`);
            log(`Stopping scene "${this.activeSceneName}" and remaining queue...`);
            return false;
        }

        if (this.runningAllScenes) {
            log("Stopping remaining queued scenes...");
            return false;
        }

        if (this.disposeInactiveManagedEgo("endAllScenes requested for retained managed ego")) {
            this.stopAllScenesRequested = false;
            return true;
        }

        this.stopAllScenesRequested = false;
        log("No active scenes to end.");
        return true;
    }

    public async startEgoControl(isCanceled: () => boolean = () => false) {
        if (isCanceled()) {
            throw new Error("Ego control start was canceled before initialization");
        }
        if (this.activeSceneName) {
            throw new Error(`Scene "${this.activeSceneName}" is already running`);
        }

        const runId = createRunId();
        if (!defaultScene) {
            throw new Error("default scene invariant violated: inner-city-driving dataset is missing");
        }
        const clonedScene = cloneScene(defaultScene);
        clonedScene.ego.vehicle.model = VehicleModel.Sultan;
        clonedScene.ego.vehicle.color = VehicleColor.Blue;
        this.activeSceneName = "ego-control";
        this.egoControlActive = true;
        this.stopCurrentSceneRequested = false;

        try {
            envService.execute(clonedScene.environment);
            await egoService.executeEgo(clonedScene.ego, "ego-control", isCanceled, runId);
            if (isCanceled()) {
                egoService.disposeCurrentEgo();
                throw new Error("Ego control start was canceled during initialization");
            }
            egoService.configureManualRouteContext(clonedScene.ego);
            if (isCanceled()) {
                egoService.disposeCurrentEgo();
                throw new Error("Ego control start was canceled before isolation setup");
            }
            const cleared = worldIsolation.start();
            log(
                `Continuous 1000m world isolation active: removed ${cleared.removedVehicles} vehicles `
                + `and ${cleared.removedPeds} NPCs initially; your EGO car is protected.`
            );
        } catch (error) {
            this.egoControlActive = false;
            this.activeSceneName = null;
            throw error;
        }
    }

    public stopEgoControl() {
        if (!this.egoControlActive) {
            log("No active ego control to stop.");
            return;
        }

        egoService.forceSafeStopAndDisposeCurrentEgo("ego control stopped");
        this.egoControlActive = false;
        this.activeSceneName = null;
        this.stopCurrentSceneRequested = false;
        log("Stopped ego control.");
    }

    public shutdownWorldIsolation() {
        worldIsolation.stop();
    }

    private disposeInactiveManagedEgo(reason: string): boolean {
        if (!egoService.hasCurrentEgo()) {
            return false;
        }
        egoService.forceSafeStopAndDisposeCurrentEgo(reason);
        log("Disposed retained managed ego.");
        return true;
    }

    public forceSafeCleanup(reason: string) {
        this.stopCurrentSceneRequested = true;
        this.stopAllScenesRequested = true;
        stopSignRunner.requestStop();
        egoService.forceSafeStopAndDisposeCurrentEgo(reason);
        this.egoControlActive = false;
        this.activeSceneName = null;
        this.runningAllScenes = false;
        this.stopCurrentSceneRequested = false;
        this.stopAllScenesRequested = false;
    }

    public setStopSignTarget(): {signPose: StopSignPose; stopLinePose: StopSignPose; egoStopPose: StopSignPose; exitPose: StopSignPose} {
        if (!this.egoControlActive || this.activeSceneName !== "ego-control") {
            throw new Error("Stop-sign calibration requires the active setup car");
        }
        return stopSignRunner.calibrateTargetFromCurrentEgo();
    }

    public clearStopSignTarget() {
        stopSignRunner.clearTarget();
    }

    public async probeStopSignTarget(rawRequest: unknown, assertCurrent: () => void) {
        if (!this.egoControlActive || this.activeSceneName !== "ego-control") {
            throw new Error("Stop-sign location probing requires the active setup car");
        }
        const ego = egoService.oldEgo;
        if (!ego || !DoesEntityExist(ego.vehicle.id)) {
            throw new Error("Stop-sign location probing requires a valid managed setup car");
        }
        if (GetPedInVehicleSeat(ego.vehicle.id, -1) !== PlayerPedId()) {
            throw new Error("Stop-sign location probing requires the player in the managed driver seat");
        }

        const probe = await placeVehicleAtStopSignLocation(
            ego.vehicle.id,
            rawRequest,
            stopSignLocationProbeOperations,
            assertCurrent,
        );
        assertCurrent();
        SetVehicleEngineOn(ego.vehicle.id, true, true, false);
        SetVehicleUndriveable(ego.vehicle.id, false);
        egoService.startCaptureCamera(ego);
        const target = stopSignRunner.calibrateTargetFromCurrentEgo();
        return {probe, target};
    }

    public async teleportStopSignStart(rawPose: unknown, assertCurrent: () => void) {
        if (!this.egoControlActive || this.activeSceneName !== "ego-control") {
            throw new Error("Saved Start teleport requires the active setup car");
        }
        const ego = egoService.oldEgo;
        if (!ego || !DoesEntityExist(ego.vehicle.id)) {
            throw new Error("Saved Start teleport requires a valid managed setup car");
        }
        if (GetPedInVehicleSeat(ego.vehicle.id, -1) !== PlayerPedId()) {
            throw new Error("Saved Start teleport requires the player in the managed driver seat");
        }

        const pose = parseStopSignPose(rawPose, "stopSignStartPose");
        await positionVehicleAtStopSignPose(ego.vehicle.id, pose, assertCurrent);
        assertCurrent();
        SetVehicleEngineOn(ego.vehicle.id, true, true, false);
        SetVehicleUndriveable(ego.vehicle.id, false);
        egoService.startCaptureCamera(ego);
        return pose;
    }

    public async startStopSignBatch(
        batchId: string,
        jobs: StopSignJob[],
        hooks: StopSignBatchHooks,
    ): Promise<number> {
        if (this.activeSceneName && !this.egoControlActive) {
            throw new Error(`Scene "${this.activeSceneName}" is already running`);
        }
        if (this.egoControlActive) {
            this.egoControlActive = false;
            this.activeSceneName = null;
        }
        const cleared = worldIsolation.start();
        log(
            `Stop-sign preflight cleared ${cleared.removedVehicles} unoccupied vehicles `
            + `and ${cleared.removedPeds} NPCs before spawning the collection car.`
        );
        this.activeSceneName = STOP_SIGN_SCENE_NAME;
        this.stopCurrentSceneRequested = false;
        try {
            return await stopSignRunner.runBatch(batchId, jobs, hooks);
        } finally {
            this.egoControlActive = egoService.hasCurrentEgo();
            this.activeSceneName = this.egoControlActive ? "ego-control" : null;
            this.stopCurrentSceneRequested = false;
        }
    }

    public currentEgoSpeed(): number | null {
        return egoService.currentSpeed();
    }

    public currentEgoYaw(): number | null {
        return egoService.currentYaw();
    }

    public currentEgoRouteForwardDelta(): number | null {
        return egoService.currentRouteForwardDelta();
    }

    public currentEgoControlTelemetry(): EgoControlTelemetry & StopSignTelemetry {
        const telemetry = egoService.currentControlTelemetry()
            ?? createNoEgoControlTelemetry(GetGameTimer());
        return {
            ...telemetry,
            ...stopSignRunner.currentTelemetry(),
        };
    }

    public addScene(name: string, scene: SceneType) {
        this.Scenes.set(name, scene);
    }

    private resolveSceneName(name: string): string {
        if (this.Scenes.has(name)) {
            return name;
        }
        const parsed = parseSceneName(name);
        if (parsed.sceneId === "inner-city-driving") {
            return canonicalInnerCitySceneName;
        }
        return name;
    }

    public getScene(name: string): SceneType | undefined {
        return this.Scenes.get(name);
    }

    public removeScene(name: string) {
        this.Scenes.delete(name);
    }

    public listScenes(): string[] {
        return Array.from(this.Scenes.keys());
    }

    public shuffleWaypoints(name: string) {
        const scene = this.Scenes.get(name);
        if (scene) {
            scene.ego.waypoints = shuffleArray(scene.ego.waypoints as any[]) as Route[];
            log(`Waypoints for scene "${name}" shuffled.`);
        } else {
            log(`Scene "${name}" not found.`);
        }
    }

    private isSceneStoppedError(error: unknown): boolean {
        if (!error) {
            return false;
        }
        return error instanceof Error && error.message === SceneStoppedErrorCode;
    }

}

function cloneScene(scene: SceneType): SceneType {
    return JSON.parse(JSON.stringify(scene)) as SceneType;
}
