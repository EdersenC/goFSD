
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

const egoService = new EgoService();
const envService = new EnvironmentService();
export const newScene = defaultScene;
const canonicalInnerCitySceneName = "inner-city-driving:default";

export type SceneType = {
    environment: Environment
    ego:Ego
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
                second: toNumber(scene?.environment?.Time?.second) ?? 0
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

    public async executeScene(name: string, options?: { shuffleWaypoints?: boolean; runId?: string }) {
        const resolvedName = this.resolveSceneName(name);
        const scene = this.Scenes.get(resolvedName);
        if (scene) {
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

            try {
                if (this.stopCurrentSceneRequested) {
                    throw new Error(SceneStoppedErrorCode);
                }

                await egoService.execute(preparedScene.ego, resolvedName, runId, preparedScene.environment);
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

    public async executeAllScenes() {
        console.log("Executing all scenes...", this.Scenes);
        this.runningAllScenes = true;
        this.stopAllScenesRequested = false;
        const runId = createRunId();
        try {
            for (const [name] of this.Scenes.entries()) {
                if (this.stopAllScenesRequested) {
                    log("endAllScenes requested; stopping queued scenes.");
                    break;
                }
                await this.executeScene(name, {shuffleWaypoints: true, runId})
                if (this.stopAllScenesRequested) {
                    log("endAllScenes requested; scene queue stopped.");
                    break;
                }
            }
        } finally {
            this.runningAllScenes = false;
            this.stopAllScenesRequested = false;
        }
    }

    public endScene() {
        if (!this.activeSceneName) {
            log("No active scene to end.");
            return;
        }

        this.stopCurrentSceneRequested = true;
        egoService.requestStop(`endScene requested for "${this.activeSceneName}"`);
        log(`Stopping scene "${this.activeSceneName}"...`);
    }

    public endAllScenes() {
        this.stopAllScenesRequested = true;
        if (this.activeSceneName) {
            this.stopCurrentSceneRequested = true;
            egoService.requestStop(`endAllScenes requested for "${this.activeSceneName}"`);
            log(`Stopping scene "${this.activeSceneName}" and remaining queue...`);
            return;
        }

        if (this.runningAllScenes) {
            log("Stopping remaining queued scenes...");
            return;
        }

        log("No active scenes to end.");
    }

    public async startEgoControl() {
        if (this.activeSceneName) {
            throw new Error(`Scene "${this.activeSceneName}" is already running`);
        }

        const runId = createRunId();
        const clonedScene = cloneScene(defaultScene);
        this.activeSceneName = "ego-control";
        this.egoControlActive = true;
        this.stopCurrentSceneRequested = false;

        try {
            envService.execute(clonedScene.environment);
            await egoService.executeEgo(clonedScene.ego, "ego-control", runId);
            egoService.configureManualRouteContext(clonedScene.ego);
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

        egoService.disposeCurrentEgo();
        this.egoControlActive = false;
        this.activeSceneName = null;
        this.stopCurrentSceneRequested = false;
        log("Stopped ego control.");
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

    public currentEgoControlTelemetry(): {
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
        routeForwardDelta: number | null
        routeHeadingError: number | null
        routeDistance: number | null
        hasLeadVehicle: boolean
        leadVehicleDistance: number | null
        gameTimeMs: number
    } | null {
        return egoService.currentControlTelemetry();
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

export async function syncFlash(durationMs = 250) :Promise<number> {
    const startTime = GetGameTimer();
    const flashUntilGameMs = startTime + durationMs;

    setTick(() => {
        if (GetGameTimer() < flashUntilGameMs) {
            // Full screen white rectangle
            DrawRect(
                0.5, 0.5,   // center
                1.0, 1.0,   // full width/height
                255, 255, 255, 255 // RGBA white
            );
        }
    });
    return  startTime
}
