
import {Environment, EnvironmentService, WeatherType} from "./environment";
import {log, shuffleArray, wait} from "./helper";
import {DrivingStyle, Ego, EgoService, Route, SceneStoppedErrorCode, VehicleColor, VehicleModel} from "./egoService";
import {defaultScene} from "./datasets";

const egoService = new EgoService();
const envService = new EnvironmentService();
export const newScene = defaultScene;

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

    public async executeScene(name: string) {
        const scene = this.Scenes.get(name);
        if (scene) {
            if (this.activeSceneName) {
                throw new Error(`Scene "${this.activeSceneName}" is already running`);
            }

            const runId = createRunId();
            this.activeSceneName = name;
            this.stopCurrentSceneRequested = false;

            try {
                if (this.stopCurrentSceneRequested) {
                    throw new Error(SceneStoppedErrorCode);
                }

                envService.execute(scene.environment);
                await egoService.execute(scene.ego, name, runId);
            } catch (error: any) {
                if (this.isSceneStoppedError(error)) {
                    log(`Scene "${name}" stopped by command.`);
                    return;
                }
                throw error;
            } finally {
                this.activeSceneName = null;
                this.stopCurrentSceneRequested = false;
            }

            log(`Scene "${name}" executed.`);
        } else {
            log(`Scene "${name}" not found.`);
        }
    }

    public async executeAllScenes() {
        console.log("Executing all scenes...", this.Scenes);
        this.runningAllScenes = true;
        this.stopAllScenesRequested = false;
        try {
            for (const [name, scene] of this.Scenes.entries()) {
                if (this.stopAllScenesRequested) {
                    log("endAllScenes requested; stopping queued scenes.");
                    break;
                }
                await this.executeScene(name)
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

    public addScene(name: string, scene: SceneType) {
        this.Scenes.set(name, scene);
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
    return flashUntilGameMs
}
