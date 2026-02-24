
import {Environment, EnvironmentService, WeatherType} from "./environment";
import {log, shuffleArray, wait} from "./helper";
import {DrivingStyle, Ego, EgoService, Route, VehicleColor, VehicleModel} from "./egoService";

const egoService = new EgoService();
const envService = new EnvironmentService();

const first: Route = {
    destination: [425.0, -979.0, 30.0] // Pillbox Hill Medical Center
}
const second: Route = {
    destination: [-72.0, -818.0, 326.0] // Vinewood Hills
}

const third: Route = {
    destination: [1855.0, 3686.0, 34.0] // Sandy Shores Airfield
}

const delPerroPier: Route = {
    destination: [-1850.0, -1203.0, 13.0] // Del Perro Pier
}

const lsia: Route = {
    destination: [-1037.0, -2737.0, 20.0] // Los Santos International Airport
}

const vinewoodSign: Route = {
    destination: [711.0, 1198.0, 348.0] // Vinewood Sign
}

const groveStreet: Route = {
    destination: [110.0, -1945.0, 20.0] // Grove Street
}

export const newScene = {
    environment: {
        weatherType: {
            type:WeatherType.CLEARING,
            persistent: true
        },
        Time: {
            hour: 12,
            minute: 0,
            second: 0
        }
    },
    ego: {
        vehicle:{
            id: 0,
            model: VehicleModel.Random,
            color: VehicleColor.Random,
            maxSpeed: 22,
            drivingStyle:DrivingStyle.Cautious,
        },
        waypoints: [first, delPerroPier, third , lsia, groveStreet, second, vinewoodSign]
    }
}

const another = {
    environment: {
        weatherType: {
            type:WeatherType.BLIZZARD,
            persistent: true
        },
        Time: {
            hour: 5,
            minute: 39,
            second: 0
        }
    },
    ego: {
        vehicle:{
            id: 0,
            model: "tailgater",
            color: VehicleColor.Red,
            maxSpeed: 25,
            drivingStyle:DrivingStyle.Cautious
        },
        waypoints: [first, second, third, delPerroPier, lsia, groveStreet, vinewoodSign]
    }
}

export type SceneType = {
    environment: Environment
    ego:Ego
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

    public async executeScene(name: string) {
        const scene = this.Scenes.get(name);
        if (scene) {
            const sceneStart = await this.syncFlash()
            envService.execute(scene.environment);
            await egoService.execute(scene.ego);
            log(`Scene "${name}" executed.`);
        } else {
            log(`Scene "${name}" not found.`);
        }
    }

    public async executeAllScenes() {
        console.log("Executing all scenes...", this.Scenes);
        for (const [name, scene] of this.Scenes.entries()) {
           await this.executeScene(name)
        }
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


    private async syncFlash(durationMs = 250) :Promise<number> {
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

        return startTime
    }

}
