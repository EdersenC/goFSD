
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

export class SceneManager {

    private  Scenes = new Map<string, SceneType>();

    public async executeScene(name: string) {
        const scene = this.Scenes.get(name);
        if (scene) {
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

}