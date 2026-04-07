import {WeatherType} from "../environment";
import {DrivingStyle, Ego, VehicleColor, VehicleModel} from "../egoService";
import {SceneType} from "../sceneManger";
import {routePresets} from "./routes";

const innerCityDrivingWaypoints = [
    routePresets.legionSquare,
    routePresets.portolaDriveParking,
    routePresets.missionRowPoliceStation,
    routePresets.richardsMajesticStudio,
    routePresets.textileCityMarket,
    routePresets.pillboxHillGarage,
    routePresets.weazelPlazaGarage,
    routePresets.strawberryCarWash,
    routePresets.rockfordPlaza,
    routePresets.ranchoLTDGasoline,
    routePresets.pillboxMedicalCenter,
    routePresets.backlotCityStudioGate,
    routePresets.littleSeoulGasStation,
    routePresets.strawberryLTDGasoline,
    routePresets.vespucciCanals,
    routePresets.orientalTheater,
    routePresets.delPerroParkingGarage,
    routePresets.littleSeoulArcadiusApproach,
    routePresets.kortzCenter,
    routePresets.delPerroPier
];

type InnerCityDrivingSceneOptions = {
    weatherType?: WeatherType;
    persistentWeather?: boolean;
    time?: {
        hour?: number;
        minute?: number;
        second?: number;
    };
    vehicle?: Partial<Ego["vehicle"]>;
};

type InnerCityDrivingTime = SceneType["environment"]["Time"];
type InnerCityDrivingVehicle = Ego["vehicle"];
type InnerCityDrivingSceneVariant = "default" | "duskRush" | "rainyCommute" | "lateNightCruise";

const defaultInnerCityDrivingWeatherType: WeatherType = WeatherType.CLEARING;
const defaultInnerCityDrivingPersistentWeather = true;

const defaultInnerCityDrivingTime: InnerCityDrivingTime = {
    hour: 16,
    minute: 30,
    second: 0
};

const defaultInnerCityDrivingVehicle: InnerCityDrivingVehicle = {
    id: 0,
    model: VehicleModel.Random,
    color: VehicleColor.Gray,
    maxSpeed: 14,
    drivingStyle: DrivingStyle.Cautious
};

export function createInnerCityDrivingScene(options: InnerCityDrivingSceneOptions = {}): SceneType {
    const time: InnerCityDrivingTime = {
        hour: options.time?.hour ?? defaultInnerCityDrivingTime.hour,
        minute: options.time?.minute ?? defaultInnerCityDrivingTime.minute,
        second: options.time?.second ?? defaultInnerCityDrivingTime.second
    };
    const vehicle: InnerCityDrivingVehicle = {
        id: options.vehicle?.id ?? defaultInnerCityDrivingVehicle.id,
        model: options.vehicle?.model ?? defaultInnerCityDrivingVehicle.model,
        color: options.vehicle?.color ?? defaultInnerCityDrivingVehicle.color,
        maxSpeed: options.vehicle?.maxSpeed ?? defaultInnerCityDrivingVehicle.maxSpeed,
        drivingStyle: options.vehicle?.drivingStyle ?? defaultInnerCityDrivingVehicle.drivingStyle,
        VehicleData: options.vehicle?.VehicleData ?? defaultInnerCityDrivingVehicle.VehicleData
    };

    return {
        environment: {
            weatherType: {
                type: options.weatherType ?? defaultInnerCityDrivingWeatherType,
                persistent: options.persistentWeather ?? defaultInnerCityDrivingPersistentWeather
            },
            Time: time
        },
        ego: {
            vehicle,
            waypoints: [...innerCityDrivingWaypoints]
        }
    };
}

export const innerCityDrivingScene: SceneType = createInnerCityDrivingScene();

export const innerCityDrivingScenes = {
    default: innerCityDrivingScene,
    duskRush: createInnerCityDrivingScene({
        weatherType: WeatherType.CLOUDS,
        time: {
            hour: 18,
            minute: 15
        },
        vehicle: {
            maxSpeed: 16,
            drivingStyle: DrivingStyle.Normal
        }
    }),
    rainyCommute: createInnerCityDrivingScene({
        weatherType: WeatherType.RAIN,
        time: {
            hour: 8,
            minute: 45
        },
        vehicle: {
            maxSpeed: 11,
            drivingStyle: DrivingStyle.Cautious
        }
    }),
    lateNightCruise: createInnerCityDrivingScene({
        weatherType: WeatherType.CLEAR,
        time: {
            hour: 23,
            minute: 30
        },
        vehicle: {
            model: VehicleModel.Sultan,
            color: VehicleColor.Black,
            maxSpeed: 18,
            drivingStyle: DrivingStyle.Normal
        }
    })
};

export type {InnerCityDrivingSceneVariant};
