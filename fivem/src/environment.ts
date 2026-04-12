//src/environment.ts
import {log} from "./helper"
import {Ego} from "./egoService";

export enum WeatherType {
    CLEAR = "CLEAR",
    EXTRA_SUNNY = "EXTRASUNNY",
    CLOUDS = "CLOUDS",
    OVERCAST = "OVERCAST",
    RAIN = "RAIN",
    CLEARING = "CLEARING",
    THUNDER = "THUNDER",
    SMOG = "SMOG",
    FOGGY = "FOGGY",
    XMAS = "XMAS",
    SNOW = "SNOW",
    SNOWLIGHT = "SNOWLIGHT",
    BLIZZARD = "BLIZZARD",
    HALLOWEEN = "HALLOWEEN",
    NEUTRAL = "NEUTRAL",
    RAIN_HALLOWEEN = "RAIN_HALLOWEEN",
    SNOW_HALLOWEEN = "SNOW_HALLOWEEN"
}



export interface Weather{
    type: WeatherType
    persistent: boolean
}

export interface Time{
    hour: number
    minute: number
    second: number
    persistent: boolean
}

export interface Environment {
    weatherType: Weather
    Time:Time
}

export class EnvironmentService {


    public execute(environment: Environment) {
        this.SetWeather(environment.weatherType);
        this.setTime(environment.Time);
        configurePopulation(undefined)
        log("Environment executed with weather: " + environment.weatherType.type + " and time: " + environment.Time.hour + ":" + environment.Time.minute + ":" + environment.Time.second);
    }


     public SetWeather(weather:Weather){
        if (weather.persistent) {
            SetWeatherTypeNowPersist(weather.type);
        } else {
            SetWeatherTypeNow(weather.type);
        }
        log("Weather set to " + weather.type);
    }

    public setTime(time: Time) {
        const { hour, minute, second } = time;
        NetworkOverrideClockTime(hour, minute, second);
        PauseClock(time.persistent)
        log(`Time set to ${hour}:${minute}:${second}`);
    }

     public get Time(): Time {
        const hour = GetClockHours();
        const minute = GetClockMinutes();
        const second = GetClockSeconds();
        return { hour, minute, second };
     }

}


export function configurePopulation(environment: Environment|undefined) {
    SetPedPopulationBudget(3);
    SetPedDensityMultiplierThisFrame(1.0)
    // SetVehiclePopulationBudget(0);
    const [x, y, z] = GetEntityCoords(PlayerPedId(), true) as [number, number, number];
    // ClearAreaOfVehicles(x, y, z, 1000.0, false, false, false, false, false);
    // SetRandomVehicleDensityMultiplierThisFrame(0.0);
    SetParkedVehicleDensityMultiplierThisFrame(3);
}


