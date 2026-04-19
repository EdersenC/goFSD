import {Environment, Time, WeatherType} from "./environment";
import {VehicleColor, VehicleModel} from "./egoService";

export type TripProfileSnapshot = {
    seed: string
    weatherType: WeatherType
    time: Time
    timeBucket: string
    vehicleModel: string
    vehicleColor: VehicleColor
    vehicleColorName: string
};

type VehicleColorOption = {
    name: string
    value: VehicleColor
};

type TimeOption = {
    label: string
    hour: number
    minute: number
    second: number
};

const WEATHER_OPTIONS: WeatherType[] = [
    WeatherType.CLEAR,
    WeatherType.EXTRA_SUNNY,
    WeatherType.CLOUDS,
    WeatherType.OVERCAST,
    WeatherType.CLEARING,
    WeatherType.RAIN,
    WeatherType.FOGGY,
    WeatherType.SMOG
];

const TIME_OPTIONS: TimeOption[] = [
    {label: "pre_dawn", hour: 5, minute: 45, second: 0},
    {label: "morning_commute", hour: 8, minute: 30, second: 0},
    {label: "midday", hour: 12, minute: 15, second: 0},
    {label: "afternoon", hour: 15, minute: 45, second: 0},
    {label: "evening_rush", hour: 18, minute: 30, second: 0},
    {label: "night", hour: 21, minute: 15, second: 0},
    {label: "late_night", hour: 23, minute: 45, second: 0}
];

const VEHICLE_MODEL_OPTIONS: string[] = [
    VehicleModel.Adder,
    VehicleModel.Zentorno,
    VehicleModel.T20,
    VehicleModel.Osiris,
    VehicleModel.TurismoR,
    VehicleModel.EntityXF,
    VehicleModel.Cheetah,
    VehicleModel.Infernus,
    VehicleModel.Comet2,
    VehicleModel.NineF,
    VehicleModel.Banshee,
    VehicleModel.Buffalo,
    VehicleModel.Sultan,
    VehicleModel.Elegy2,
    VehicleModel.F620,
    VehicleModel.Dominator,
    VehicleModel.Gauntlet,
    VehicleModel.Futo,
    VehicleModel.Dubsta,
    VehicleModel.Baller2
];

const VEHICLE_COLOR_OPTIONS: VehicleColorOption[] = [
    {name: "Black", value: VehicleColor.Black},
    {name: "Gray", value: VehicleColor.Gray},
    {name: "LightGray", value: VehicleColor.LightGray},
    {name: "IceWhite", value: VehicleColor.IceWhite},
    {name: "Blue", value: VehicleColor.Blue},
    {name: "DarkBlue", value: VehicleColor.DarkBlue},
    {name: "MidnightBlue", value: VehicleColor.MidnightBlue},
    {name: "MidnightPurple", value: VehicleColor.MidnightPurple},
    {name: "SchafterPurple", value: VehicleColor.SchafterPurple},
    {name: "Red", value: VehicleColor.Red},
    {name: "DarkRed", value: VehicleColor.DarkRed},
    {name: "Orange", value: VehicleColor.Orange},
    {name: "Yellow", value: VehicleColor.Yellow},
    {name: "LimeGreen", value: VehicleColor.LimeGreen},
    {name: "Green", value: VehicleColor.Green},
    {name: "ForestGreen", value: VehicleColor.ForestGreen},
    {name: "FoliageGreen", value: VehicleColor.FoliageGreen},
    {name: "OliveDarb", value: VehicleColor.OliveDarb},
    {name: "DarkEarth", value: VehicleColor.DarkEarth},
    {name: "DesertTan", value: VehicleColor.DesertTan}
];

class SeededRandom {
    private state: number;

    constructor(seed: string) {
        this.state = hashString(seed) || 0x6d2b79f5;
    }

    next(): number {
        this.state += 0x6d2b79f5;
        let value = this.state;
        value = Math.imul(value ^ (value >>> 15), value | 1);
        value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
        return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    }

    pick<T>(items: T[]): T {
        const index = Math.floor(this.next() * items.length);
        return items[Math.min(index, items.length - 1)];
    }
}

function hashString(value: string): number {
    let hash = 2166136261;
    for (let index = 0; index < value.length; index += 1) {
        hash ^= value.charCodeAt(index);
        hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
}

export function buildDeterministicTripProfile(
    runId: string,
    sceneId: string,
    sceneVariant: string,
    tripIndex: number,
    baseEnvironment?: Environment
): TripProfileSnapshot {
    const seed = `${runId}:${sceneId}:${sceneVariant}:${tripIndex}`;
    const rng = new SeededRandom(seed);
    const timeChoice = rng.pick(TIME_OPTIONS);
    const colorChoice = rng.pick(VEHICLE_COLOR_OPTIONS);

    return {
        seed,
        weatherType: rng.pick(WEATHER_OPTIONS),
        time: {
            hour: timeChoice.hour,
            minute: timeChoice.minute,
            second: timeChoice.second,
            persistent: baseEnvironment?.Time?.persistent ?? true
        },
        timeBucket: timeChoice.label,
        vehicleModel: rng.pick(VEHICLE_MODEL_OPTIONS),
        vehicleColor: colorChoice.value,
        vehicleColorName: colorChoice.name
    };
}
