import {Route} from "../egoService";

export const routePresets = {
    legionSquare: {destination: [213.8, -856.7, 30.8]},
    pillboxHillGarage: {destination: [79.2, -744.7, 44.3]},
    pillboxMedicalCenter: {destination: [307.2, -595.1, 43.3]},
    missionRowPoliceStation: {destination: [428.9, -984.5, 30.7]},
    textileCityMarket: {destination: [427.1, -806.2, 29.5]},
    strawberryCarWash: {destination: [24.5, -1391.8, 29.3]},
    strawberryLTDGasoline: {destination: [-47.4, -1760.7, 29.4]},
    ranchoLTDGasoline: {destination: [819.8, -1029.6, 26.4]},
    littleSeoulGasStation: {destination: [-715.2, -935.0, 19.2]},
    littleSeoulArcadiusApproach: {destination: [-758.9, -586.8, 30.3]},
    portolaDriveParking: {destination: [-731.6, -227.3, 37.1]},
    rockfordPlaza: {destination: [-679.5, -884.7, 24.5]},
    weazelPlazaGarage: {destination: [-903.1, -451.2, 39.6]},
    richardsMajesticStudio: {destination: [-1074.6, -503.1, 36.0]},
    backlotCityStudioGate: {destination: [-1037.2, -474.1, 36.8]},
    delPerroParkingGarage: {destination: [-1456.8, -497.8, 32.8]},
    delPerroPier: {destination: [-1627.270, -983.829, 12.645]},
    vespucciCanals: {destination: [-1183.5, -1074.4, 2.2]},
    kortzCenter: {destination: [-2243.8, 270.8, 174.6]},
    orientalTheater: {destination: [296.5, 185.4, 104.3]},
    littleSeoulParking: {destination: [-464.828, -777.476, 34.859]},
    rockfordHillParking: {destination: [-777.282, 373.133, 87.353]},
} satisfies Record<string, Route>;

export type RoutePresetId = keyof typeof routePresets;
