import {isValidEntity, log, ensureModelLoaded, wait, AbortControllerCompat, Evaluator} from "./helper"


export enum DrivingFlag {
    // Stopping/Collision avoidance
    StopForCars                             = 1,
    StopForPeds                             = 2,
    SwerveAroundAllCars                     = 4,
    SteerAroundStationaryCars               = 8,
    SteerAroundPeds                         = 16,
    SteerAroundObjects                      = 32,
    DontSteerAroundPlayerPed                = 64,
    StopAtLights                            = 128,

    // Movement behavior
    GoOffRoadWhenAvoiding                   = 256,
    DriveIntoOncomingTraffic                = 512,
    DriveInReverse                          = 1024,
    UseWanderFallbackInsteadOfStraightLine  = 2048,
    AvoidRestrictedAreas                    = 4096,

    // Pathfinding and navigation
    PreventBackgroundPathfinding            = 8192,  // Only works on MISSION_CRUISE
    AdjustCruiseSpeedBasedOnRoadSpeed       = 16384,
    UseShortCutLinks                        = 262144,
    ChangeLanesAroundObstructions           = 524288,
    UseSwitchedOffNodes                     = 2097152, // Cruise tasks ignore this, only for goto's
    PreferNavmeshRoute                      = 4194304, // If primarily driving off-road
    ForceStraightLine                       = 16777216,
    UseStringPullingAtJunctions             = 33554432,

    // Plane and highway behavior
    PlaneTaxiMode                           = 8388608, // Only works for planes with MISSION_GOTO
    AvoidHighways                           = 536870912,
    ForceJoinInRoadDirection                = 1073741824
}

export enum DrivingStyle {
    // Careful, predictable driving with all safety measures
    Normal =
        DrivingFlag.StopForCars |
        DrivingFlag.StopForPeds |
        DrivingFlag.SteerAroundStationaryCars |
        DrivingFlag.SteerAroundObjects |
        DrivingFlag.SteerAroundPeds |
        DrivingFlag.StopAtLights |
        DrivingFlag.ChangeLanesAroundObstructions,

    // Fast but still cautious, ignores some traffic
    Rush =
        DrivingFlag.SwerveAroundAllCars |
        DrivingFlag.SteerAroundPeds |
        DrivingFlag.StopAtLights |
        DrivingFlag.DriveIntoOncomingTraffic |
        DrivingFlag.ChangeLanesAroundObstructions |
        DrivingFlag.AdjustCruiseSpeedBasedOnRoadSpeed,

    // Very defensive driving, avoids everything
    Cautious =
        DrivingFlag.StopForCars |
        DrivingFlag.StopForPeds |
        DrivingFlag.SteerAroundStationaryCars |
        DrivingFlag.SteerAroundPeds |
        DrivingFlag.SteerAroundObjects |
        DrivingFlag.StopAtLights |
        DrivingFlag.ChangeLanesAroundObstructions|
        DrivingFlag.UseWanderFallbackInsteadOfStraightLine,

    // Aggressive but not reckless, ignores some rules
    Aggressive =
        DrivingFlag.SwerveAroundAllCars |
        DrivingFlag.SteerAroundPeds |
        DrivingFlag.DriveIntoOncomingTraffic |
        DrivingFlag.ChangeLanesAroundObstructions |
        DrivingFlag.DontSteerAroundPlayerPed,

    // Complete disregard for rules and safety
    Reckless =
        DrivingFlag.SwerveAroundAllCars |
        DrivingFlag.DriveIntoOncomingTraffic |
        DrivingFlag.GoOffRoadWhenAvoiding |
        DrivingFlag.DontSteerAroundPlayerPed|
        DrivingFlag.SteerAroundObjects
}

export enum VehicleColor {
    Random = -1,
    // Neutral colors
    Black = 12,
    Gray = 13,
    LightGray = 14,
    IceWhite = 131,

    // Blue colors
    Blue = 83,
    DarkBlue = 82,
    MidnightBlue = 84,

    // Purple colors
    MidnightPurple = 149,
    SchafterPurple = 148,

    // Red colors
    Red = 39,
    DarkRed = 40,

    // Orange/Yellow colors
    Orange = 41,
    Yellow = 42,

    // Green colors
    LimeGreen = 55,
    Green = 128,
    ForestGreen = 151,
    FoliageGreen = 155,
    OliveDarb = 152,

    // Earth tones
    DarkEarth = 153,
    DesertTan = 154
}

export enum VehicleModel {
    Random = "random",
    Adder = "adder",
    Zentorno = "zentorno",
    T20 = "t20",
    Osiris = "osiris",
    TurismoR = "turismor",
    EntityXF = "entityxf",
    Cheetah = "cheetah",
    Infernus = "infernus",
    Comet2 = "comet2",
    NineF = "ninef",
    Banshee = "banshee",
    Buffalo = "buffalo",
    Sultan = "sultan",
    Elegy2 = "elegy2",
    F620 = "f620",
    Dominator = "dominator",
    Gauntlet = "gauntlet",
    Futo = "futo",
    Dubsta = "dubsta",
    Baller2 = "baller2"
}


interface Vehicle {
    id: number
    model: VehicleModel | string
    color: VehicleColor
    maxSpeed: number
    drivingStyle: DrivingStyle
}

export interface Route {
    destination: [number, number, number]
}

export interface Ego {
    vehicle: Vehicle
    waypoints?: Route[]
}



export class EgoService {

    public oldEgo: Ego | null = null;


    public async execute(ego: Ego){
        if (this.oldEgo) {
            this.cleanUp(this.oldEgo);
        }
        this.oldEgo = ego;
        await this.setVehicle(ego);
        SetPlayerInvincible(PlayerId(), true);
        SetVehRadioStation(ego.vehicle.id, "OFF");
        this.makeVehicleGodMode(ego.vehicle.id)
        this.setCaptureCamera(ego);
        this.makePlayerUnaware(PlayerPedId(),true)
        await this.routeEgo(ego)
        return
    }



    private watchDog(evaluator:()=>Evaluator, timeoutMs = 60000): AbortSignal {
        const abortController = new (AbortControllerCompat as any)();
        const startTime = Date.now();

        const interval = setInterval(() => {
            if (Date.now() - startTime > timeoutMs) {
                console.log("Watchdog timeout reached, aborting.");
                abortController.abort();
                return;
            }

            const evalResult = evaluator();
            if (evalResult.success) {
                console.log(evalResult.message);
                abortController.abort();
                return;
            }
        }, 1000);

        abortController.signal.addEventListener("abort", () => clearInterval(interval), { once: true });
        return abortController.signal;
    }

    private async routeEgo(ego: Ego) {
        if (!ego.waypoints || ego.waypoints.length === 0){
            console.log('No waypoints defined for ego, skipping routing.');
            return
        }
        for (const waypoint of ego.waypoints) {
            console.log('Routing to waypoint:', waypoint.destination);
            const [x, y, z] = this.configureRoute(waypoint, ego.vehicle.maxSpeed, ego.vehicle.drivingStyle).destination;
            SetNewWaypoint(x,y);
            await wait(1000)
            const blip = GetFirstBlipInfoId(8)
            this.driveToWaypoint(PlayerPedId(), ego.vehicle.id, blip, ego.vehicle.drivingStyle, ego.vehicle.maxSpeed);
            const signal = this.watchDog( this.drivingLoop(ego,[x,y,z]), 10*60_000);
            // Wait for the watchdog to abort before continuing
            await new Promise<void>((resolve) => {
                signal.addEventListener("abort", () => resolve());
            });
        }
        TaskVehicleDriveWander(PlayerPedId(), ego.vehicle.id, ego.vehicle.maxSpeed, ego.vehicle.drivingStyle);
    }


    private drivingLoop(ego:Ego, cords:[number,number,number]): () => Evaluator {
        let lastTargetSpeed = ego.vehicle.maxSpeed;
        const highwaySpeed = 65;
        const [x, y, z] = cords;
        return () => {
            const targetSpeed = GetIsPlayerDrivingOnHighway(PlayerId()) ? highwaySpeed : ego.vehicle.maxSpeed;
            if (targetSpeed !== lastTargetSpeed) {
                if (targetSpeed === highwaySpeed) {
                    console.log(`Player is on highway, increasing speed.`);
                }
                SetDriveTaskCruiseSpeed(PlayerPedId(), targetSpeed);
                lastTargetSpeed = targetSpeed;
            }
            const isWaypointActive = IsWaypointActive();
            console.log(' way point is  ',isWaypointActive)
            return {
                success: !isWaypointActive,
                message: `Waypoint reached at (${x.toFixed(2)}, ${y.toFixed(2)}, ${z.toFixed(2)}).`
            }
        }
    }




    public makePlayerUnaware(egoId:number, toggle: boolean) {
        SetMaxWantedLevel(0);
        SetEveryoneIgnorePlayer(PlayerId(),toggle)
        SetPoliceIgnorePlayer(PlayerId(),toggle)
        SetPlayerCanBeHassledByGangs(PlayerId(), !toggle);
        SetPedCanBeTargetted(egoId, !toggle);
        SetPedKeepTask(egoId,toggle);
        SetBlockingOfNonTemporaryEvents(egoId, toggle);
    }

    public setCaptureCamera(ego: Ego) {
        setInterval(() => {
            if (!isValidEntity(ego.vehicle.id)) return;

            if (!IsPedInVehicle(PlayerPedId(), ego.vehicle.id, false)) return;

            // Keep a stable chase view and prevent free-look while lock is enabled.
            SetCinematicButtonActive(false);
            SetCinematicModeActive(false);
            StopCinematicCamShaking(true);
            InvalidateIdleCam();
            InvalidateVehicleIdleCam();
            SetFollowVehicleCamViewMode(1);
            SetGameplayCamRelativeHeading(0.0);
            SetGameplayCamRelativePitch(-6.0, 1.0);

            // Block look/camera/cinematic controls.
            DisableControlAction(0, 1, true);
            DisableControlAction(0, 2, true);
            DisableControlAction(0, 0, true);
            DisableControlAction(0, 26, true);
            DisableControlAction(0, 80, true);
            DisableControlAction(0, 81, true);
            DisableControlAction(0, 95, true);
            DisableControlAction(0, 99, true);
        }, 0);

    }

    public async setVehicle(ego: Ego){
        const resolvedModel = this.resolveVehicleModel(ego.vehicle.model);
        const resolvedColor = this.resolveVehicleColor(ego.vehicle.color);

        const vehicleModel = GetHashKey(resolvedModel);
        const vehicleOk = await ensureModelLoaded(vehicleModel);
        if (!vehicleOk) {
            console.log('Failed to load vehicle or driver model.');
            return;
        }

        const newVehicle = this.spawnVehicle(PlayerPedId(),vehicleModel, resolvedColor);
        if (!isValidEntity(newVehicle)) {
            SetModelAsNoLongerNeeded(vehicleModel);
            console.log(`Failed to spawn vehicle model: ${resolvedModel}`);
            return;
        }
        const incontrol = NetworkRequestControlOfEntity(newVehicle);
        console.log(`Requested control of vehicle ${newVehicle}, in control: ${incontrol}`);
        SetVehicleNumberPlateText(newVehicle, "EGO");
        SetVehicleOnGroundProperly(newVehicle);
        this.addPedToVehicle(PlayerPedId(), newVehicle);
        ego.vehicle.id = newVehicle;
    }

    public makeVehicleGodMode(veh: number) {
        SetVehicleStrong(veh, true);
        SetEntityCanBeDamaged(veh, false);
        SetVehicleCanBeVisiblyDamaged(veh, false)
        SetVehicleDoorsLocked(veh,10)
        SetEntityProofs(veh, true, true, true, true, true, true, true, true);
        SetVehicleTyresCanBurst(veh, false);
        SetVehicleDeformationFixed(veh);
        SetVehicleDirtLevel(veh, 0.0);
        WashDecalsFromVehicle(veh, 1.0);
    }

    public spawnVehicle(egoId:number, model: number, color: VehicleColor){
        const [px, py, pz] = GetEntityCoords(egoId, false) as unknown as [number, number, number];
        const heading = GetEntityHeading(egoId);
        const forward = GetEntityForwardVector(egoId) as unknown as [number, number, number];
        const spawnX = px + (Number(forward[0]) || 0) * 10.0;
        const spawnY = py + (Number(forward[1]) || 0) * 10.0;
        const spawnZ = pz + 0.5;
        console.log(`Spawing Vehicle at ${px}, ${py}, ${pz} with heading ${heading} and forward vector ${forward}`);
        const vehicle = CreateVehicle(model,spawnX, spawnY, spawnZ, heading, true, false);
        SetVehicleColours(vehicle, color, color);
        return vehicle;
    }

    public addPedToVehicle(egoId:number, vehicle: number, ) {
        SetPedIntoVehicle(egoId, vehicle, -1)
        console.log('Adding player ped to vehicle:', vehicle);
    }

    public driveToWaypoint(driver: number, vehicle: number, waypoint:number ,drivingStyle:DrivingStyle, speed: number = 20.0): void {
        if (DoesBlipExist(waypoint)) {
            const [x, y, z] = GetBlipInfoIdCoord(waypoint) as unknown as [number, number, number];
            TaskVehicleDriveToCoordLongrange(driver, vehicle, x, y, z + 1.0, speed, drivingStyle, 8.0);
                log(`NPC driving to waypoint at (${x.toFixed(2)}, ${y.toFixed(2)}, ${z.toFixed(2)}) with speed ${speed} and driving style ${drivingStyle}.`);
        } else {
            log("No waypoint found for driving.");
        }
    }

    private configureRoute(route:Route , speed: number, drivingStyle: DrivingStyle):Route {
        const [x, y, z] =route.destination
        if (IsPointOnRoad(x, y, z, -1)) return route;
        const closestRoad = GetClosestRoad(x, y, z, 1.0, 1, false);
        if (!closestRoad){
            console.log(`No road found near (${x}, ${y}, ${z}), using original destination.`);
            return route;
        }
        const [rx, ry, rz] = closestRoad[1] as unknown as [number, number, number];
        console.log(`Adjusted route from (${x}, ${y}, ${z}) to closest road at (${rx}, ${ry}, ${rz}).`);
        return {destination: [rx, ry, rz]};
    }


    public incrementSpeed(vehicle: number, increment: number = 5.0): void {
        const currentSpeed = GetEntitySpeed(vehicle);
        const newSpeed = currentSpeed + increment;
        SetVehicleForwardSpeed(vehicle, newSpeed);
    }

    private cleanUp(ego: Ego) {
        if (!isValidEntity(ego.vehicle.id)) {
            console.log('No valid vehicle to clean up for ego:', ego);
        }

        SetEntityAsMissionEntity(ego.vehicle.id, true, true);
        wait(1000)
        DeleteVehicle(ego.vehicle.id);
        console.log('Cleaned up old vehicle with ID:', ego.vehicle.id);
    }


     public updateVehicle(vehicle: Vehicle){
    }

     public removeVehicle(){
    }

    private resolveVehicleColor(color: VehicleColor): VehicleColor {
        if (color !== VehicleColor.Random) {
            return color;
        }
        const values = Object.values(VehicleColor).filter((value) => typeof value === "number") as number[];
        const options = values.filter((value) => value !== VehicleColor.Random);
        const pick = options[Math.floor(Math.random() * options.length)];
        return pick as VehicleColor;
    }

    private resolveVehicleModel(model: VehicleModel | string): string {
        if (typeof model === "string" && model.toLowerCase() === VehicleModel.Random) {
            const values = Object.values(VehicleModel).filter((value) => value !== VehicleModel.Random) as string[];
            // @ts-ignore
            return values[Math.floor(Math.random() * values.length)];
        }
        return model;
    }
}
