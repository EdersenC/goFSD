import {isValidEntity, log, ensureModelLoaded, wait, AbortControllerCompat, Evaluator} from "./helper"
import {configurePopulation} from "./environment";
import {syncFlash} from "./sceneManger";


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
    VehicleData?: VehicleData[]
}

export interface Route {
    destination: [number, number, number]
}

export interface Ego {
    vehicle: Vehicle
    waypoints?: Route[]
}

export interface VehicleData {
    time: number
    currentSpeed: number
    drivingStyle: DrivingStyle
    acceleration: number
    isStopped: boolean | number
    Steering: number
    yaw: number
}

export interface WaypointCompleted {
    runId: string
    sceneId: string
    sceneVariant: string
    tripIndex: number
    chunkIndex: number
    chunkDurationMs: number
    isTripComplete: boolean
    syncTime: number
    endTime: number
    fromDestination: [number, number, number]
    toDestination: [number, number, number]
    vehicle: Omit<Vehicle, "id">
    vehicleData: VehicleData[]
}

export const SceneStoppedErrorCode = "SCENE_STOPPED";

export class EgoService {
    private static readonly MAX_TRIP_VEHICLE_DATA_POINTS = 1000;

    public oldEgo: Ego | null = null;
    public SceneName: string =''
    public RunId: string = ''
    private tripChunkIndex = 0;
    private activeTripContext: {
        sceneId: string
        sceneVariant: string
        tripIndex: number
        syncTime: number
        chunkStartTime: number
        fromDestination: [number, number, number]
        toDestination: [number, number, number]
    } | null = null;
    private stopRequested = false;
    private stopReason = "";
    private activeAbortController: AbortControllerCompat | null = null;

    public async execute(ego: Ego, sceneName: string, runId?: string) {
        ego.vehicle.VehicleData = []
        this.SceneName = sceneName
        this.RunId = runId && runId.trim() !== "" ? runId : this.createRunId()
        this.stopRequested = false;
        this.stopReason = "";
        this.tripChunkIndex = 0;
        this.activeTripContext = null;
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

    public requestStop(reason = "scene stop requested") {
        this.stopRequested = true;
        this.stopReason = reason;

        if (this.activeAbortController && !(this.activeAbortController as any).signal?.aborted) {
            this.activeAbortController.abort();
        }
    }


    private async watchDog(evaluator: () => Promise<{ success: boolean; message: string }>, timeoutMs = 60000): Promise<AbortSignal> {
        const abortController = new (AbortControllerCompat as any)();
        this.activeAbortController = abortController;
        const startTime = Date.now();

        const interval = setInterval(async () => {
            if (this.stopRequested) {
                console.log(`Watchdog stopping due to request: ${this.stopReason}`);
                abortController.abort();
                return;
            }

            if (Date.now() - startTime > timeoutMs) {
                console.log("Watchdog timeout reached, aborting.");
                abortController.abort();
                return;
            }

            const evalResult = await evaluator();
            if (evalResult.success) {
                console.log(evalResult.message);
                abortController.abort();
                return;
            }
        }, 50);

        abortController.signal.addEventListener("abort", () => {
            clearInterval(interval);
            if (this.activeAbortController === abortController) {
                this.activeAbortController = null;
            }
        }, { once: true });
        return abortController.signal;
    }

    private async routeEgo(ego: Ego) {
        if (!ego.waypoints || ego.waypoints.length === 0){
            console.log('No waypoints defined for ego, skipping routing.');
            return
        }
        const sceneInfo = this.parseSceneName(this.SceneName);
        let previousDestination = GetEntityCoords(ego.vehicle.id, false) as [number, number, number];
        for (const [tripIndex, waypoint] of ego.waypoints.entries()) {
            if (this.stopRequested) {
                throw new Error(SceneStoppedErrorCode);
            }

            this.tripChunkIndex = 0;
            const syncTime = await syncFlash();
            console.log('Routing to waypoint:', waypoint.destination);
            this.activeTripContext = {
                sceneId: sceneInfo.sceneId,
                sceneVariant: sceneInfo.sceneVariant,
                tripIndex,
                syncTime,
                chunkStartTime: syncTime,
                fromDestination: previousDestination,
                toDestination: waypoint.destination
            };
            const [x, y, z] = this.configureRoute(waypoint, ego.vehicle.maxSpeed, ego.vehicle.drivingStyle).destination;
            SetNewWaypoint(x,y);
            await wait(1000)
            const blip = GetFirstBlipInfoId(8)
            this.driveToWaypoint(PlayerPedId(), ego.vehicle.id, blip, ego.vehicle.drivingStyle, ego.vehicle.maxSpeed);
            const signal = await this.watchDog( await this.drivingLoop(ego,[x,y,z]), 10*60_000);
            // Wait for the watchdog to abort before continuing
            await new Promise<void>((resolve) => {
                signal.addEventListener("abort", () => resolve());
            });

            if (this.stopRequested) {
                this.emitTripChunk(
                    ego,
                    GetGameTimer(),
                    true
                );
                this.activeTripContext = null;
                ClearGpsPlayerWaypoint()
                throw new Error(SceneStoppedErrorCode);
            }

            this.emitTripChunk(
                ego,
                GetGameTimer(),
                true
            );
            previousDestination = waypoint.destination
            this.activeTripContext = null;
        }

        if (this.stopRequested) {
            ClearGpsPlayerWaypoint()
            throw new Error(SceneStoppedErrorCode);
        }
        TaskVehicleDriveWander(PlayerPedId(), ego.vehicle.id, ego.vehicle.maxSpeed, ego.vehicle.drivingStyle);
    }

    private emitTripChunk( ego: Ego, endTime: number, isTripComplete: boolean )
    {
        if (!this.activeTripContext){
         return;
        }

        const vehicleDataChunk = ego.vehicle.VehicleData ?? [];
        const payload: WaypointCompleted = {
            runId: this.RunId,
            sceneId: this.activeTripContext.sceneId,
            sceneVariant: this.activeTripContext.sceneVariant,
            tripIndex: this.activeTripContext.tripIndex,
            chunkIndex: this.tripChunkIndex,
            chunkDurationMs: Math.max(0, endTime - this.activeTripContext.chunkStartTime),
            isTripComplete,
            syncTime: this.activeTripContext.syncTime,
            endTime,
            fromDestination: this.activeTripContext.fromDestination,
            toDestination: this.activeTripContext.toDestination,
            vehicle: {
                model: ego.vehicle.model,
                color: ego.vehicle.color,
                maxSpeed: ego.vehicle.maxSpeed,
                drivingStyle: ego.vehicle.drivingStyle,
                VehicleData: undefined,
            },
            vehicleData: vehicleDataChunk,
        };

        console.log(
            `[ego] emitting chunk ${payload.chunkIndex} for trip ${payload.tripIndex} run ${this.RunId} with ${payload.vehicleData.length} data points complete=${isTripComplete}`
        );

        emitNet("ego:vehicleData", payload);
        ego.vehicle.VehicleData = [];
        this.tripChunkIndex += 1;
        this.activeTripContext.chunkStartTime = endTime;
    }

    private parseSceneName(sceneName: string): { sceneId: string; sceneVariant: string } {
        const [sceneId, sceneVariant] = sceneName.split(":");
        return {
            sceneId: sceneId || "unknown-scene",
            sceneVariant: sceneVariant || "default"
        };
    }

    private createRunId(): string {
        const now = new Date();
        const year = now.getFullYear();
        const month = this.pad2(now.getMonth() + 1);
        const day = this.pad2(now.getDate());
        const hours24 = now.getHours();
        const meridiem = hours24 >= 12 ? "PM" : "AM";
        const hours12 = hours24 % 12 || 12;
        const minutes = this.pad2(now.getMinutes());
        const seconds = this.pad2(now.getSeconds());
        const suffix = Math.random().toString(36).slice(2, 8);
        return `${year}-${month}-${day}_${this.pad2(hours12)}-${minutes}-${seconds}${meridiem}_${suffix}`;
    }

    private pad2(value: number): string {
        return String(value).padStart(2, "0");
    }



    private async drivingLoop(ego:Ego, cords:[number,number,number]): Promise<() => Promise<{
        success: boolean;
        message: string
    }>> {
        let lastTargetSpeed = ego.vehicle.maxSpeed;
        const highwaySpeed = 65;
        const [x, y, z] = cords;
        return async () => {
            const targetSpeed = GetIsPlayerDrivingOnHighway(PlayerId()) ? highwaySpeed : ego.vehicle.maxSpeed;
            if (targetSpeed !== lastTargetSpeed) {
                if (targetSpeed === highwaySpeed) {
                    console.log(`Player is on highway, increasing speed.`);
                }
                SetDriveTaskCruiseSpeed(PlayerPedId(), targetSpeed);
                lastTargetSpeed = targetSpeed;
            }


            this.collectEgoData(ego);
            const isWaypointActive = IsWaypointActive();
            return {
                success: !isWaypointActive,
                message: `Waypoint reached at (${x.toFixed(2)}, ${y.toFixed(2)}, ${z.toFixed(2)}).`
            }
        }
    }



    public collectEgoData(ego: Ego) {
        const id = ego.vehicle.id;
        const currentSpeed = GetEntitySpeed(ego.vehicle.id);
        const isStopped :boolean | number = IsVehicleStopped(id)
        const acceleration = GetVehicleCurrentAcceleration(id);
        const Steering = GetVehicleWheelSteeringAngle(id, 0);
        const yaw = GetEntityHeading(id)
        const time = GetGameTimer();

        if (ego.vehicle.VehicleData) {
            const lastData = ego.vehicle.VehicleData[ego.vehicle.VehicleData.length - 1];
            if (lastData?.isStopped && isStopped) {
                return
            }
        }



        const data: VehicleData = {
            time,
            currentSpeed,
            drivingStyle: ego.vehicle.drivingStyle,
            acceleration,
            isStopped,
            Steering,
            yaw,
        };


        if (!ego.vehicle.VehicleData) {
            ego.vehicle.VehicleData = [];
        }
        ego.vehicle.VehicleData.push(data)

        if (ego.vehicle.VehicleData.length >= EgoService.MAX_TRIP_VEHICLE_DATA_POINTS) {
            this.emitTripChunk(ego, GetGameTimer(), false);
        }

        console.log('Collected ego data:', data);
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
        ego.vehicle.model = resolvedModel;
        ego.vehicle.color = resolvedColor;
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

    private cleanUp(ego: Ego) {
        if (!isValidEntity(ego.vehicle.id)) {
            console.log('No valid vehicle to clean up for ego:', ego);
        }

        SetEntityAsMissionEntity(ego.vehicle.id, true, true);
        wait(1000)
        DeleteVehicle(ego.vehicle.id);
        console.log('Cleaned up old vehicle with ID:', ego.vehicle.id);
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
