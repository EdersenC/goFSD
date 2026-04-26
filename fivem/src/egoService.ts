import {isValidEntity, log, ensureModelLoaded, wait, AbortControllerCompat, Evaluator} from "./helper"
import {Environment, EnvironmentService} from "./environment";
import {syncFlash} from "./sceneManger";
import {requestCapture, requestTripFinalize, TripFinalizeResponse} from "./captureControl";
import {buildDeterministicTripProfile, TripProfileSnapshot} from "./tripProfiles";


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

type Vector3 = [number, number, number]

const VehicleNodeFlags = {
    OffRoad: 1 << 0,
    Highway: 1 << 6,
    Junction: 1 << 7,
    TrafficLight: 1 << 8,
} as const;

export interface VehicleData {
    time: number
    currentSpeed: number
    drivingStyle: DrivingStyle
    acceleration: number
    brakePressureAvg: number
    isStopped: boolean | number
    Steering: number
    yaw: number
    coords: [number, number, number]
    gps: [number, number, number]
    velocityWorldX?: number
    velocityWorldY?: number
    velocityWorldZ?: number
    velocityForward?: number
    velocityLateral?: number
    velocityVertical?: number
    yawRate?: number
    gear?: number
    rpm?: number
    engineHealth?: number
    bodyHealth?: number
    routeGpsValid?: boolean
    routeDistance?: number
    routeHeadingError?: number
    routeForwardDelta?: number
    routeLateralDelta?: number
    roadNodeDistance?: number
    roadNodeHeading?: number
    roadNodeDensity?: number
    roadLaneCountForward?: number
    roadLaneCountBackward?: number
    roadEdgeSpan?: number
    isOnRoad?: boolean
    isOffroadNode?: boolean
    isInJunction?: boolean
    hasTrafficLightNode?: boolean
    isHighway?: boolean
    nearbyVehicleCount30m?: number
    nearbyPedCount20m?: number
    hasLeadVehicle?: boolean
    leadVehicleDistance?: number
    leadVehicleRelativeSpeed?: number
    leadVehicleTTC?: number
    leadVehicleHeadingDelta?: number
    streetNameHash?: number
    crossingRoadHash?: number
    isStoppedAtTrafficLights?: boolean
    eventCollision?: boolean
    eventOffroad?: boolean
    eventWrongWay?: boolean
    eventReversing?: boolean
    eventHandbrake?: boolean
    timeSinceSyncMs?: number
    timeSinceChunkStartMs?: number
    dataPointIndex?: number
    chunkIndex?: number
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
    tripProfile: TripProfileSnapshot
    vehicle: Omit<Vehicle, "id">
    vehicleData: VehicleData[]
}

export const SceneStoppedErrorCode = "SCENE_STOPPED";
const EGO_BUILD_ID = "2026-04-21-capture-failfast-v1";
const HardFailureErrorCode = "HARD_TRIP_FAILURE";

export class EgoService {
    private static readonly MAX_TRIP_VEHICLE_DATA_POINTS = 200;
    private readonly environmentService = new EnvironmentService();

    public oldEgo: Ego | null = null;
    public SceneName: string =''
    public RunId: string = ''
    private baseEnvironment: Environment | null = null;
    private tripChunkIndex = 0;
    private tripDataPointIndex = 0;
    private activeTripContext: {
        sceneId: string
        sceneVariant: string
        tripIndex: number
        syncTime: number
        chunkStartTime: number
        fromDestination: [number, number, number]
        toDestination: [number, number, number]
        tripProfile: TripProfileSnapshot
    } | null = null;
    private stopRequested = false;
    private stopReason = "";
    private cameraTickId: number | null = null;
    // @ts-ignore
    private activeAbortController: AbortControllerCompat | null = null;

    public async execute(ego: Ego, sceneName: string, runId?: string, baseEnvironment?: Environment) {
        await this.executeEgo(ego, sceneName, runId, baseEnvironment);
        await this.routeEgo(ego)
        return
    }

    public async executeEgo(ego: Ego, sceneName: string, runId?: string, baseEnvironment?: Environment) {
        ego.vehicle.VehicleData = []
        this.SceneName = sceneName
        this.RunId = runId && runId.trim() !== "" ? runId : this.createRunId()
        this.baseEnvironment = baseEnvironment ? JSON.parse(JSON.stringify(baseEnvironment)) as Environment : null;
        this.stopRequested = false;
        this.stopReason = "";
        this.tripChunkIndex = 0;
        this.tripDataPointIndex = 0;
        this.activeTripContext = null;
        this.stopManagedLoops();
        if (this.oldEgo) {
            this.cleanUp(this.oldEgo);
        }
        this.oldEgo = ego;
        ClearPedTasksImmediately(PlayerPedId());
        await this.setVehicle(ego);
        if (!isValidEntity(ego.vehicle.id)) {
            throw new Error("failed to initialize ego vehicle");
        }
        SetPlayerInvincible(PlayerId(), true);
        SetVehRadioStation(ego.vehicle.id, "OFF");
        SetVehicleEngineOn(ego.vehicle.id, true, true, false);
        SetVehicleUndriveable(ego.vehicle.id, false);
        FreezeEntityPosition(ego.vehicle.id, false);
        this.makeVehicleGodMode(ego.vehicle.id)
        this.startCaptureCamera(ego);
        this.makePlayerUnaware(PlayerPedId(),true)
        console.log(`[ego-control] FiveM ego setup ready build=${EGO_BUILD_ID} chunk_points=${EgoService.MAX_TRIP_VEHICLE_DATA_POINTS}; vehicle actuation now comes from the Go virtual controller.`);
        return
    }

    private resetEgoControlRuntime() {
        this.stopManagedLoops();
    }

    private stopManagedLoops() {
        this.stopCaptureCamera();
    }

    public disposeCurrentEgo() {
        this.stopRequested = true;
        this.stopReason = "ego disposed";
        if (this.activeAbortController && !(this.activeAbortController as any).signal?.aborted) {
            this.activeAbortController.abort();
        }
        this.resetEgoControlRuntime();

        if (this.oldEgo) {
            this.cleanUp(this.oldEgo);
        }
        this.oldEgo = null;
        this.activeTripContext = null;
        this.tripChunkIndex = 0;
        this.tripDataPointIndex = 0;
        SetPlayerInvincible(PlayerId(), false);
        this.makePlayerUnaware(PlayerPedId(), false);
    }

    public currentSpeed(): number | null {
        if (!this.oldEgo || !isValidEntity(this.oldEgo.vehicle.id)) {
            return null;
        }
        return GetEntitySpeed(this.oldEgo.vehicle.id);
    }

    public currentYaw(): number | null {
        if (!this.oldEgo || !isValidEntity(this.oldEgo.vehicle.id)) {
            return null;
        }
        return GetEntityHeading(this.oldEgo.vehicle.id);
    }

    public currentRouteForwardDelta(): number | null {
        if (!this.oldEgo || !isValidEntity(this.oldEgo.vehicle.id)) {
            return null;
        }
        const id = this.oldEgo.vehicle.id;
        const coords = GetEntityCoords(id, true) as Vector3;
        const yaw = GetEntityHeading(id);
        const [gpsRouteFound, gpsRouteCoords] = GetPosAlongGpsTypeRoute(true, 1, 0);
        const gps = this.toVector3(gpsRouteCoords) ?? [0, 0, 0];
        const routeContext = this.collectRouteContext(id, coords, yaw, gpsRouteFound, gps);
        if (!routeContext.routeGpsValid || routeContext.routeForwardDelta === undefined) {
            return null;
        }
        return routeContext.routeForwardDelta;
    }

    public currentControlTelemetry(): {
        currentSpeed: number
        currentYaw: number
        yawRate: number
        steering: number
        acceleration: number
        brakePressureAvg: number
        routeForwardDelta: number | null
        routeHeadingError: number | null
        routeDistance: number | null
        hasLeadVehicle: boolean
        leadVehicleDistance: number | null
        gameTimeMs: number
    } | null {
        if (!this.oldEgo || !isValidEntity(this.oldEgo.vehicle.id)) {
            return null;
        }
        const id = this.oldEgo.vehicle.id;
        const coords = GetEntityCoords(id, true) as Vector3;
        const yaw = GetEntityHeading(id);
        const currentSpeed = GetEntitySpeed(id);
        const acceleration = GetVehicleCurrentAcceleration(id);
        const brakePressureAvg = this.averageBrakePressure(id);
        const steering = GetVehicleWheelSteeringAngle(id, 0);
        const rotationVelocity = this.toVector3(GetEntityRotationVelocity(id)) ?? [0, 0, 0];
        const [gpsRouteFound, gpsRouteCoords] = GetPosAlongGpsTypeRoute(true, 1, 0);
        const gps = this.toVector3(gpsRouteCoords) ?? [0, 0, 0];
        const routeContext = this.collectRouteContext(id, coords, yaw, gpsRouteFound, gps);
        const nearbyActors = this.collectNearbyActorSummary(id, coords, yaw, currentSpeed);
        return {
            currentSpeed,
            currentYaw: yaw,
            yawRate: rotationVelocity[2],
            steering,
            acceleration,
            brakePressureAvg,
            routeForwardDelta: routeContext.routeForwardDelta ?? null,
            routeHeadingError: routeContext.routeHeadingError ?? null,
            routeDistance: routeContext.routeDistance ?? null,
            hasLeadVehicle: nearbyActors.hasLeadVehicle,
            leadVehicleDistance: nearbyActors.leadVehicleDistance ?? null,
            gameTimeMs: GetGameTimer(),
        };
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
            let captureStarted = false;
            let tripCaptureClosed = false;

            try {
                this.tripChunkIndex = 0;
                this.tripDataPointIndex = 0;
                const tripProfile = buildDeterministicTripProfile(
                    this.RunId,
                    sceneInfo.sceneId,
                    sceneInfo.sceneVariant,
                    tripIndex,
                    this.baseEnvironment ?? undefined
                );
                await this.prepareTripRuntime(ego, tripProfile);
                await wait(1000)

                try {
                    await requestCapture("start", {
                        runId: this.RunId,
                        tripIndex,
                        sceneId: sceneInfo.sceneId,
                        sceneVariant: sceneInfo.sceneVariant,
                        sceneName: this.SceneName
                    });
                    captureStarted = true;
                } catch (error) {
                    console.error(`[ego] failed to start capture for trip ${tripIndex} run ${this.RunId}: ${error}`);
                    throw error;
                }

                await wait(1000)
                const syncTime = await syncFlash(1000);
                console.log('Routing to waypoint:', waypoint.destination);
                this.activeTripContext = {
                    sceneId: sceneInfo.sceneId,
                    sceneVariant: sceneInfo.sceneVariant,
                    tripIndex,
                    syncTime,
                    chunkStartTime: syncTime,
                    fromDestination: previousDestination,
                    toDestination: waypoint.destination,
                    tripProfile
                };
                const [x, y, z] = this.configureRoute(waypoint, ego.vehicle.maxSpeed, ego.vehicle.drivingStyle).destination;
                SetNewWaypoint(x,y);
                await wait(1000)
                const blip = GetFirstBlipInfoId(8)
                this.driveToWaypoint(PlayerPedId(), ego.vehicle.id, blip, ego.vehicle.drivingStyle, ego.vehicle.maxSpeed);
                const signal = await this.watchDog( await this.drivingLoop(ego,[x,y,z]), 10*60_000);
                await new Promise<void>((resolve) => {
                    signal.addEventListener("abort", () => resolve());
                });

                const tripEndedByStop = this.stopRequested;
                await this.finalizeTripCapture(ego, tripIndex, sceneInfo, GetGameTimer());
                tripCaptureClosed = true;
                if (!tripEndedByStop) {
                    previousDestination = waypoint.destination
                }
                this.activeTripContext = null;

                if (tripEndedByStop) {
                    ClearGpsPlayerWaypoint()
                    throw new Error(SceneStoppedErrorCode);
                }
            } catch (error: any) {
                if (this.stopRequested || error?.message === SceneStoppedErrorCode) {
                    this.activeTripContext = null;
                    ego.vehicle.VehicleData = [];
                    ClearGpsPlayerWaypoint();
                    throw error;
                }
                await this.enterHardFailureState(
                    ego,
                    tripIndex,
                    sceneInfo,
                    `trip ${tripIndex} failed for run ${this.RunId}: ${error?.message ?? error}`,
                    captureStarted && !tripCaptureClosed
                );
            }
        }

        if (this.stopRequested) {
            ClearGpsPlayerWaypoint()
            throw new Error(SceneStoppedErrorCode);
        }
        TaskVehicleDriveWander(PlayerPedId(), ego.vehicle.id, ego.vehicle.maxSpeed, ego.vehicle.drivingStyle);
    }

    private async enterHardFailureState(
        ego: Ego,
        tripIndex: number,
        sceneInfo: { sceneId: string; sceneVariant: string },
        reason: string,
        shouldAbortCapture: boolean
    ): Promise<never> {
        console.error(`[ego] HARD FAILURE build=${EGO_BUILD_ID} run=${this.RunId} scene=${sceneInfo.sceneId}:${sceneInfo.sceneVariant} trip=${tripIndex}: ${reason}`);
        this.stopRequested = true;
        this.stopReason = reason;
        if (this.activeAbortController && !(this.activeAbortController as any).signal?.aborted) {
            this.activeAbortController.abort();
        }
        this.stopManagedLoops();
        ClearGpsPlayerWaypoint();
        ClearPedTasksImmediately(PlayerPedId());

        if (shouldAbortCapture) {
            try {
                await requestCapture("abort", {
                    runId: this.RunId,
                    tripIndex,
                    sceneId: sceneInfo.sceneId,
                    sceneVariant: sceneInfo.sceneVariant,
                    sceneName: this.SceneName
                });
            } catch (abortError: any) {
                console.error(`[ego] capture abort failed during hard failure for trip ${tripIndex} run ${this.RunId}: ${abortError?.message ?? abortError}`);
            }
        }

        if (isValidEntity(ego.vehicle.id)) {
            SetVehicleHandbrake(ego.vehicle.id, true);
            SetVehicleEngineOn(ego.vehicle.id, false, true, true);
            SetVehicleUndriveable(ego.vehicle.id, true);
            FreezeEntityPosition(ego.vehicle.id, true);
        }
        ego.vehicle.VehicleData = [];
        this.activeTripContext = null;
        throw new Error(`${HardFailureErrorCode}: ${reason}`);
    }

    private async finalizeTripCapture(
        ego: Ego,
        tripIndex: number,
        sceneInfo: { sceneId: string; sceneVariant: string },
        endTime: number
    ) {
        if (!this.activeTripContext) {
            throw new Error(`trip ${tripIndex} capture finalization requested without an active trip context`);
        }

        this.emitTripChunk(ego, endTime, true);
        console.log(`[ego] finalize-trip request build=${EGO_BUILD_ID} run=${this.RunId} scene=${sceneInfo.sceneId}:${sceneInfo.sceneVariant} trip=${tripIndex}`);
        const finalizeResponse = await requestTripFinalize({
            runId: this.RunId,
            tripIndex,
            sceneId: sceneInfo.sceneId,
            sceneVariant: sceneInfo.sceneVariant,
            sceneName: this.SceneName
        });
        const finalizedTrip = this.requireFinalizedTrip(finalizeResponse, tripIndex, sceneInfo);
        console.log(`[ego] finalize-trip ack run=${finalizedTrip.runId} scene=${finalizedTrip.sceneId}:${finalizedTrip.sceneVariant} trip=${finalizedTrip.tripIndex}`);
        console.log(`[ego] capture:stop request run=${finalizedTrip.runId} scene=${finalizedTrip.sceneId}:${finalizedTrip.sceneVariant} trip=${finalizedTrip.tripIndex}`);
        await requestCapture("stop", {
            runId: finalizedTrip.runId,
            tripIndex: finalizedTrip.tripIndex,
            sceneId: finalizedTrip.sceneId,
            sceneVariant: finalizedTrip.sceneVariant,
            sceneName: this.SceneName
        });
        console.log(`[ego] capture:stop ack run=${finalizedTrip.runId} scene=${finalizedTrip.sceneId}:${finalizedTrip.sceneVariant} trip=${finalizedTrip.tripIndex}`);
    }

    private requireFinalizedTrip(
        response: TripFinalizeResponse,
        tripIndex: number,
        sceneInfo: { sceneId: string; sceneVariant: string }
    ) {
        const runId = typeof response.runId === "string" ? response.runId.trim() : "";
        const sceneId = typeof response.sceneId === "string" ? response.sceneId.trim() : "";
        const sceneVariant = typeof response.sceneVariant === "string" ? response.sceneVariant.trim() : "";
        const finalizedTripIndex = typeof response.tripIndex === "number" ? response.tripIndex : Number.NaN;

        if (!runId || !sceneId || !sceneVariant || !Number.isInteger(finalizedTripIndex) || finalizedTripIndex < 0) {
            throw new Error("finalize-trip ack was missing required trip identity fields");
        }
        if (runId !== this.RunId) {
            throw new Error(`finalize-trip ack returned mismatched runId ${runId} for run ${this.RunId}`);
        }
        if (sceneId !== sceneInfo.sceneId || sceneVariant !== sceneInfo.sceneVariant || finalizedTripIndex !== tripIndex) {
            throw new Error(
                `finalize-trip ack returned mismatched trip identity: expected ${sceneInfo.sceneId}:${sceneInfo.sceneVariant}:${tripIndex}, got ${sceneId}:${sceneVariant}:${finalizedTripIndex}`
            );
        }

        return {
            runId,
            sceneId,
            sceneVariant,
            tripIndex: finalizedTripIndex
        };
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
            tripProfile: this.activeTripContext.tripProfile,
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

    private async prepareTripRuntime(ego: Ego, tripProfile: TripProfileSnapshot) {
        if (this.baseEnvironment) {
            this.environmentService.execute({
                weatherType: {
                    type: tripProfile.weatherType,
                    persistent: this.baseEnvironment.weatherType?.persistent ?? true
                },
                Time: tripProfile.time
            });
        }

        if (this.oldEgo && isValidEntity(this.oldEgo.vehicle.id)) {
            this.cleanUp(this.oldEgo);
        }

        await this.setVehicle(ego, tripProfile.vehicleModel, tripProfile.vehicleColor);
        if (!isValidEntity(ego.vehicle.id)) {
            throw new Error("failed to initialize trip vehicle");
        }

        SetVehRadioStation(ego.vehicle.id, "OFF");
        SetVehicleEngineOn(ego.vehicle.id, true, true, false);
        SetVehicleUndriveable(ego.vehicle.id, false);
        FreezeEntityPosition(ego.vehicle.id, false);
        this.makeVehicleGodMode(ego.vehicle.id);
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
        const brakePressureAvg = this.averageBrakePressure(id);
        const Steering = GetVehicleWheelSteeringAngle(id, 0);
        const yaw = GetEntityHeading(id)
        const time = GetGameTimer();
        const coords = this.toVector3(GetEntityCoords(id, false)) ?? [0, 0, 0];
        const [gpsRouteFound, gpsRouteCoords] = GetPosAlongGpsTypeRoute(true, 1, 0);
        const gps = this.toVector3(gpsRouteCoords) ?? [0, 0, 0];
        const velocityWorld = this.toVector3(GetEntityVelocity(id)) ?? [0, 0, 0];
        const velocityLocal = this.toVector3(GetEntitySpeedVector(id, true)) ?? [0, 0, 0];
        const rotationVelocity = this.toVector3(GetEntityRotationVelocity(id)) ?? [0, 0, 0];
        const gear = GetVehicleCurrentGear(id);
        const rpm = GetVehicleCurrentRpm(id);
        const engineHealth = GetVehicleEngineHealth(id);
        const bodyHealth = GetVehicleBodyHealth(id);
        const handbrake = GetVehicleHandbrake(id);
        const stoppedAtTrafficLights = IsVehicleStoppedAtTrafficLights(id);
        const routeContext = this.collectRouteContext(id, coords, yaw, gpsRouteFound, gps);
        const nearbyActors = this.collectNearbyActorSummary(id, coords, yaw, currentSpeed);
        const [streetNameHash, crossingRoadHash] = GetStreetNameAtCoord(coords[0], coords[1], coords[2]);
        const timeSinceSyncMs = this.activeTripContext ? Math.max(0, time - this.activeTripContext.syncTime) : undefined;
        const timeSinceChunkStartMs = this.activeTripContext ? Math.max(0, time - this.activeTripContext.chunkStartTime) : undefined;
        const eventCollision = HasEntityCollidedWithAnything(id);
        const eventOffroad = !routeContext.isOnRoad || routeContext.isOffroadNode;
        const eventWrongWay = routeContext.roadNodeDistance !== undefined
            && routeContext.roadNodeDistance <= 12
            && routeContext.roadNodeHeading !== undefined
            && Math.abs(this.headingDeltaDegrees(yaw, routeContext.roadNodeHeading)) >= 110;
        const eventReversing = velocityLocal[1] < -0.75;

        const data: VehicleData = {
            time,
            currentSpeed,
            drivingStyle: ego.vehicle.drivingStyle,
            acceleration,
            brakePressureAvg,
            isStopped,
            Steering,
            yaw,
            coords,
            gps,
            velocityWorldX: velocityWorld[0],
            velocityWorldY: velocityWorld[1],
            velocityWorldZ: velocityWorld[2],
            velocityForward: velocityLocal[1],
            velocityLateral: velocityLocal[0],
            velocityVertical: velocityLocal[2],
            yawRate: rotationVelocity[2],
            gear,
            rpm,
            engineHealth,
            bodyHealth,
            routeGpsValid: routeContext.routeGpsValid,
            routeDistance: routeContext.routeDistance,
            routeHeadingError: routeContext.routeHeadingError,
            routeForwardDelta: routeContext.routeForwardDelta,
            routeLateralDelta: routeContext.routeLateralDelta,
            roadNodeDistance: routeContext.roadNodeDistance,
            roadNodeHeading: routeContext.roadNodeHeading,
            roadNodeDensity: routeContext.roadNodeDensity,
            roadLaneCountForward: routeContext.roadLaneCountForward,
            roadLaneCountBackward: routeContext.roadLaneCountBackward,
            roadEdgeSpan: routeContext.roadEdgeSpan,
            isOnRoad: routeContext.isOnRoad,
            isOffroadNode: routeContext.isOffroadNode,
            isInJunction: routeContext.isInJunction,
            hasTrafficLightNode: routeContext.hasTrafficLightNode,
            isHighway: routeContext.isHighway,
            nearbyVehicleCount30m: nearbyActors.nearbyVehicleCount30m,
            nearbyPedCount20m: nearbyActors.nearbyPedCount20m,
            hasLeadVehicle: nearbyActors.hasLeadVehicle,
            leadVehicleDistance: nearbyActors.leadVehicleDistance,
            leadVehicleRelativeSpeed: nearbyActors.leadVehicleRelativeSpeed,
            leadVehicleTTC: nearbyActors.leadVehicleTTC,
            leadVehicleHeadingDelta: nearbyActors.leadVehicleHeadingDelta,
            streetNameHash,
            crossingRoadHash,
            isStoppedAtTrafficLights: stoppedAtTrafficLights,
            eventCollision,
            eventOffroad,
            eventWrongWay,
            eventReversing,
            eventHandbrake: handbrake,
            timeSinceSyncMs,
            timeSinceChunkStartMs,
            dataPointIndex: this.tripDataPointIndex,
            chunkIndex: this.tripChunkIndex
        };

        if (!ego.vehicle.VehicleData) {
            ego.vehicle.VehicleData = [];
        }
        ego.vehicle.VehicleData.push(data)
        this.tripDataPointIndex += 1;

        if (ego.vehicle.VehicleData.length >= EgoService.MAX_TRIP_VEHICLE_DATA_POINTS) {
            this.emitTripChunk(ego, GetGameTimer(), false);
        }
    }

    private averageBrakePressure(vehicle: number): number {
        if (!isValidEntity(vehicle)) {
            return 0;
        }
        const wheelCount = GetVehicleNumberOfWheels(vehicle);
        if (!Number.isFinite(wheelCount) || wheelCount <= 0) {
            return 0;
        }
        let totalPressure = 0;
        let observedWheels = 0;
        for (let wheelIndex = 0; wheelIndex < wheelCount; wheelIndex += 1) {
            const pressure = GetVehicleWheelBrakePressure(vehicle, wheelIndex);
            if (!Number.isFinite(pressure)) {
                continue;
            }
            totalPressure += pressure;
            observedWheels += 1;
        }
        if (observedWheels <= 0) {
            return 0;
        }
        return this.clampNumber(totalPressure / observedWheels, 0, 1);
    }

    private clampNumber(value: number, minValue: number, maxValue: number): number {
        if (!Number.isFinite(value)) {
            return minValue;
        }
        return Math.min(Math.max(value, minValue), maxValue);
    }

    private collectRouteContext(id: number, coords: Vector3, heading: number, gpsRouteFound: boolean, gps: Vector3) {
        const isOnRoad = IsPointOnRoad(coords[0], coords[1], coords[2], id);
        const context: {
            routeGpsValid: boolean
            routeDistance?: number
            routeHeadingError?: number
            routeForwardDelta?: number
            routeLateralDelta?: number
            roadNodeDistance?: number
            roadNodeHeading?: number
            roadNodeDensity?: number
            roadLaneCountForward?: number
            roadLaneCountBackward?: number
            roadEdgeSpan?: number
            isOnRoad: boolean
            isOffroadNode: boolean
            isInJunction: boolean
            hasTrafficLightNode: boolean
            isHighway: boolean
        } = {
            routeGpsValid: false,
            isOnRoad,
            isOffroadNode: false,
            isInJunction: false,
            hasTrafficLightNode: false,
            isHighway: false,
        };

        if (gpsRouteFound && this.isFiniteVector3(gps)) {
            const delta = this.subtractVectors(gps, coords);
            const deltaHeading = this.vectorHeadingDegrees(delta);
            const forward = this.headingForwardVector(heading);
            const right = this.headingRightVector(heading);
            context.routeGpsValid = true;
            context.routeDistance = this.vectorLength(delta);
            context.routeHeadingError = this.headingDeltaDegrees(deltaHeading, heading);
            context.routeForwardDelta = this.dotProduct(delta, forward);
            context.routeLateralDelta = this.dotProduct(delta, right);
        }

        const [nodeFound, nodeCoordsRaw, nodeHeading] = GetClosestVehicleNodeWithHeading(coords[0], coords[1], coords[2], 1, 3, 0);
        const nodeCoords = this.toVector3(nodeCoordsRaw);
        if (nodeFound && nodeCoords) {
            context.roadNodeDistance = this.distanceBetween(coords, nodeCoords);
            context.roadNodeHeading = nodeHeading;
        }

        const [propertiesFound, nodeDensity, nodeFlags] = GetVehicleNodeProperties(coords[0], coords[1], coords[2]);
        if (propertiesFound) {
            context.roadNodeDensity = nodeDensity;
            context.isOffroadNode = (nodeFlags & VehicleNodeFlags.OffRoad) !== 0;
            context.isInJunction = (nodeFlags & VehicleNodeFlags.Junction) !== 0;
            context.hasTrafficLightNode = (nodeFlags & VehicleNodeFlags.TrafficLight) !== 0;
            context.isHighway = (nodeFlags & VehicleNodeFlags.Highway) !== 0;
        }

        const [roadFound, roadEdgeA, roadEdgeB, laneCountForward, laneCountBackward] = GetClosestRoad(coords[0], coords[1], coords[2], 1.5, 0, false);
        const roadEdgeAVector = this.toVector3(roadEdgeA);
        const roadEdgeBVector = this.toVector3(roadEdgeB);
        if (roadFound && roadEdgeAVector && roadEdgeBVector) {
            context.roadLaneCountForward = laneCountForward;
            context.roadLaneCountBackward = laneCountBackward;
            context.roadEdgeSpan = this.distanceBetween(roadEdgeAVector, roadEdgeBVector);
        }

        return context;
    }

    private collectNearbyActorSummary(id: number, coords: Vector3, heading: number, egoSpeed: number) {
        const vehicles = ((GetGamePool("CVehicle") as number[]) ?? []) as number[];
        const peds = ((GetGamePool("CPed") as number[]) ?? []) as number[];
        let nearbyVehicleCount30m = 0;
        let nearbyPedCount20m = 0;
        let bestLeadDistance = Number.POSITIVE_INFINITY;
        let leadVehicleDistance: number | undefined;
        let leadVehicleRelativeSpeed: number | undefined;
        let leadVehicleTTC: number | undefined;
        let leadVehicleHeadingDelta: number | undefined;
        let hasLeadVehicle = false;
        const forward = this.headingForwardVector(heading);
        const right = this.headingRightVector(heading);

        for (const vehicleId of vehicles) {
            if (vehicleId === id || !isValidEntity(vehicleId)) {
                continue;
            }
            const vehicleCoords = this.toVector3(GetEntityCoords(vehicleId, false));
            if (!vehicleCoords) {
                continue;
            }
            const offset = this.subtractVectors(vehicleCoords, coords);
            const distance = this.vectorLength(offset);
            if (distance <= 30) {
                nearbyVehicleCount30m += 1;
            }
            if (distance < 2 || distance > 50) {
                continue;
            }

            const forwardDistance = this.dotProduct(offset, forward);
            const lateralDistance = Math.abs(this.dotProduct(offset, right));
            if (forwardDistance <= 0 || lateralDistance > Math.max(4.5, forwardDistance * 0.4)) {
                continue;
            }
            if (distance >= bestLeadDistance) {
                continue;
            }

            const candidateSpeed = GetEntitySpeed(vehicleId);
            const relativeSpeed = Math.max(0, egoSpeed - candidateSpeed);
            bestLeadDistance = distance;
            hasLeadVehicle = true;
            leadVehicleDistance = distance;
            leadVehicleRelativeSpeed = relativeSpeed;
            leadVehicleTTC = relativeSpeed > 0.25 ? distance / relativeSpeed : undefined;
            leadVehicleHeadingDelta = Math.abs(this.headingDeltaDegrees(heading, GetEntityHeading(vehicleId)));
        }

        const playerPed = PlayerPedId();
        for (const pedId of peds) {
            if (pedId === playerPed || !isValidEntity(pedId) || IsPedInVehicle(pedId, id, false)) {
                continue;
            }
            const pedCoords = this.toVector3(GetEntityCoords(pedId, false));
            if (!pedCoords) {
                continue;
            }
            if (this.distanceBetween(coords, pedCoords) <= 20) {
                nearbyPedCount20m += 1;
            }
        }

        return {
            nearbyVehicleCount30m,
            nearbyPedCount20m,
            hasLeadVehicle,
            leadVehicleDistance,
            leadVehicleRelativeSpeed,
            leadVehicleTTC,
            leadVehicleHeadingDelta,
        };
    }

    private toVector3(value: unknown): Vector3 | null {
        if (!Array.isArray(value) || value.length < 3) {
            return null;
        }
        const x = Number(value[0]);
        const y = Number(value[1]);
        const z = Number(value[2]);
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
            return null;
        }
        return [x, y, z];
    }

    private isFiniteVector3(value: Vector3 | null): value is Vector3 {
        return !!value
            && Number.isFinite(value[0])
            && Number.isFinite(value[1])
            && Number.isFinite(value[2]);
    }

    private subtractVectors(a: Vector3, b: Vector3): Vector3 {
        return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    }

    private vectorLength(value: Vector3): number {
        return Math.sqrt((value[0] * value[0]) + (value[1] * value[1]) + (value[2] * value[2]));
    }

    private distanceBetween(a: Vector3, b: Vector3): number {
        return this.vectorLength(this.subtractVectors(a, b));
    }

    private dotProduct(a: Vector3, b: Vector3): number {
        return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]);
    }

    private headingForwardVector(heading: number): Vector3 {
        const radians = heading * (Math.PI / 180);
        return [Math.sin(radians), Math.cos(radians), 0];
    }

    private headingRightVector(heading: number): Vector3 {
        const radians = heading * (Math.PI / 180);
        return [Math.cos(radians), -Math.sin(radians), 0];
    }

    private vectorHeadingDegrees(value: Vector3): number {
        const radians = Math.atan2(value[0], value[1]);
        return this.normalizeHeadingDegrees(radians * (180 / Math.PI));
    }

    private normalizeHeadingDegrees(heading: number): number {
        let normalized = heading % 360;
        if (normalized < 0) {
            normalized += 360;
        }
        return normalized;
    }

    private headingDeltaDegrees(targetHeading: number, sourceHeading: number): number {
        let delta = this.normalizeHeadingDegrees(targetHeading) - this.normalizeHeadingDegrees(sourceHeading);
        while (delta > 180) {
            delta -= 360;
        }
        while (delta < -180) {
            delta += 360;
        }
        return delta;
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

    public startCaptureCamera(ego: Ego) {
        if (this.cameraTickId !== null) {
            clearTick(this.cameraTickId);
        }
        this.cameraTickId = setTick(() => {
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
        });
    }

    private stopCaptureCamera() {
        if (this.cameraTickId === null) {
            return;
        }
        clearTick(this.cameraTickId);
        this.cameraTickId = null;
    }

    public async setVehicle(ego: Ego, modelOverride?: string, colorOverride?: VehicleColor){
        const resolvedModel = modelOverride ?? this.resolveVehicleModel(ego.vehicle.model);
        const resolvedColor = colorOverride ?? this.resolveVehicleColor(ego.vehicle.color);

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
        await this.addPedToVehicle(PlayerPedId(), newVehicle);
        SetVehicleEngineOn(newVehicle, true, true, false);
        SetVehicleUndriveable(newVehicle, false);
        FreezeEntityPosition(newVehicle, false);
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

    public async addPedToVehicle(egoId:number, vehicle: number) {
        TaskWarpPedIntoVehicle(egoId, vehicle, -1);
        SetPedIntoVehicle(egoId, vehicle, -1);

        const startedAt = GetGameTimer();
        while (GetGameTimer() - startedAt < 1000) {
            if (IsPedInVehicle(egoId, vehicle, false) && GetPedInVehicleSeat(vehicle, -1) === egoId) {
                console.log(`Player ped seated in vehicle ${vehicle} as driver.`);
                return;
            }
            await wait(50);
            SetPedIntoVehicle(egoId, vehicle, -1);
        }

        console.log(`Failed to confirm player ped seated in vehicle ${vehicle} as driver.`);
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
            return;
        }

        SetEntityAsMissionEntity(ego.vehicle.id, true, true);
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
