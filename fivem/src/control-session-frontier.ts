export type ControlConnectLease = {
    generation: number
    playerSource: number
};

export type ControlPollLease = {
    generation: number
    playerSource: number
    lastSeenCommandId: string
};

export type ControlConnectCompletion =
    | {kind: "stale"}
    | {kind: "safety-sync-required", generation: number, playerSource: number};

export type ControlOwnerClaimResult =
    | {kind: "accepted", generation: number}
    | {kind: "rejected", ownerPlayerSource: number};

export type ControlSessionFrontierOptions = {
    ownerFreshnessTimeoutMs?: number
    now?: () => number
};

const defaultOwnerFreshnessTimeoutMs = 12_000;

export type ControlSafetySyncIdentity = {
    requestId: string
    sessionId: string
    safetyEpoch: number
    playerSource: number
};

export function matchesControlSafetySyncAck(
    expected: ControlSafetySyncIdentity,
    playerSource: number | null,
    value: unknown,
): boolean {
    const candidate = value as Partial<Omit<ControlSafetySyncIdentity, "playerSource">> | null;
    return playerSource === expected.playerSource
        && candidate?.requestId === expected.requestId
        && candidate?.sessionId === expected.sessionId
        && candidate?.safetyEpoch === expected.safetyEpoch;
}

/**
 * Separates commands delivered before and after a backend consumer reset.
 * A generation is pollable only after the client acknowledges its reset epoch.
 */
export class ControlSessionFrontier {
    private readonly ownerFreshnessTimeoutMs: number;
    private readonly now: () => number;
    private requestedGeneration = 0;
    private readyGeneration = 0;
    private ownerPlayerSource: number | null = null;
    private ownerLastSeenAtMs: number | null = null;
    private requestedPlayerSource: number | null = null;
    private readyPlayerSource: number | null = null;
    private connectingGeneration: number | null = null;
    private connectingPlayerSource: number | null = null;
    private syncingGeneration: number | null = null;
    private lastSeenCommandId = "";

    public constructor(options: ControlSessionFrontierOptions = {}) {
        const ownerFreshnessTimeoutMs = options.ownerFreshnessTimeoutMs ?? defaultOwnerFreshnessTimeoutMs;
        if (!Number.isFinite(ownerFreshnessTimeoutMs) || ownerFreshnessTimeoutMs <= 0) {
            throw new Error(`Invalid control owner freshness timeout: ${String(ownerFreshnessTimeoutMs)}`);
        }
        this.ownerFreshnessTimeoutMs = ownerFreshnessTimeoutMs;
        this.now = options.now ?? Date.now;
    }

    public beginReconnect(playerSource: number): ControlOwnerClaimResult {
        assertPlayerSource(playerSource);
        if (this.ownerPlayerSource !== null && this.ownerPlayerSource !== playerSource) {
            return {kind: "rejected", ownerPlayerSource: this.ownerPlayerSource};
        }

        const generation = this.advanceGeneration();
        this.ownerPlayerSource = playerSource;
        this.ownerLastSeenAtMs = this.currentTime();
        this.requestedPlayerSource = playerSource;
        this.readyPlayerSource = null;
        this.syncingGeneration = null;
        this.lastSeenCommandId = "";
        return {kind: "accepted", generation};
    }

    public noteOwnerHeartbeat(playerSource: number | null): boolean {
        if (playerSource === null || playerSource !== this.ownerPlayerSource) {
            return false;
        }
        this.ownerLastSeenAtMs = this.currentTime();
        return true;
    }

    /** Invalidates a ready backend session while retaining the active in-game owner. */
    public requestBackendReconnect(): boolean {
        if (
            this.ownerPlayerSource === null
            || this.requestedPlayerSource !== this.ownerPlayerSource
            || this.needsConnect()
            || this.connectingGeneration !== null
            || this.syncingGeneration !== null
        ) {
            return false;
        }

        this.advanceGeneration();
        this.requestedPlayerSource = this.ownerPlayerSource;
        this.readyPlayerSource = null;
        this.syncingGeneration = null;
        this.lastSeenCommandId = "";
        return true;
    }

    public releaseOwner(playerSource: number | null): boolean {
        if (playerSource === null || playerSource !== this.ownerPlayerSource) {
            return false;
        }

        this.advanceGeneration();
        this.ownerPlayerSource = null;
        this.ownerLastSeenAtMs = null;
        this.requestedPlayerSource = null;
        this.readyPlayerSource = null;
        this.syncingGeneration = null;
        this.lastSeenCommandId = "";
        return true;
    }

    public needsConnect(): boolean {
        return this.ownerPlayerSource !== null
            && this.requestedPlayerSource === this.ownerPlayerSource
            && this.readyGeneration !== this.requestedGeneration;
    }

    public isCurrentGeneration(generation: number): boolean {
        return generation === this.requestedGeneration;
    }

    public isRequestedPlayerSource(playerSource: number | null): boolean {
        return playerSource !== null && playerSource === this.requestedPlayerSource;
    }

    public getReadyPlayerSource(): number | null {
        if (
            this.requestedGeneration === 0
            || !this.ownerIsFresh()
            || this.ownerPlayerSource === null
            || this.ownerPlayerSource !== this.requestedPlayerSource
            || this.readyGeneration !== this.requestedGeneration
            || this.readyPlayerSource === null
            || this.readyPlayerSource !== this.requestedPlayerSource
            || this.connectingGeneration !== null
            || this.syncingGeneration !== null
        ) {
            return null;
        }
        return this.readyPlayerSource;
    }

    public beginConnect(): ControlConnectLease | undefined {
        if (!this.needsConnect() || this.connectingGeneration !== null || this.syncingGeneration !== null) {
            return undefined;
        }

        if (this.requestedPlayerSource === null) {
            throw new Error("Control reconnect has no registered player source");
        }
        this.connectingGeneration = this.requestedGeneration;
        this.connectingPlayerSource = this.requestedPlayerSource;
        return {
            generation: this.connectingGeneration,
            playerSource: this.connectingPlayerSource,
        };
    }

    public completeConnect(lease: ControlConnectLease): ControlConnectCompletion {
        this.assertActiveConnect(lease);
        this.connectingGeneration = null;
        this.connectingPlayerSource = null;
        if (lease.generation !== this.requestedGeneration) {
            return {kind: "stale"};
        }

        this.syncingGeneration = lease.generation;
        return {
            kind: "safety-sync-required",
            generation: lease.generation,
            playerSource: lease.playerSource,
        };
    }

    public failConnect(lease: ControlConnectLease): void {
        this.assertActiveConnect(lease);
        this.connectingGeneration = null;
        this.connectingPlayerSource = null;
    }

    public completeSafetySync(generation: number, playerSource: number): boolean {
        if (
            this.syncingGeneration !== generation
            || generation !== this.requestedGeneration
            || playerSource !== this.requestedPlayerSource
        ) {
            return false;
        }

        this.syncingGeneration = null;
        this.readyGeneration = generation;
        this.readyPlayerSource = playerSource;
        this.lastSeenCommandId = "";
        return true;
    }

    public failSafetySync(generation: number): void {
        if (this.syncingGeneration === generation) {
            this.syncingGeneration = null;
        }
    }

    public beginPoll(): ControlPollLease | undefined {
        const playerSource = this.getReadyPlayerSource();
        if (playerSource === null) {
            return undefined;
        }

        return {
            generation: this.readyGeneration,
            playerSource,
            lastSeenCommandId: this.lastSeenCommandId,
        };
    }

    public isPollCurrent(lease: ControlPollLease): boolean {
        return this.ownerIsFresh()
            && lease.playerSource === this.ownerPlayerSource
            && lease.generation === this.requestedGeneration
            && lease.generation === this.readyGeneration
            && lease.playerSource === this.requestedPlayerSource
            && lease.playerSource === this.readyPlayerSource
            && this.connectingGeneration === null
            && this.syncingGeneration === null;
    }

    public acceptPoll(lease: ControlPollLease): boolean {
        return this.isPollCurrent(lease);
    }

    public acceptDispatchConfirmation(lease: ControlPollLease, commandId: string): boolean {
        const normalizedCommandId = commandId.trim();
        if (!normalizedCommandId) {
            throw new Error("Dispatch confirmation requires a command id");
        }
        if (!this.isPollCurrent(lease)) {
            return false;
        }

        this.lastSeenCommandId = normalizedCommandId;
        return true;
    }

    public dispatchConfirmedCommand(
        lease: ControlPollLease,
        commandId: string,
        dispatch: () => void,
    ): boolean {
        const normalizedCommandId = commandId.trim();
        if (!normalizedCommandId) {
            throw new Error("Confirmed dispatch requires a command id");
        }
        if (!this.isPollCurrent(lease)) {
            return false;
        }

        dispatch();
        this.lastSeenCommandId = normalizedCommandId;
        return true;
    }

    private assertActiveConnect(lease: ControlConnectLease): void {
        if (
            lease.generation !== this.connectingGeneration
            || lease.playerSource !== this.connectingPlayerSource
        ) {
            throw new Error(
                `Control connect lease ${lease.generation}/${lease.playerSource} is not active; `
                + `active generation/source is ${String(this.connectingGeneration)}/${String(this.connectingPlayerSource)}`,
            );
        }
    }

    private ownerIsFresh(): boolean {
        if (this.ownerLastSeenAtMs === null) {
            return false;
        }
        const ageMs = this.currentTime() - this.ownerLastSeenAtMs;
        return ageMs >= 0 && ageMs <= this.ownerFreshnessTimeoutMs;
    }

    private advanceGeneration(): number {
        if (this.requestedGeneration === Number.MAX_SAFE_INTEGER) {
            throw new Error("Control session generation exhausted");
        }
        this.requestedGeneration += 1;
        return this.requestedGeneration;
    }

    private currentTime(): number {
        const value = this.now();
        if (!Number.isFinite(value)) {
            throw new Error(`Invalid control session time: ${String(value)}`);
        }
        return value;
    }
}

function assertPlayerSource(playerSource: number): void {
    if (!Number.isSafeInteger(playerSource) || playerSource < 1) {
        throw new Error(`Invalid control player source: ${String(playerSource)}`);
    }
}
