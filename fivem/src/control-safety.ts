export type ControlSafetySnapshot = {
    appliedSafetyEpoch: number
    inFlightSafetyStarts: number
    hasStaleSafetyStarts: boolean
};

export type ControlSafetyEpochSync = {
    requestId: string
    sessionId: string
    safetyEpoch: number
};

export function parseControlSafetyEpochSync(value: unknown): ControlSafetyEpochSync {
    const candidate = value as Partial<ControlSafetyEpochSync> | null;
    const requestId = typeof candidate?.requestId === "string" ? candidate.requestId.trim() : "";
    const sessionId = typeof candidate?.sessionId === "string" ? candidate.sessionId.trim() : "";
    const safetyEpoch = candidate?.safetyEpoch;
    if (typeof safetyEpoch !== "number" || !requestId || !sessionId || !Number.isSafeInteger(safetyEpoch) || safetyEpoch < 1) {
        throw new Error("Control safety epoch synchronization payload is invalid");
    }
    return {requestId, sessionId, safetyEpoch};
}

export function canAcknowledgeControlSafetyEpochSync(
    sync: ControlSafetyEpochSync,
    cleanupComplete: boolean,
    snapshot: ControlSafetySnapshot,
): boolean {
    return cleanupComplete
        && snapshot.appliedSafetyEpoch >= sync.safetyEpoch
        && snapshot.inFlightSafetyStarts === 0;
}

export class StaleSafetyStartError extends Error {
    public constructor() {
        super("Safety start was canceled by a newer emergency stop");
        this.name = "StaleSafetyStartError";
    }
}

export type SafetyStartLease = {
    readonly safetyEpoch: number
    isStale: () => boolean
    throwIfStale: () => void
    finish: () => ControlSafetySnapshot
};

export type GuardedSafetyStartResult<T> =
    | {kind: "completed", value: T}
    | {kind: "canceled"};

type ActiveSafetyStart = {
    safetyEpoch: number
    finished: boolean
};

export class ControlSafetyBarrier {
    private appliedSafetyEpoch = 0;
    private nextLeaseId = 0;
    private readonly starts = new Map<number, ActiveSafetyStart>();

    public beginStart(safetyEpoch: number): SafetyStartLease {
        assertSafetyEpoch(safetyEpoch);
        if (safetyEpoch < this.appliedSafetyEpoch) {
            throw new StaleSafetyStartError();
        }
        this.appliedSafetyEpoch = safetyEpoch;
        if (this.starts.size > 0) {
            throw new StaleSafetyStartError();
        }

        const leaseId = ++this.nextLeaseId;
        const start: ActiveSafetyStart = {safetyEpoch, finished: false};
        this.starts.set(leaseId, start);

        return {
            safetyEpoch,
            isStale: () => safetyEpoch < this.appliedSafetyEpoch,
            throwIfStale: () => {
                if (safetyEpoch < this.appliedSafetyEpoch) {
                    throw new StaleSafetyStartError();
                }
            },
            finish: () => {
                if (!start.finished) {
                    start.finished = true;
                    this.starts.delete(leaseId);
                }
                return this.snapshot();
            },
        };
    }

    public applyEmergencyStop(safetyEpoch: number): ControlSafetySnapshot {
        assertSafetyEpoch(safetyEpoch);
        this.appliedSafetyEpoch = Math.max(this.appliedSafetyEpoch, safetyEpoch);
        return this.snapshot();
    }

    public acceptNonStartCommand(safetyEpoch: number): boolean {
        assertSafetyEpoch(safetyEpoch);
        if (safetyEpoch < this.appliedSafetyEpoch) {
            return false;
        }
        this.appliedSafetyEpoch = safetyEpoch;
        return true;
    }

    public snapshot(): ControlSafetySnapshot {
        let hasStaleSafetyStarts = false;
        for (const start of this.starts.values()) {
            if (start.safetyEpoch < this.appliedSafetyEpoch) {
                hasStaleSafetyStarts = true;
                break;
            }
        }
        return {
            appliedSafetyEpoch: this.appliedSafetyEpoch,
            inFlightSafetyStarts: this.starts.size,
            hasStaleSafetyStarts,
        };
    }
}

export async function runGuardedSafetyStart<T>(
    barrier: ControlSafetyBarrier,
    safetyEpoch: number,
    work: (lease: SafetyStartLease) => Promise<T>,
    cleanup: () => Promise<void> | void,
    onBarrierChange: (snapshot: ControlSafetySnapshot) => void = () => undefined,
): Promise<GuardedSafetyStartResult<T>> {
    let lease: SafetyStartLease;
    try {
        lease = barrier.beginStart(safetyEpoch);
    } catch (error) {
        if (error instanceof StaleSafetyStartError) {
            onBarrierChange(barrier.snapshot());
            return {kind: "canceled"};
        }
        throw error;
    }

    onBarrierChange(barrier.snapshot());
    try {
        lease.throwIfStale();
        const value = await work(lease);
        lease.throwIfStale();
        return {kind: "completed", value};
    } catch (error) {
        if (!lease.isStale()) {
            throw error;
        }
        await cleanup();
        return {kind: "canceled"};
    } finally {
        onBarrierChange(lease.finish());
    }
}

function assertSafetyEpoch(safetyEpoch: number): void {
    if (!Number.isSafeInteger(safetyEpoch) || safetyEpoch < 1) {
        throw new Error(`Invalid control safety epoch: ${String(safetyEpoch)}`);
    }
}
