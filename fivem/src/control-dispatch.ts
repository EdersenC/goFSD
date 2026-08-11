export type ControlDispatchCommandType =
    | "startScene"
    | "runAllScenes"
    | "endScene"
    | "endAllScenes"
    | "startEgo"
    | "stopEgo"
    | "setStopSignTarget"
    | "clearStopSignTarget"
    | "setStopSignCatalogWaypoint"
    | "probeStopSignTarget"
    | "startStopSignBatch";

export type ControlDispatchCommand = {
    id: string
    type: ControlDispatchCommandType
    safetyEpoch: number
    [key: string]: unknown
};

export type ControlDispatchRejectionReason = "missing" | "canceled" | "staleSafetyEpoch";

export type ControlDispatchConfirmation =
    | {
        commandId: string
        confirmed: true
        command: ControlDispatchCommand
    }
    | {
        commandId: string
        confirmed: false
        reason: ControlDispatchRejectionReason
    };

export class ControlDispatchTimeoutError extends Error {
    public constructor(label: string, timeoutMs: number) {
        super(`${label} timed out after ${timeoutMs}ms`);
        this.name = "ControlDispatchTimeoutError";
    }
}

export class ControlPollSingleFlight {
    private inFlight = false;

    public async run(work: () => Promise<void>): Promise<boolean> {
        if (this.inFlight) {
            return false;
        }

        this.inFlight = true;
        try {
            await work();
            return true;
        } finally {
            this.inFlight = false;
        }
    }
}

export async function runWithAbortTimeout<T>(
    label: string,
    timeoutMs: number,
    work: (signal: AbortSignal) => Promise<T>,
): Promise<T> {
    const normalizedLabel = label.trim();
    if (!normalizedLabel) {
        throw new Error("Timed control dispatch work requires a label");
    }
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
        throw new Error(`Invalid control dispatch timeout: ${String(timeoutMs)}`);
    }

    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const timeout = new Promise<never>((_resolve, reject) => {
        timer = setTimeout(() => {
            reject(new ControlDispatchTimeoutError(normalizedLabel, timeoutMs));
            controller.abort();
        }, timeoutMs);
    });

    try {
        return await Promise.race([work(controller.signal), timeout]);
    } finally {
        if (timer !== undefined) {
            clearTimeout(timer);
        }
    }
}

const rejectionReasons = new Set<ControlDispatchRejectionReason>([
    "missing",
    "canceled",
    "staleSafetyEpoch",
]);

const commandTypes = new Set<ControlDispatchCommandType>([
    "startScene",
    "runAllScenes",
    "endScene",
    "endAllScenes",
    "startEgo",
    "stopEgo",
    "setStopSignTarget",
    "clearStopSignTarget",
    "setStopSignCatalogWaypoint",
    "probeStopSignTarget",
    "startStopSignBatch",
]);

export function parseControlDispatchConfirmation(
    value: unknown,
    expectedCommandId: string,
): ControlDispatchConfirmation {
    const commandId = expectedCommandId.trim();
    if (!commandId) {
        throw new Error("Dispatch confirmation requires an expected command id");
    }
    if (!isRecord(value) || value.commandId !== commandId || typeof value.confirmed !== "boolean") {
        throw new Error(`Invalid dispatch confirmation for command ${commandId}`);
    }

    if (!value.confirmed) {
        if (!isRejectionReason(value.reason) || value.command != null) {
            throw new Error(`Invalid dispatch rejection for command ${commandId}`);
        }
        return {
            commandId,
            confirmed: false,
            reason: value.reason,
        };
    }

    const command = value.command;
    if (
        value.reason != null
        || !isRecord(command)
        || command.id !== commandId
        || !isCommandType(command.type)
        || !Number.isSafeInteger(command.safetyEpoch)
        || Number(command.safetyEpoch) < 1
    ) {
        throw new Error(`Invalid confirmed command payload for command ${commandId}`);
    }

    return {
        commandId,
        confirmed: true,
        command: command as ControlDispatchCommand,
    };
}

export function dispatchConfirmationAdvancesCursor(confirmation: ControlDispatchConfirmation): boolean {
    return confirmation.confirmed || confirmation.reason !== "missing";
}

function isRejectionReason(value: unknown): value is ControlDispatchRejectionReason {
    return typeof value === "string" && rejectionReasons.has(value as ControlDispatchRejectionReason);
}

function isCommandType(value: unknown): value is ControlDispatchCommandType {
    return typeof value === "string" && commandTypes.has(value as ControlDispatchCommandType);
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}
