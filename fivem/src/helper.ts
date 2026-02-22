



export const shuffleArray = (array: any[]) => {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}


export function isValidEntity(entity: number): boolean {
    return entity !== 0 && DoesEntityExist(entity);
}

export function log(message: string) {
    emit('chat:addMessage', message)
}

export async function ensureModelLoaded(model: number, timeoutMs = 5000): Promise<boolean> {
    if (!IsModelInCdimage(model)) return false;
    if (HasModelLoaded(model)) return true;

    const started = GetGameTimer();
    RequestModel(model);
    while (!HasModelLoaded(model)) {
        await wait(25);
        if (GetGameTimer() - started > timeoutMs) return false;
    }
    return true;
}


export function wait(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}


export interface Evaluator{
    success: boolean
    message?:string
}

export type AbortListener = () => void;

export class SimpleAbortSignal {
    public aborted = false;
    private listeners = new Set<AbortListener>();

    addEventListener(type: "abort", cb: AbortListener, opts?: { once?: boolean }) {
        if (type !== "abort") return;

        if (this.aborted) {
            cb();
            return;
        }

        if (opts?.once) {
            const onceCb = () => {
                this.listeners.delete(onceCb);
                cb();
            };
            this.listeners.add(onceCb);
        } else {
            this.listeners.add(cb);
        }
    }

    removeEventListener(type: "abort", cb: AbortListener) {
        if (type !== "abort") return;
        this.listeners.delete(cb);
    }

    _fireAbort() {
        if (this.aborted) return;
        this.aborted = true;
        for (const cb of [...this.listeners]) cb();
        this.listeners.clear();
    }
}

export class SimpleAbortController {
    public signal = new SimpleAbortSignal();
    abort() {
        this.signal._fireAbort();
    }
}

export const AbortControllerCompat: typeof AbortController | typeof SimpleAbortController =
    (globalThis as any).AbortController ?? (SimpleAbortController as any);