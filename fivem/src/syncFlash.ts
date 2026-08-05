/**
 * Draws the capture-alignment flash and returns its first game-clock timestamp.
 */
export async function syncFlash(durationMs = 250): Promise<number> {
    const startTime = GetGameTimer();
    const flashUntilGameMs = startTime + durationMs;
    return new Promise<number>((resolve) => {
        const tickId = setTick(() => {
            if (GetGameTimer() >= flashUntilGameMs) {
                clearTick(tickId);
                resolve(startTime);
                return;
            }
            DrawRect(0.5, 0.5, 1, 1, 255, 255, 255, 255);
        });
    });
}
