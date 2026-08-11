export type ManagedEgoCompletion = {
    forceSafeStopAndDisposeCurrentEgo: (reason: string) => void
};

export function completeSuccessfulScene(
    sceneName: string,
    managedEgo: ManagedEgoCompletion,
): void {
    const normalizedSceneName = sceneName.trim();
    if (!normalizedSceneName) {
        throw new Error("Successful scene completion invariant violated: scene name is empty");
    }

    managedEgo.forceSafeStopAndDisposeCurrentEgo(
        `successful scene "${normalizedSceneName}" completed`,
    );
}
