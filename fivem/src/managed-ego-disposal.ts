export type ManagedEgoEntity = {
    vehicle: {id: number}
};

/** A failed verified deletion must retain its handle so every later Hold fails closed and retries it. */
export function requireManagedEgoDisposed<T extends ManagedEgoEntity>(
    managedEgo: T | null,
    isEntityValid: (entity: number) => boolean,
    restoreManagedEgo: (ego: T) => void,
): void {
    const vehicle = managedEgo?.vehicle.id ?? 0;
    if (!isEntityValid(vehicle)) {
        return;
    }
    if (managedEgo === null) {
        throw new Error(`Managed ego disposal invariant violated for vehicle ${vehicle}`);
    }

    restoreManagedEgo(managedEgo);
    throw new Error(`Failed to dispose managed ego vehicle ${vehicle} during fail-safe cleanup`);
}
