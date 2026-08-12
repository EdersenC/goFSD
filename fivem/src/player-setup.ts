export const FULL_WEAPON_NAMES = [
    "WEAPON_ACIDPACKAGE",
    "WEAPON_ADVANCEDRIFLE",
    "WEAPON_APPISTOL",
    "WEAPON_ASSAULTRIFLE",
    "WEAPON_ASSAULTRIFLE_MK2",
    "WEAPON_ASSAULTSHOTGUN",
    "WEAPON_ASSAULTSMG",
    "WEAPON_AUTOSHOTGUN",
    "WEAPON_BALL",
    "WEAPON_BAT",
    "WEAPON_BATTLEAXE",
    "WEAPON_BATTLERIFLE",
    "WEAPON_BOTTLE",
    "WEAPON_BULLPUPRIFLE",
    "WEAPON_BULLPUPRIFLE_MK2",
    "WEAPON_BULLPUPSHOTGUN",
    "WEAPON_BZGAS",
    "WEAPON_CANDYCANE",
    "WEAPON_CARBINERIFLE",
    "WEAPON_CARBINERIFLE_MK2",
    "WEAPON_CERAMICPISTOL",
    "WEAPON_COMBATMG",
    "WEAPON_COMBATMG_MK2",
    "WEAPON_COMBATPDW",
    "WEAPON_COMBATPISTOL",
    "WEAPON_COMBATSHOTGUN",
    "WEAPON_COMPACTLAUNCHER",
    "WEAPON_COMPACTRIFLE",
    "WEAPON_CROWBAR",
    "WEAPON_DAGGER",
    "WEAPON_DBSHOTGUN",
    "WEAPON_DOUBLEACTION",
    "WEAPON_EMPLAUNCHER",
    "WEAPON_FERTILIZERCAN",
    "WEAPON_FIREEXTINGUISHER",
    "WEAPON_FIREWORK",
    "WEAPON_FLARE",
    "WEAPON_FLAREGUN",
    "WEAPON_FLASHLIGHT",
    "WEAPON_GADGETPISTOL",
    "WEAPON_GOLFCLUB",
    "WEAPON_GRENADE",
    "WEAPON_GRENADELAUNCHER",
    "WEAPON_GRENADELAUNCHER_SMOKE",
    "WEAPON_GUSENBERG",
    "WEAPON_HACKINGDEVICE",
    "WEAPON_HAMMER",
    "WEAPON_HATCHET",
    "WEAPON_HAZARDCAN",
    "WEAPON_HEAVYPISTOL",
    "WEAPON_HEAVYRIFLE",
    "WEAPON_HEAVYSHOTGUN",
    "WEAPON_HEAVYSNIPER",
    "WEAPON_HEAVYSNIPER_MK2",
    "WEAPON_HOMINGLAUNCHER",
    "WEAPON_KNIFE",
    "WEAPON_KNUCKLE",
    "WEAPON_MACHETE",
    "WEAPON_MACHINEPISTOL",
    "WEAPON_MARKSMANPISTOL",
    "WEAPON_MARKSMANRIFLE",
    "WEAPON_MARKSMANRIFLE_MK2",
    "WEAPON_METALDETECTOR",
    "WEAPON_MG",
    "WEAPON_MICROSMG",
    "WEAPON_MILITARYRIFLE",
    "WEAPON_MINIGUN",
    "WEAPON_MINISMG",
    "WEAPON_MOLOTOV",
    "WEAPON_MUSKET",
    "WEAPON_NAVYREVOLVER",
    "WEAPON_NIGHTSTICK",
    "WEAPON_PETROLCAN",
    "WEAPON_PIPEBOMB",
    "WEAPON_PISTOL",
    "WEAPON_PISTOL50",
    "WEAPON_PISTOLXM3",
    "WEAPON_PISTOL_MK2",
    "WEAPON_POOLCUE",
    "WEAPON_PRECISIONRIFLE",
    "WEAPON_PROXMINE",
    "WEAPON_PUMPSHOTGUN",
    "WEAPON_PUMPSHOTGUN_MK2",
    "WEAPON_RAILGUN",
    "WEAPON_RAILGUNXM3",
    "WEAPON_RAYCARBINE",
    "WEAPON_RAYMINIGUN",
    "WEAPON_RAYPISTOL",
    "WEAPON_REVOLVER",
    "WEAPON_REVOLVER_MK2",
    "WEAPON_RPG",
    "WEAPON_SAWNOFFSHOTGUN",
    "WEAPON_SMG",
    "WEAPON_SMG_MK2",
    "WEAPON_SMOKEGRENADE",
    "WEAPON_SNIPERRIFLE",
    "WEAPON_SNOWBALL",
    "WEAPON_SNOWLAUNCHER",
    "WEAPON_SNSPISTOL",
    "WEAPON_SNSPISTOL_MK2",
    "WEAPON_SPECIALCARBINE",
    "WEAPON_SPECIALCARBINE_MK2",
    "WEAPON_STICKYBOMB",
    "WEAPON_STONE_HATCHET",
    "WEAPON_STUNGUN",
    "WEAPON_STUNGUN_MP",
    "WEAPON_STUNROD",
    "WEAPON_SWITCHBLADE",
    "WEAPON_TACTICALRIFLE",
    "WEAPON_TECPISTOL",
    "WEAPON_VINTAGEPISTOL",
    "WEAPON_WRENCH",
] as const;

export const PLAYER_WEAPON_AMMO = 9999;

export type PlayerSetupOperations = {
    playerId: () => number
    playerPedId: () => number
    entityExists: (entity: number) => boolean
    setPlayerInvincible: (player: number, enabled: boolean) => void
    setEntityInvincible: (entity: number, enabled: boolean) => void
    setEntityCanBeDamaged: (entity: number, enabled: boolean) => void
    setPedDropsWeaponsWhenDead: (ped: number, enabled: boolean) => void
    setPedInfiniteAmmoClip: (ped: number, enabled: boolean) => void
    hashWeaponName: (weaponName: string) => number
    isWeaponValid: (weaponHash: number) => boolean
    giveWeaponToPed: (ped: number, weaponHash: number, ammo: number) => void
    setPedInfiniteAmmo: (ped: number, enabled: boolean, weaponHash: number) => void
};

export type PlayerSetupResult = {
    ped: number
    grantedWeaponCount: number
};

export function applyPlayerGodMode(operations: PlayerSetupOperations): number | null {
    const ped = operations.playerPedId();
    if (ped === 0 || !operations.entityExists(ped)) {
        return null;
    }
    operations.setPlayerInvincible(operations.playerId(), true);
    operations.setEntityInvincible(ped, true);
    operations.setEntityCanBeDamaged(ped, false);
    operations.setPedDropsWeaponsWhenDead(ped, false);
    return ped;
}

export function grantFullWeaponLoadout(
    ped: number,
    operations: PlayerSetupOperations,
): number {
    let grantedWeaponCount = 0;
    for (const weaponName of FULL_WEAPON_NAMES) {
        const weaponHash = operations.hashWeaponName(weaponName);
        if (!operations.isWeaponValid(weaponHash)) {
            continue;
        }
        operations.giveWeaponToPed(ped, weaponHash, PLAYER_WEAPON_AMMO);
        operations.setPedInfiniteAmmo(ped, true, weaponHash);
        grantedWeaponCount += 1;
    }
    operations.setPedInfiniteAmmoClip(ped, true);
    return grantedWeaponCount;
}

export function applyJoinedPlayerSetup(
    operations: PlayerSetupOperations,
): PlayerSetupResult | null {
    const ped = applyPlayerGodMode(operations);
    if (ped === null) {
        return null;
    }
    return {
        ped,
        grantedWeaponCount: grantFullWeaponLoadout(ped, operations),
    };
}

export function disablePlayerGodMode(operations: PlayerSetupOperations) {
    const ped = operations.playerPedId();
    operations.setPlayerInvincible(operations.playerId(), false);
    if (ped === 0 || !operations.entityExists(ped)) {
        return;
    }
    operations.setEntityInvincible(ped, false);
    operations.setEntityCanBeDamaged(ped, true);
}
