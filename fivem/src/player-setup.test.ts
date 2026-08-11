import {
    applyJoinedPlayerSetup,
    applyPlayerGodMode,
    disablePlayerGodMode,
    FULL_WEAPON_NAMES,
    PLAYER_WEAPON_AMMO,
    PlayerSetupOperations,
} from "./player-setup";

function assert(condition: boolean, message: string): asserts condition {
    if (!condition) {
        throw new Error(message);
    }
}

function test(name: string, body: () => void) {
    body();
    console.log(`ok - ${name}`);
}

function setupFixture(validPed = true) {
    const calls: string[] = [];
    const granted: Array<{hash: number, ammo: number}> = [];
    const operations: PlayerSetupOperations = {
        playerId: () => 7,
        playerPedId: () => 42,
        entityExists: () => validPed,
        setPlayerInvincible: (player, enabled) => calls.push(`player:${player}:${enabled}`),
        setEntityInvincible: (ped, enabled) => calls.push(`entity:${ped}:${enabled}`),
        setEntityCanBeDamaged: (ped, enabled) => calls.push(`damaged:${ped}:${enabled}`),
        setPedDropsWeaponsWhenDead: (ped, enabled) => calls.push(`drops:${ped}:${enabled}`),
        setPedInfiniteAmmoClip: (ped, enabled) => calls.push(`clip:${ped}:${enabled}`),
        hashWeaponName: (name) => FULL_WEAPON_NAMES.indexOf(name as typeof FULL_WEAPON_NAMES[number]) + 100,
        isWeaponValid: (hash) => hash !== 100,
        giveWeaponToPed: (_ped, hash, ammo) => granted.push({hash, ammo}),
        setPedInfiniteAmmo: (ped, enabled, hash) => calls.push(`ammo:${ped}:${hash}:${enabled}`),
    };
    return {operations, calls, granted};
}

test("joined player setup owns god mode and grants every valid official weapon", () => {
    const {operations, calls, granted} = setupFixture();
    const result = applyJoinedPlayerSetup(operations);
    assert(result?.ped === 42, "expected the current player ped");
    assert(result.grantedWeaponCount === FULL_WEAPON_NAMES.length - 1, "invalid weapon hashes must be skipped");
    assert(granted.length === result.grantedWeaponCount, "every valid weapon must be granted once");
    assert(granted.every((weapon) => weapon.ammo === PLAYER_WEAPON_AMMO), "weapons must receive the configured ammo");
    assert(calls.includes("player:7:true"), "player invincibility must be enabled");
    assert(calls.includes("entity:42:true"), "ped invincibility must be enabled");
    assert(calls.includes("damaged:42:false"), "ped damage must be disabled");
    assert(calls.includes("clip:42:true"), "infinite clip must be enabled after granting weapons");
});

test("player setup is a no-op until a valid player ped exists", () => {
    const {operations, calls, granted} = setupFixture(false);
    assert(applyPlayerGodMode(operations) === null, "invalid player ped must be rejected");
    assert(calls.length === 0 && granted.length === 0, "no native mutations are allowed before spawn");
});

test("resource cleanup releases only player god mode", () => {
    const {operations, calls} = setupFixture();
    disablePlayerGodMode(operations);
    assert(calls.includes("player:7:false"), "player invincibility must be released");
    assert(calls.includes("entity:42:false"), "ped invincibility must be released");
    assert(calls.includes("damaged:42:true"), "ped damage must be restored");
});

console.log("player setup tests passed");
