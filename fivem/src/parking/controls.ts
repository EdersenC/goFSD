export const PARKING_DISABLED_CONTROL_IDS = [
    59, // INPUT_VEH_MOVE_LR
    60, // INPUT_VEH_MOVE_UD
    63, // INPUT_VEH_MOVE_LEFT_ONLY
    64, // INPUT_VEH_MOVE_RIGHT_ONLY
    71, // INPUT_VEH_ACCELERATE
    72, // INPUT_VEH_BRAKE
    75, // INPUT_VEH_EXIT
    76, // INPUT_VEH_HANDBRAKE
] as const;

export const PARKING_POST_FLASH_CLEAN_FRAME_BUFFER_MS = 350;
export const PARKING_EVALUATION_SPAWN_SETTLE_MS = 1000;
export const PARKING_EVALUATION_COLLISION_CLEAN_BUFFER_MS = 350;

/** Suppresses player and virtual-gamepad inputs without interrupting AI vehicle tasks. */
export function suppressParkingVehicleControls() {
    for (const controlId of PARKING_DISABLED_CONTROL_IDS) {
        DisableControlAction(0, controlId, true);
    }
}
