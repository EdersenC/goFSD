import {parseStopSignCatalogPosition, setStopSignCatalogWaypoint} from "./catalog-waypoint";

function assert(condition: unknown, message: string): asserts condition {
    if (!condition) {
        throw new Error(message);
    }
}

const calls: Array<[number, number]> = [];
const position = setStopSignCatalogWaypoint(
    {x: -2335.7034, y: 3269.0288, z: 31.81049},
    {setNewWaypoint: (x, y) => calls.push([x, y])},
);
assert(position.z === 31.81049, "catalog position should preserve elevation");
assert(calls.length === 1 && calls[0]![0] === -2335.7034 && calls[0]![1] === 3269.0288, "catalog position should set the GTA waypoint");

for (const invalid of [null, {}, {x: Number.NaN, y: 0, z: 0}, {x: 20_000, y: 0, z: 0}]) {
    let rejected = false;
    try {
        parseStopSignCatalogPosition(invalid);
    } catch {
        rejected = true;
    }
    assert(rejected, `invalid catalog position must be rejected: ${JSON.stringify(invalid)}`);
}

console.log("stop-sign catalog waypoint tests passed");
