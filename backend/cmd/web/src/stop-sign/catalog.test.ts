import {parseStopSignCatalogCsv} from "./catalog";

const csv = `id,model,kind,world_scope,is_stop_control,x,y,z,rotation_x,rotation_y,rotation_z,rotation_w,source_ymap
gta-v-sign-0001,prop_sign_road_01a,standard_stop,map,True,-2335.7034,3269.0288,31.81049,0,0,0.9,-0.2,first.ymap
gta-v-sign-0002,prop_sign_road_01a,standard_stop,map,True,-2266.7798,3336.8223,31.844904,0.055,-0.026,0.9,-0.3,"folder,with-comma.ymap"
ignored,prop,other,map,False,0,0,0,0,0,0,1,ignored.ymap`;

const locations = parseStopSignCatalogCsv(csv);
assert(locations.length === 2, "only stop-control rows should be returned");
assert(locations[0]?.x === -2335.7034, "coordinates should parse exactly");
assert(locations[1]?.tilted === true, "tilted props should be identified");
assert(locations[1]?.sourceYmap === "folder,with-comma.ymap", "quoted CSV fields should parse");

for (const invalid of ["", "id,x\nonly,1", csv.replace("gta-v-sign-0002", "gta-v-sign-0001")]) {
    let rejected = false;
    try {
        parseStopSignCatalogCsv(invalid);
    } catch {
        rejected = true;
    }
    assert(rejected, "invalid catalog must fail closed");
}

console.log("stop-sign catalog parser tests passed");

function assert(condition: unknown, message: string): asserts condition {
    if (!condition) {
        throw new Error(message);
    }
}
