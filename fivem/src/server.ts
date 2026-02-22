//srv/server.ts

import fetch from "node-fetch";
import {log} from "./helper";
import {newScene} from "./sceneManger";
import {EgoService} from "./egoService";

async function callAPI() {
    const res = await fetch("https://api.github.com/repos/citizenfx/fivem");
    const data = await res.json();
    return data;
}


onNet("demo:requestScenes", async () => {
    const src = global.source as number;
    console.log("[server] received request for scenes from", src);
    const sceneData = newScene;
    log("Sending Scene: " + sceneData);
    emitNet("demo:responseScenes", src, sceneData);
});


console.log("[server] loaded");
onNet("demo:ping", async (msg: string) => {
    const src = global.source as number; // snapshot who called
    console.log("[server] from", src, msg);
    const data = await callAPI();
    emitNet("demo:pong", src, data);
});
