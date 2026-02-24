//srv/server.ts

import fetch from "node-fetch";
import {log} from "./helper";
import {newScene, normalizeScenePayload, SceneType} from "./sceneManger";

console.log("[server] loaded");

async function getDataSetScenes(id:string): Promise<SceneType> {
    const response = await fetch(`http://localhost:8080/datasets/${id}/scene`);
    console.log(`[server] fetched scene data for dataset ${id} with status ${response.status}`, response);
    const data = await response.json();
    // @ts-ignore
    const scene = normalizeScenePayload(data?.scene);
    console.log(scene);
    return scene
}


onNet("demo:requestScenes", async () => {
    const src = global.source as number;
    console.log("[server] received request for scenes from", src);
    const sceneData = await getDataSetScenes('patrol-default');
    console.log("Sending Scene: " + sceneData);
    console.log(`customScene: ${JSON.stringify(newScene)}`);
    emitNet("demo:responseScenes", src, sceneData);
});

