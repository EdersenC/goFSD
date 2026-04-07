import {SceneType} from "../sceneManger";
import {innerCityDrivingScene} from "./inner-city-driving";

export const localScenes: Record<string, SceneType> = {
    "inner-city-driving": innerCityDrivingScene
};

export const defaultSceneId = "inner-city-driving";

export const localSceneIds = Object.keys(localScenes);

export function getLocalScene(id: string): SceneType | undefined {
    return localScenes[id];
}

export const defaultScene = localScenes[defaultSceneId];
