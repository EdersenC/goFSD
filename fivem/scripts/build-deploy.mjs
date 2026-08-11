import {constants as fsConstants} from "node:fs";
import {
    access,
    copyFile,
    lstat,
    mkdir,
    readFile,
    realpath,
    rename,
    unlink,
} from "node:fs/promises";
import path from "node:path";
import {homedir} from "node:os";
import {fileURLToPath} from "node:url";
import {build, context} from "esbuild";

const supportedArguments = new Set(["--dry-run", "--watch"]);
const requestedArguments = process.argv.slice(2);
const unsupportedArguments = requestedArguments.filter((argument) => !supportedArguments.has(argument));
if (unsupportedArguments.length > 0) {
    throw new Error(`Unsupported deploy argument(s): ${unsupportedArguments.join(", ")}`);
}

const watch = requestedArguments.includes("--watch");
const dryRun = requestedArguments.includes("--dry-run");
if (watch && dryRun) {
    throw new Error("--watch and --dry-run cannot be used together");
}

const projectDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const resourceDir = await resolveResourceDirectory(process.env.FIVEM_RESOURCE_DIR, projectDir);
const resourceDistDir = path.join(resourceDir, "dist");
const deployments = [
    {
        label: "client bundle",
        source: path.join(projectDir, "dist", "client.js"),
        destination: path.join(resourceDistDir, "client.js"),
    },
    {
        label: "server bundle",
        source: path.join(projectDir, "dist", "server.js"),
        destination: path.join(resourceDistDir, "server.js"),
    },
    {
        label: "resource manifest",
        source: path.join(projectDir, "fxmanifest.lua"),
        destination: path.join(resourceDir, "fxmanifest.lua"),
    },
];

if (dryRun) {
    await validateDeploymentSources(deployments, {bundlesRequired: false});
    await validateDeploymentDestinations(deployments);
    console.log(`FiveM deploy target is valid: ${resourceDir}`);
    for (const deployment of deployments) {
        console.log(`[dry-run] ${deployment.label}: ${deployment.source} -> ${deployment.destination}`);
    }
} else if (watch) {
    await watchAndDeploy();
} else {
    await buildRepositoryBundles();
    await deployArtifacts();
}

function buildOptions() {
    return [
        {
            entryPoints: ["src/client.ts"],
            platform: "browser",
            format: "iife",
            target: "es2017",
            outfile: path.join(projectDir, "dist", "client.js"),
        },
        {
            entryPoints: ["src/server.ts"],
            platform: "node",
            format: "cjs",
            target: "node18",
            outfile: path.join(projectDir, "dist", "server.js"),
        },
    ];
}

async function buildRepositoryBundles() {
    await Promise.all(buildOptions().map((options) => build({
        absWorkingDir: projectDir,
        bundle: true,
        ...options,
    })));
}

async function deployArtifacts() {
    await validateResourceWriteAccess(resourceDir);
    await ensureResourceDistDirectory(resourceDistDir);
    await access(resourceDistDir, fsConstants.W_OK);
    await validateDeploymentSources(deployments, {bundlesRequired: true});
    await validateDeploymentDestinations(deployments);

    for (const deployment of deployments) {
        await replaceDeploymentFile(deployment.source, deployment.destination);
        console.log(`Deployed ${deployment.label} to ${deployment.destination}`);
    }
}

async function replaceDeploymentFile(source, destination) {
    const suffix = `${process.pid}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const stagingPath = path.join(path.dirname(destination), `.${path.basename(destination)}.${suffix}.staging`);
    const backupPath = path.join(path.dirname(destination), `.${path.basename(destination)}.${suffix}.backup`);
    const destinationExists = await optionalLstat(destination) !== null;
    let backupCreated = false;

    await copyFile(source, stagingPath);
    try {
        if (destinationExists) {
            await rename(destination, backupPath);
            backupCreated = true;
        }
        try {
            await rename(stagingPath, destination);
        } catch (error) {
            if (backupCreated) {
                await rename(backupPath, destination);
                backupCreated = false;
            }
            throw error;
        }
        if (backupCreated) {
            await unlink(backupPath);
            backupCreated = false;
        }
    } finally {
        await unlinkIfPresent(stagingPath);
        if (backupCreated && await optionalLstat(destination) === null) {
            await rename(backupPath, destination);
            backupCreated = false;
        }
    }
}

async function unlinkIfPresent(targetPath) {
    try {
        await unlink(targetPath);
    } catch (error) {
        if (error?.code !== "ENOENT") {
            throw error;
        }
    }
}

async function watchAndDeploy() {
    const buildHealth = [false, false];
    let watching = false;
    let deployTimer = null;
    let deployQueue = Promise.resolve();

    const scheduleDeploy = () => {
        if (!watching || buildHealth.some((healthy) => !healthy)) {
            return;
        }
        if (deployTimer !== null) {
            clearTimeout(deployTimer);
        }
        deployTimer = setTimeout(() => {
            deployTimer = null;
            deployQueue = deployQueue
                .then(() => deployArtifacts())
                .catch((error) => {
                    console.error(`FiveM watch deploy failed: ${error?.message ?? error}`);
                });
        }, 100);
    };

    const contexts = await Promise.all(buildOptions().map((options, index) => context({
        absWorkingDir: projectDir,
        bundle: true,
        ...options,
        plugins: [{
            name: `deploy-after-${index === 0 ? "client" : "server"}-build`,
            setup(buildContext) {
                buildContext.onEnd((result) => {
                    buildHealth[index] = result.errors.length === 0;
                    scheduleDeploy();
                });
            },
        }],
    })));

    await Promise.all(contexts.map((buildContext) => buildContext.rebuild()));
    await deployArtifacts();
    watching = true;
    await Promise.all(contexts.map((buildContext) => buildContext.watch()));
    console.log(`Watching repository bundles and deploying successful builds to ${resourceDir}`);
}

async function resolveResourceDirectory(value, sourceProjectDir) {
    const configuredPath = value?.trim() || path.join(
        homedir(),
        "Fivem",
        "data",
        "cfx-server-data",
        "resources",
        "FSD"
    );
    if (!path.isAbsolute(configuredPath)) {
        throw new Error("FIVEM_RESOURCE_DIR must be an absolute path");
    }

    const configuredDir = path.resolve(configuredPath);
    if (configuredDir === path.parse(configuredDir).root) {
        throw new Error("FIVEM_RESOURCE_DIR cannot be a filesystem root");
    }

    const configuredStats = await requiredLstat(
        configuredDir,
        value?.trim() ? "FIVEM_RESOURCE_DIR" : "auto-detected FiveM FSD resource directory"
    );
    if (configuredStats.isSymbolicLink() || !configuredStats.isDirectory()) {
        throw new Error("FIVEM_RESOURCE_DIR must be a real directory, not a file or symbolic link");
    }

    const resolvedDir = await realpath(configuredDir);
    if (pathsOverlap(resolvedDir, sourceProjectDir)) {
        throw new Error("FIVEM_RESOURCE_DIR must be outside the source fivem project");
    }

    const manifestPath = path.join(resolvedDir, "fxmanifest.lua");
    const manifestStats = await requiredLstat(manifestPath, "target fxmanifest.lua");
    if (manifestStats.isSymbolicLink() || !manifestStats.isFile()) {
        throw new Error("Target fxmanifest.lua must be a regular file, not a symbolic link");
    }
    assertFiveMResourceManifest(await readFile(manifestPath, "utf8"));

    const distPath = path.join(resolvedDir, "dist");
    const distStats = await optionalLstat(distPath);
    if (distStats && (distStats.isSymbolicLink() || !distStats.isDirectory())) {
        throw new Error("Target dist must be a real directory when it already exists");
    }

    return resolvedDir;
}

async function validateResourceWriteAccess(targetDir) {
    await access(targetDir, fsConstants.W_OK);
    await access(path.join(targetDir, "fxmanifest.lua"), fsConstants.W_OK);
}

function pathsOverlap(firstPath, secondPath) {
    return isSameOrChild(firstPath, secondPath) || isSameOrChild(secondPath, firstPath);
}

function isSameOrChild(parentPath, candidatePath) {
    const relative = path.relative(parentPath, candidatePath);
    return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

function assertFiveMResourceManifest(content) {
    const requiredMarkers = [
        [/^\s*fx_version\s+['"][^'"]+['"]/m, "fx_version"],
        [/^\s*game\s+['"]gta5['"]/m, "game 'gta5'"],
        [/dist\/client\.js/, "dist/client.js client script"],
        [/dist\/server\.js/, "dist/server.js server script"],
    ];
    const missing = requiredMarkers
        .filter(([pattern]) => !pattern.test(content))
        .map(([, label]) => label);
    if (missing.length > 0) {
        throw new Error(
            `FIVEM_RESOURCE_DIR does not look like the FSD resource; target manifest is missing ${missing.join(", ")}`
        );
    }
}

async function ensureResourceDistDirectory(distPath) {
    const stats = await optionalLstat(distPath);
    if (stats) {
        if (stats.isSymbolicLink() || !stats.isDirectory()) {
            throw new Error("Target dist must be a real directory");
        }
        return;
    }
    await mkdir(distPath);
}

async function validateDeploymentSources(items, {bundlesRequired}) {
    for (const item of items) {
        const stats = await optionalLstat(item.source);
        if (!stats) {
            const isBundle = item.source.endsWith(".js");
            if (!bundlesRequired && isBundle) {
                continue;
            }
            throw new Error(`Deployment source is missing: ${item.source}`);
        }
        if (stats.isSymbolicLink() || !stats.isFile()) {
            throw new Error(`Deployment source must be a regular file: ${item.source}`);
        }
    }
}

async function validateDeploymentDestinations(items) {
    for (const item of items) {
        const stats = await optionalLstat(item.destination);
        if (stats && (stats.isSymbolicLink() || !stats.isFile())) {
            throw new Error(`Refusing to overwrite a non-file or symbolic link: ${item.destination}`);
        }
    }
}

async function requiredLstat(targetPath, label) {
    const stats = await optionalLstat(targetPath);
    if (!stats) {
        throw new Error(`${label} does not exist: ${targetPath}`);
    }
    return stats;
}

async function optionalLstat(targetPath) {
    try {
        return await lstat(targetPath);
    } catch (error) {
        if (error?.code === "ENOENT") {
            return null;
        }
        throw error;
    }
}
