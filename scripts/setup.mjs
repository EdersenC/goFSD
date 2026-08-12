import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const packageRoots = [
  join(projectRoot, "fivem"),
  join(projectRoot, "backend", "cmd", "web"),
];

function parseArguments(arguments_) {
  if (arguments_.length === 0) {
    return { dryRun: false };
  }
  if (arguments_.length === 1 && arguments_[0] === "--dry-run") {
    return { dryRun: true };
  }
  throw new Error("Usage: npm run setup -- [--dry-run]");
}

function assertLockfiles() {
  for (const packageRoot of packageRoots) {
    const lockfile = join(packageRoot, "package-lock.json");
    if (!existsSync(lockfile)) {
      throw new Error(`Lockfile is missing: ${lockfile}`);
    }
  }
}

function lockedInstall(packageRoot) {
  const npmCli = process.env.npm_execpath;
  if (!npmCli) {
    throw new Error("Run setup through npm: npm run setup");
  }

  const result = spawnSync(
    process.execPath,
    [npmCli, "--prefix", packageRoot, "ci", "--no-audit", "--no-fund"],
    { cwd: projectRoot, stdio: "inherit" },
  );
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    throw new Error(`npm ci failed for ${relative(projectRoot, packageRoot)}`);
  }
}

function main() {
  const { dryRun } = parseArguments(process.argv.slice(2));
  assertLockfiles();

  console.log("Stop Sign Lab setup");
  for (const packageRoot of packageRoots) {
    const packageName = relative(projectRoot, packageRoot).replaceAll("\\", "/");
    if (dryRun) {
      console.log(`dry   npm --prefix ${packageName} ci --no-audit --no-fund`);
      continue;
    }
    console.log(`install ${packageName}`);
    lockedInstall(packageRoot);
  }

  if (!dryRun) {
    console.log("\nReady. Start the workspace with: npm start");
  }
}

try {
  main();
} catch (error) {
  console.error(`error ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
}
