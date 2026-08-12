export type StopSignCatalogLocation = {
    id: string
    model: string
    kind: string
    x: number
    y: number
    z: number
    tilted: boolean
    sourceYmap: string
};

const requiredColumns = [
    "id",
    "model",
    "kind",
    "is_stop_control",
    "x",
    "y",
    "z",
    "rotation_x",
    "rotation_y",
    "source_ymap",
] as const;

export function parseStopSignCatalogCsv(csv: string): StopSignCatalogLocation[] {
    const rows = parseCsvRows(csv);
    if (rows.length < 2) {
        throw new Error("Stop-sign catalog has no locations");
    }
    const header = rows[0] ?? [];
    const indexes = new Map(header.map((name, index) => [name.trim(), index]));
    for (const column of requiredColumns) {
        if (indexes.get(column) === undefined) {
            throw new Error(`Stop-sign catalog is missing ${column}`);
        }
    }

    const seen = new Set<string>();
    const locations: StopSignCatalogLocation[] = [];
    for (let rowIndex = 1; rowIndex < rows.length; rowIndex += 1) {
        const row = rows[rowIndex] ?? [];
        if (row.length === 1 && !row[0]?.trim()) {
            continue;
        }
        const read = (column: string) => row[indexes.get(column)!]?.trim() ?? "";
        const id = read("id");
        if (!id || seen.has(id)) {
            throw new Error(`Stop-sign catalog row ${rowIndex + 1} has an empty or duplicate id`);
        }
        if (read("is_stop_control").toLowerCase() !== "true") {
            continue;
        }
        seen.add(id);
        const rotationX = finiteNumber(read("rotation_x"), rowIndex, "rotation_x");
        const rotationY = finiteNumber(read("rotation_y"), rowIndex, "rotation_y");
        locations.push({
            id,
            model: read("model"),
            kind: read("kind"),
            x: finiteNumber(read("x"), rowIndex, "x"),
            y: finiteNumber(read("y"), rowIndex, "y"),
            z: finiteNumber(read("z"), rowIndex, "z"),
            tilted: Math.abs(rotationX) > .001 || Math.abs(rotationY) > .001,
            sourceYmap: read("source_ymap"),
        });
    }
    if (locations.length === 0) {
        throw new Error("Stop-sign catalog has no stop-control locations");
    }
    return locations;
}

function finiteNumber(value: string, rowIndex: number, column: string): number {
    const parsed = Number(value);
    if (!value || !Number.isFinite(parsed)) {
        throw new Error(`Stop-sign catalog row ${rowIndex + 1} has invalid ${column}`);
    }
    return parsed;
}

function parseCsvRows(csv: string): string[][] {
    const rows: string[][] = [];
    let row: string[] = [];
    let field = "";
    let quoted = false;
    for (let index = 0; index < csv.length; index += 1) {
        const character = csv[index]!;
        if (character === '"') {
            if (quoted && csv[index + 1] === '"') {
                field += '"';
                index += 1;
            } else {
                quoted = !quoted;
            }
            continue;
        }
        if (!quoted && character === ",") {
            row.push(field);
            field = "";
            continue;
        }
        if (!quoted && (character === "\n" || character === "\r")) {
            if (character === "\r" && csv[index + 1] === "\n") {
                index += 1;
            }
            row.push(field);
            rows.push(row);
            row = [];
            field = "";
            continue;
        }
        field += character;
    }
    if (field || row.length > 0) {
        row.push(field);
        rows.push(row);
    }
    if (quoted) {
        throw new Error("Stop-sign catalog has an unterminated quoted field");
    }
    return rows;
}
