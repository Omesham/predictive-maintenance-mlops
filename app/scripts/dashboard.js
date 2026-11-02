
// Predictive maintenance dashboard.
const form = document.getElementById("prediction-form");
const fileInput = document.getElementById("csv-file");
const predictButton = document.getElementById("predict-button");
const uploadStatus = document.getElementById("upload-status");
const downloadLink = document.getElementById("download-link");
const chartStatus = document.getElementById("chart-status");
const sensorSelect = document.getElementById("sensor-select");
const engineSelect = document.getElementById("engine-select");

const metricsSection = document.getElementById("metrics-section");
const chartsSection = document.getElementById("charts-section");

const MAX_FILE_SIZE = 5 * 1024 * 1024;
const MAX_BAR_ENGINES = 50;

// Illustrative thresholds, not validated maintenance limits.
const CRITICAL_RUL = 30;
const WARNING_RUL = 80;

let downloadUrl = null;
let predictionRows = [];
let latestEngineRows = [];

const charts = {};

// Parse CSV, including quoted fields and escaped quotes.
function parseCSV(text) {
    const rows = [];
    let row = [];
    let field = "";
    let quoted = false;

    text = text.replace(/^\uFEFF/, "");

    for (let i = 0; i < text.length; i++) {
        const character = text[i];

        if (character === '"') {
            if (quoted && text[i + 1] === '"') {
                field += '"';
                i++;
            } else {
                quoted = !quoted;
            }
        } else if (character === "," && !quoted) {
            row.push(field);
            field = "";
        } else if (
            (character === "\n" || character === "\r") &&
            !quoted
        ) {
            if (character === "\r" && text[i + 1] === "\n") {
                i++;
            }

            row.push(field);
            field = "";

            if (row.some(value => value.trim() !== "")) {
                rows.push(row);
            }

            row = [];
        } else {
            field += character;
        }
    }

    if (quoted) {
        throw new Error("The prediction CSV contains an unmatched quote.");
    }

    row.push(field);

    if (row.some(value => value.trim() !== "")) {
        rows.push(row);
    }

    if (rows.length < 2) {
        throw new Error("The prediction CSV contains no data rows.");
    }

    const headers = rows.shift().map(value => value.trim());

    if (new Set(headers).size !== headers.length) {
        throw new Error("The prediction CSV contains duplicate column names.");
    }

    return rows.map((values, index) => {
        if (values.length !== headers.length) {
            throw new Error(
                `Invalid CSV field count on row ${index + 2}.`
            );
        }

        const record = {};

        headers.forEach((header, columnIndex) => {
            record[header] = values[columnIndex].trim();
        });

        return record;
    });
}

function numeric(value) {
    if (value === "" || value === undefined || value === null) {
        return NaN;
    }

    return Number(value);
}

function formatCycles(value) {
    return Number.isFinite(value) ? value.toFixed(1) : "—";
}

function category(rul) {
    if (rul <= CRITICAL_RUL) return "Critical";
    if (rul <= WARNING_RUL) return "Warning";
    return "Healthy";
}

function replaceChart(name, canvasId, config) {
    if (charts[name]) {
        charts[name].destroy();
    }

    charts[name] = new Chart(
        document.getElementById(canvasId),
        config
    );
}

function chartOptions(extra = {}) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
            tooltip: {
                mode: "nearest",
                intersect: false
            }
        },
        ...extra
    };
}

// Group all observations belonging to the same engine.
function groupByEngine(rows) {
    const groups = new Map();

    rows.forEach(row => {
        if (!groups.has(row.engineId)) {
            groups.set(row.engineId, []);
        }

        groups.get(row.engineId).push(row);
    });

    // Sort each engine's observations by operating cycle.
    groups.forEach(records => {
        records.sort((a, b) => a.cycle - b.cycle);
    });

    return groups;
}

// Select the latest observation for each engine.
function getLatestEngineRows(rows) {
    const groups = groupByEngine(rows);

    return Array.from(groups.values()).map(records => {
        return records[records.length - 1];
    });
}

// Populate the engine selector.
function populateEngineSelector() {
    const engineIds = [
        ...new Set(predictionRows.map(row => row.engineId))
    ].sort((a, b) => a - b);

    const previousEngine = engineSelect.value;

    engineSelect.replaceChildren(
        ...engineIds.map(engineId => {
            const option = document.createElement("option");
            option.value = String(engineId);
            option.textContent = `Engine ${engineId}`;
            return option;
        })
    );

    if (engineIds.some(id => String(id) === previousEngine)) {
        engineSelect.value = previousEngine;
    }
}

// Populate available sensor measurements.
function populateSensorSelector(parsedRows) {
    const sensors = Object.keys(parsedRows[0]).filter(column =>
        /^sensor_measurement\d+$/.test(column)
    );

    if (!sensors.length) {
        throw new Error("No sensor_measurement columns were found.");
    }

    const previousSensor = sensorSelect.value;

    sensorSelect.replaceChildren(
        ...sensors.map(sensor => {
            const option = document.createElement("option");
            option.value = sensor;
            option.textContent = sensor.replace(
                "sensor_measurement",
                "Sensor "
            );

            return option;
        })
    );

    if (sensors.includes(previousSensor)) {
        sensorSelect.value = previousSensor;
    } else {
        sensorSelect.value = sensors.includes("sensor_measurement2")
            ? "sensor_measurement2"
            : sensors[0];
    }
}

// Update the four summary metrics.
function updateMetrics() {
    const ruls = latestEngineRows.map(row => row.rul);

    const average = ruls.reduce(
        (sum, value) => sum + value,
        0
    ) / ruls.length;

    const minimum = Math.min(...ruls);

    const criticalCount = ruls.filter(
        value => value <= CRITICAL_RUL
    ).length;

    document.getElementById("metric-total").textContent =
        String(latestEngineRows.length);

    document.getElementById("metric-average").textContent =
        formatCycles(average);

    document.getElementById("metric-minimum").textContent =
        formatCycles(minimum);

    document.getElementById("metric-alerts").textContent =
        String(criticalCount);
}

// Chart 1: Latest predicted RUL for each engine.
function drawRulChart() {
    const engines = [...latestEngineRows]
        .sort((a, b) => a.rul - b.rul)
        .slice(0, MAX_BAR_ENGINES);

    replaceChart("rul", "rul-chart", {
        type: "bar",
        data: {
            labels: engines.map(
                row => `Engine ${row.engineId}`
            ),
            datasets: [{
                label: "Predicted RUL (cycles)",
                data: engines.map(row => row.rul),
                backgroundColor: engines.map(row => {
                    if (row.rul <= CRITICAL_RUL) return "#dc5151";
                    if (row.rul <= WARNING_RUL) return "#e9a23b";
                    return "#2b8a78";
                }),
                borderRadius: 4
            }]
        },
        options: chartOptions({
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: "Remaining cycles"
                    }
                }
            }
        })
    });
}

// Chart 2: Count each engine once using its latest RUL.
function drawHealthChart() {
    const counts = {
        Critical: 0,
        Warning: 0,
        Healthy: 0
    };

    latestEngineRows.forEach(row => {
        counts[category(row.rul)]++;
    });

    replaceChart("health", "health-chart", {
        type: "doughnut",
        data: {
            labels: Object.keys(counts),
            datasets: [{
                data: Object.values(counts),
                backgroundColor: [
                    "#dc5151",
                    "#e9a23b",
                    "#2b8a78"
                ],
                borderWidth: 2
            }]
        },
        options: chartOptions({
            plugins: {
                legend: {
                    display: true,
                    position: "bottom"
                }
            }
        })
    });
}

// Charts 3 and 4: Use observations from the selected engine.
function updateSensorCharts() {
    if (!predictionRows.length || typeof Chart === "undefined") {
        return;
    }

    const engineId = numeric(engineSelect.value);
    const sensor = sensorSelect.value;

    // Only include records belonging to the selected engine.
    const engineRows = predictionRows
        .filter(row =>
            row.engineId === engineId &&
            Number.isFinite(numeric(row[sensor]))
        )
        .sort((a, b) => a.cycle - b.cycle);

    // Chart 3: Sensor readings over operating cycles.
    replaceChart("sensor", "sensor-chart", {
        type: "line",
        data: {
            datasets: [{
                label: `Engine ${engineId}: ${sensor}`,
                data: engineRows.map(row => ({
                    x: row.cycle,
                    y: numeric(row[sensor])
                })),
                borderColor: "#2563eb",
                backgroundColor: "#2563eb",
                pointRadius: 4,
                showLine: true,
                tension: 0.15
            }]
        },
        options: chartOptions({
            scales: {
                x: {
                    type: "linear",
                    title: {
                        display: true,
                        text: "Operating cycle"
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: sensor
                    }
                }
            }
        })
    });

    // Chart 4: Sensor values against predicted RUL.
    replaceChart("scatter", "scatter-chart", {
        type: "scatter",
        data: {
            datasets: [{
                label: `Engine ${engineId}`,
                data: engineRows.map(row => ({
                    x: numeric(row[sensor]),
                    y: row.rul
                })),
                backgroundColor: "#7c3aed",
                pointRadius: 5
            }]
        },
        options: chartOptions({
            scales: {
                x: {
                    title: {
                        display: true,
                        text: sensor
                    }
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: "Predicted RUL (cycles)"
                    }
                }
            }
        })
    });
}

// Draw all dashboard charts.
function drawCharts() {
    if (typeof Chart === "undefined") {
        throw new Error("Chart.js could not load.");
    }

    latestEngineRows = getLatestEngineRows(predictionRows);

    updateMetrics();
    drawRulChart();
    drawHealthChart();
    updateSensorCharts();

    chartStatus.textContent =
        `Showing ${latestEngineRows.length} engines across ` +
        `${predictionRows.length} observations. ` +
        "Health categories use illustrative RUL thresholds.";
}

// Update charts when the selected engine or sensor changes.
engineSelect.addEventListener("change", updateSensorCharts);
sensorSelect.addEventListener("change", updateSensorCharts);

// Handle CSV upload and predictions.
form.addEventListener("submit", async event => {
    event.preventDefault();

    const file = fileInput.files[0];

    if (!file) {
        uploadStatus.textContent = "Please select a CSV file.";
        return;
    }

    if (file.size > MAX_FILE_SIZE) {
        uploadStatus.textContent =
            "The CSV file must not exceed 5 MB.";
        return;
    }

    predictButton.disabled = true;
    downloadLink.hidden = true;
    metricsSection.hidden = true;
    chartsSection.hidden = true;

    uploadStatus.textContent = "Generating predictions...";

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("/api/upload", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            const missingColumns = error.detail?.missing_columns;

            throw new Error(
                Array.isArray(missingColumns)
                    ? `This CSV is missing required columns: ${missingColumns.join(", ")}.`
                    : typeof error.detail === "string"
                        ? error.detail
                        : "Unable to process the uploaded CSV."
            );
        }

        const result = await response.blob();

        // Preserve CSV download functionality.
        if (downloadUrl) {
            URL.revokeObjectURL(downloadUrl);
        }

        downloadUrl = URL.createObjectURL(result);
        downloadLink.href = downloadUrl;
        downloadLink.hidden = false;

        uploadStatus.textContent =
            "Predictions generated successfully. Your CSV is ready to download.";

        try {
            const parsed = parseCSV(await result.text());

            const requiredColumns = [
                "unit_number",
                "time_cycles",
                "predicted_rul"
            ];

            const missingColumns = requiredColumns.filter(
                column => !(column in parsed[0])
            );

            if (missingColumns.length) {
                throw new Error(
                    `Prediction results are missing: ${missingColumns.join(", ")}.`
                );
            }

            const valid = parsed.map(row => ({
                ...row,
                engineId: numeric(row.unit_number),
                cycle: numeric(row.time_cycles),
                rul: numeric(row.predicted_rul)
            }));

            if (valid.some(row =>
                !Number.isInteger(row.engineId) ||
                row.engineId <= 0 ||
                !Number.isInteger(row.cycle) ||
                row.cycle <= 0 ||
                !Number.isFinite(row.rul)
            )) {
                throw new Error(
                    "Prediction results contain invalid engine IDs, cycles, or RUL values."
                );
            }

            predictionRows = valid;

            populateEngineSelector();
            populateSensorSelector(parsed);

            drawCharts();

            // Reveal the dashboard only after charts are ready.
            metricsSection.hidden = false;
            chartsSection.hidden = false;

        } catch (chartError) {
            chartStatus.textContent =
                `Download is ready, but charts could not be updated: ${chartError.message}`;

            metricsSection.hidden = true;
            chartsSection.hidden = true;
        }

    } catch (error) {
        uploadStatus.textContent = error.message;
    } finally {
        predictButton.disabled = false;
    }
});
