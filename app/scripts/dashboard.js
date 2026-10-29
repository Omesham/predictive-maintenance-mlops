// Reference the upload form and its interface elements.
const form = document.getElementById("prediction-form");
const fileInput = document.getElementById("csv-file");
const predictButton = document.getElementById("predict-button");
const uploadStatus = document.getElementById("upload-status");
const downloadLink = document.getElementById("download-link");
const fileSize = 5 * 1024 * 1024
let downloadUrl = null;

form.addEventListener("submit", async (event) => {
    // Prevent the browser from reloading the page.
    event.preventDefault();

    const file = fileInput.files[0];

    if (!file) {
        uploadStatus.textContent = "Please select a CSV file.";
        return;
    }

    // Enforce the same file size limit used by the backend.
    if (file.size > fileSize) {
        uploadStatus.textContent = "The CSV file must not exceed 5 MB.";
        return;
    }

    // Reset the previous result before starting another prediction.
    downloadLink.hidden = true;
    predictButton.disabled = true;
    uploadStatus.textContent = "Generating predictions...";

    // Package the uploaded file for FastAPI.
    const formData = new FormData();
    formData.append("file", file);

    try {
        // Submit the CSV to our existing prediction upload endpoint.
        const response = await fetch("/api/upload", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            const missingColumns = error.detail?.missing_columns;
            throw new Error(
                Array.isArray(missingColumns)
                    ? `This CSV is missing required sensor columns: ${missingColumns.join(", ")}. Please upload engine sensor data with all required columns.`
                    : typeof error.detail === "string"
                        ? error.detail
                        : "Unable to process the uploaded CSV."
            );
        }

        // Convert the returned CSV into a downloadable browser object.
        const result = await response.blob();

        // Release the previous download URL to avoid memory leaks.
        if (downloadUrl) {
            URL.revokeObjectURL(downloadUrl);
        }

        downloadUrl = URL.createObjectURL(result);
        downloadLink.href = downloadUrl;
        downloadLink.hidden = false;

        uploadStatus.textContent =
            "Predictions generated successfully. Your CSV is ready to download.";

    } catch (error) {
        uploadStatus.textContent = error.message;
    } finally {
        // Re-enable the button whether the request succeeds or fails.
        predictButton.disabled = false;
    }
});