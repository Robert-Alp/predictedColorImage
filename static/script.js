const dropzone    = document.getElementById("dropzone");
const fileInput   = document.getElementById("file-input");
const label       = document.getElementById("dropzone-label");
const btnColorize = document.getElementById("btn-colorize");
const spinner     = document.getElementById("spinner");
const errorMsg    = document.getElementById("error-msg");
const result      = document.getElementById("result");
const imgOriginal  = document.getElementById("img-original");
const imgColorized = document.getElementById("img-colorized");

let selectedFile = null;

// ── Sélection de fichier ───────────────────────────────────────────────────

dropzone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("over");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("over"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("over");
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});

function setFile(file) {
  const allowed = ["image/jpeg", "image/png"];
  if (!allowed.includes(file.type)) {
    showError("Seuls les fichiers JPG et PNG sont acceptés.");
    return;
  }
  selectedFile = file;
  label.textContent = file.name;
  dropzone.classList.add("has-file");
  btnColorize.disabled = false;
  hideError();
  result.classList.add("hidden");
}

// ── Envoi et affichage ────────────────────────────────────────────────────

btnColorize.addEventListener("click", async () => {
  if (!selectedFile) return;

  btnColorize.disabled = true;
  spinner.classList.remove("hidden");
  result.classList.add("hidden");
  hideError();

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const resp = await fetch("/colorize", { method: "POST", body: formData });
    const data = await resp.json();

    if (!resp.ok) {
      showError(data.error || "Erreur serveur.");
      return;
    }

    imgOriginal.src  = "data:image/png;base64," + data.original;
    imgColorized.src = "data:image/png;base64," + data.colorized;
    result.classList.remove("hidden");
  } catch (err) {
    showError("Impossible de contacter le serveur.");
  } finally {
    spinner.classList.add("hidden");
    btnColorize.disabled = false;
  }
});

// ── Helpers ───────────────────────────────────────────────────────────────

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.classList.remove("hidden");
}
function hideError() {
  errorMsg.classList.add("hidden");
}
