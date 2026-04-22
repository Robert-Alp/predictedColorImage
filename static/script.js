const dropzone    = document.getElementById("dropzone");
const fileInput   = document.getElementById("file-input");
const label       = document.getElementById("dropzone-label");
const btnColorize = document.getElementById("btn-colorize");
const modeToggle  = document.getElementById("mode-toggle");
const spinner     = document.getElementById("spinner");
const errorMsg    = document.getElementById("error-msg");
const warningMsg  = document.getElementById("warning-msg");
const result      = document.getElementById("result");
const imgOriginal  = document.getElementById("img-original");
const imgMask      = document.getElementById("img-mask");
const imgRepaired  = document.getElementById("img-repaired");
const imgColorized = document.getElementById("img-colorized");
const restorePanels = document.querySelectorAll(".restore-only");

let selectedFile = null;

// ── Toggle ────────────────────────────────────────────────────────────────
modeToggle.addEventListener("change", () => {
  result.classList.add("hidden");
  hideError();
  hideWarning();
});

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
  hideWarning();
  result.classList.add("hidden");
}

// ── Envoi et affichage ────────────────────────────────────────────────────
btnColorize.addEventListener("click", async () => {
  if (!selectedFile) return;

  const mode = modeToggle.checked ? "restore" : "colorize";

  btnColorize.disabled = true;
  spinner.classList.remove("hidden");
  result.classList.add("hidden");
  hideError();
  hideWarning();

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("mode", mode);

  try {
    const resp = await fetch("/process", { method: "POST", body: formData });
    const data = await resp.json();

    if (!resp.ok) {
      showError(data.error || "Erreur serveur.");
      return;
    }

    if (data.warning) showWarning(data.warning);

    imgOriginal.src  = "data:image/png;base64," + data.original;
    imgColorized.src = "data:image/png;base64," + data.colorized;

    if (mode === "restore" && data.mask) {
      imgMask.src     = "data:image/png;base64," + data.mask;
      imgRepaired.src = "data:image/png;base64," + data.repaired;
      restorePanels.forEach(el => el.classList.remove("hidden"));
    } else {
      restorePanels.forEach(el => el.classList.add("hidden"));
    }

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
function hideError() { errorMsg.classList.add("hidden"); }

function showWarning(msg) {
  warningMsg.textContent = msg;
  warningMsg.classList.remove("hidden");
}
function hideWarning() { warningMsg.classList.add("hidden"); }
