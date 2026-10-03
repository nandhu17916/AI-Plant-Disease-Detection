(() => {
  const C = window.APP_CONFIG, T = window.I18N;
  const $ = (id) => document.getElementById(id);
  const root = document.documentElement;
  let lang = localStorage.getItem("lang"); if (!T[lang]) lang = "en";
  let file = null, lastResult = null, lastErr = null;
  const t = (k) => T[lang][k];

  /* ---------- theme ---------- */
  function setTheme(th) { root.dataset.theme = th; localStorage.setItem("theme", th); $("themeBtn").setAttribute("aria-pressed", th === "dark"); }
  setTheme(localStorage.getItem("theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  $("themeBtn").onclick = () => setTheme(root.dataset.theme === "dark" ? "light" : "dark");

  /* ---------- language ---------- */
  function applyLang() {
    root.lang = lang; $("langSel").value = lang;
    document.querySelectorAll("[data-i18n]").forEach((el) => (el.textContent = t(el.dataset.i18n)));
    document.querySelectorAll("[data-i18n-aria]").forEach((el) => el.setAttribute("aria-label", t(el.dataset.i18nAria)));
    document.title = t("title");
    if (lastErr) showError(lastErr);
    if (lastResult) renderResult(lastResult);
  }
  $("langSel").onchange = (e) => { lang = e.target.value; localStorage.setItem("lang", lang); applyLang(); };

  /* ---------- errors ---------- */
  function showError(key) { lastErr = key; const b = $("errorBox"); b.hidden = false; b.textContent = t(key); }
  function clearError() { lastErr = null; $("errorBox").hidden = true; }
  function hideOutputs() { lastResult = null; $("invalidBox").hidden = true; $("resultCard").hidden = true; }

  /* ---------- file handling ---------- */
  function setFile(f) {
    clearError(); hideOutputs();
    if (!f) return;
    if (!C.ALLOWED_TYPES.includes(f.type)) return showError("errType");
    if (f.size > C.MAX_MB * 1048576) return showError("errSize");
    file = f;
    const url = URL.createObjectURL(f);
    $("previewImg").onload = () => URL.revokeObjectURL(url);
    $("previewImg").src = url;
    $("dropZone").hidden = true; $("previewBox").hidden = false;
  }
  function reset() {
    file = null; $("fileInput").value = ""; $("cameraInput").value = ""; $("previewImg").removeAttribute("src");
    $("previewBox").hidden = true; $("dropZone").hidden = false; clearError(); hideOutputs();
  }
  $("pickBtn").onclick = () => $("fileInput").click();
  $("camBtn").onclick = () => $("cameraInput").click();
  $("fileInput").onchange = (e) => setFile(e.target.files[0]);
  $("cameraInput").onchange = (e) => setFile(e.target.files[0]);
  $("removeBtn").onclick = reset; $("againBtn").onclick = reset;
  const dz = $("dropZone");
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("fileInput").click(); } });

  /* ---------- detection (Flask POST /predict) ---------- */
  function setBusy(b) {
    $("detectBtn").disabled = b; $("detectBtn").classList.toggle("busy", b);
    $("detectLabel").textContent = t(b ? "analyzing" : "detect");
  }
  $("detectBtn").onclick = async () => {
    clearError(); hideOutputs();
    if (!file) return showError("errNone");
    const fd = new FormData(); fd.append(C.FIELD_NAME, file); fd.append("lang", "en"); // backend returns English; the UI translates it (see i18n.js)
    const ctrl = new AbortController(), timer = setTimeout(() => ctrl.abort(), C.TIMEOUT_MS);
    setBusy(true);
    try {
      const res = await fetch(C.API_URL, { method: "POST", body: fd, signal: ctrl.signal });
      if (!res.ok) throw new Error("upload");
      const data = await res.json();
      if (data.valid_leaf === false) { $("invalidBox").hidden = false; return; }
      if (data.valid_leaf !== true) throw new Error("upload");
      lastResult = data; renderResult(data);
    } catch (err) {
      showError(err.message === "upload" ? "errUpload" : "errServer");
    } finally { clearTimeout(timer); setBusy(false); }
  };

  function toBullets(text) { return String(text || "").split(/(?<=[.!?])\s+/).filter(Boolean); }
  function fillList(id, items) {
    const ul = $(id); ul.replaceChildren();
    items.forEach((txt) => { const li = document.createElement("li"); li.textContent = txt; ul.appendChild(li); });
  }

  function renderResult(d) {
    const pct = Math.max(0, Math.min(100, Number(d.confidence) || 0));
    const L = T[lang], M = window.I18N_MAP, dm = M.diseases[d.disease], pk = M.plants[d.plant];
    $("rPlant").textContent = (pk && L.plants[pk]) || d.plant || "";
    $("rDisease").textContent = (dm && L.diseases[dm.id]) || d.disease || "";
    fillList("rAction", dm ? L.advice[dm.cat].action : toBullets(d.action));
    fillList("rPrev", dm ? L.advice[dm.cat].prevention : toBullets(d.prevention));
    $("rPct").textContent = pct.toFixed(2) + "%";
    const arc = $("gaugeArc"), len = 2 * Math.PI * 52;
    arc.style.strokeDasharray = len; arc.style.strokeDashoffset = len;
    $("resultCard").hidden = false;
    requestAnimationFrame(() => requestAnimationFrame(() => (arc.style.strokeDashoffset = len * (1 - pct / 100))));
    $("resultCard").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
  applyLang();
})();
