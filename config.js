// Change the backend URL here only.
window.APP_CONFIG = {
  API_URL: "http://127.0.0.1:5000/predict", // Flask: POST /predict, field "leaf"
  FIELD_NAME: "leaf",
  MAX_MB: 5,
  TIMEOUT_MS: 30000,
  ALLOWED_TYPES: ["image/jpeg", "image/png", "image/webp"]
};
