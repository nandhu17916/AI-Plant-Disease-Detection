"""
AI-Based Plant Disease Detection - Flask backend
Endpoint: POST /predict  (multipart/form-data, field "leaf", optional "lang": en | ta | hi)

Files expected in the SAME folder as this app.py:
  leaf_validator.keras   -> leaf / non-leaf model
  leaf_model.keras       -> 15-class plant disease model
  class_names.json       -> class names in training order (list, or {"0": "name", ...})

Install:  pip install flask flask-cors tensorflow pillow numpy
Run:      python app.py
"""
import io
import json
import os
import re

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

# ============================================================ CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VALIDATOR_PATH = os.path.join(BASE_DIR, "leaf_validator.keras")
DISEASE_PATH = os.path.join(BASE_DIR, "leaf_model.keras")
CLASSES_PATH = os.path.join(BASE_DIR, "class_names.json")

MAX_UPLOAD_MB = 5
LANGS = ("en", "ta", "hi")
DEFAULT_SIZE = (224, 224)      # used only if a model does not report its input size

# Set to False if the model already contains a Rescaling / preprocessing layer.
# The validator contains MobileNetV2 preprocess_input inside the model (TrueDivide 127.5 + Subtract 1),
# so it expects RAW 0-255 pixels. Dividing by 255 here would preprocess twice.
VALIDATOR_DIVIDE_BY_255 = False
DISEASE_DIVIDE_BY_255 = True

# ---- Leaf validator settings (VERIFIED from the training notebook + model file) ----------
# Training: image_dataset_from_directory(label_mode="binary") -> class_names = ['leaf', 'non_leaf']
#           so leaf = 0, non_leaf = 1, and Dense(1, sigmoid) outputs P(NON-LEAF).
#           (The notebook's own check also uses: prediction < 0.5 -> LEAF.)
VALIDATOR_OUTPUT_MEANS = "non_leaf"   # sigmoid output close to 1.0 = NON-LEAF
LEAF_CLASS_INDEX = 1                  # only used if the validator had 2 output neurons (it has 1)
LEAF_THRESHOLD = 0.5                  # leaf_score = 1 - output; leaf_score >= 0.5 -> LEAF
DEBUG_VALIDATOR = True                # prints VALIDATOR DEBUG lines in the terminal
# Call /predict?debug=1 (or send form field debug=1) to also get "validator_debug" in the JSON.

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
CORS(app)

# ============================================================ KNOWLEDGE BASE (15 classes)
PLANTS = {
    "pepper": {"en": "Bell Pepper", "ta": "குடைமிளகாய்", "hi": "शिमला मिर्च"},
    "potato": {"en": "Potato", "ta": "உருளைக்கிழங்கு", "hi": "आलू"},
    "tomato": {"en": "Tomato", "ta": "தக்காளி", "hi": "टमाटर"},
}

# disease id -> (category, names)
DISEASES = {
    "bacterial_spot": ("bacterial", {"en": "Bacterial Spot", "ta": "பாக்டீரியா புள்ளி நோய்", "hi": "जीवाणु धब्बा रोग"}),
    "early_blight": ("blight", {"en": "Early Blight", "ta": "முன்கருகல் நோய்", "hi": "अगेती झुलसा"}),
    "late_blight": ("blight", {"en": "Late Blight", "ta": "பின்கருகல் நோய்", "hi": "पछेती झुलसा"}),
    "leaf_mold": ("leaf_fungus", {"en": "Leaf Mold", "ta": "இலை பூஞ்சை நோய்", "hi": "पत्ती फफूंद रोग"}),
    "septoria": ("leaf_fungus", {"en": "Septoria Leaf Spot", "ta": "செப்டோரியா இலைப்புள்ளி நோய்", "hi": "सेप्टोरिया पत्ती धब्बा"}),
    "target_spot": ("leaf_fungus", {"en": "Target Spot", "ta": "இலக்குப் புள்ளி நோய்", "hi": "टारगेट स्पॉट (लक्ष्य धब्बा)"}),
    "spider_mites": ("mite", {"en": "Spider Mites (Two-spotted)", "ta": "சிலந்திப் பேன் (இரு புள்ளி)", "hi": "मकड़ी माइट (दो-धब्बेदार)"}),
    "curl_virus": ("virus_curl", {"en": "Yellow Leaf Curl Virus", "ta": "மஞ்சள் இலை சுருட்டை வைரஸ்", "hi": "पीला पत्ती मरोड़ वायरस"}),
    "mosaic_virus": ("virus_mosaic", {"en": "Mosaic Virus", "ta": "மொசைக் வைரஸ்", "hi": "मोज़ेक वायरस"}),
    "healthy": ("healthy", {"en": "Healthy", "ta": "ஆரோக்கியமானது", "hi": "स्वस्थ"}),
}

ADVICE = {
    "bacterial": {
        "action": {
            "en": "Remove and destroy badly infected leaves. Avoid touching wet plants. Spray a copper-based bactericide as advised by your local agriculture officer.",
            "ta": "அதிகம் பாதிக்கப்பட்ட இலைகளை அகற்றி அழிக்கவும். ஈரமான செடிகளைத் தொடுவதைத் தவிர்க்கவும். உள்ளூர் வேளாண் அலுவலரின் ஆலோசனைப்படி காப்பர் அடிப்படையிலான பாக்டீரியா எதிர்ப்பு மருந்தைத் தெளிக்கவும்.",
            "hi": "अधिक संक्रमित पत्तियाँ तोड़कर नष्ट करें। गीले पौधों को छूने से बचें। स्थानीय कृषि अधिकारी की सलाह के अनुसार कॉपर आधारित जीवाणुनाशक का छिड़काव करें।",
        },
        "prevention": {
            "en": "Use disease-free seed or seedlings, rotate crops every season, water at the base instead of overhead, and keep tools clean.",
            "ta": "நோயற்ற விதை அல்லது நாற்றுகளைப் பயன்படுத்தவும், ஒவ்வொரு பருவமும் பயிர் சுழற்சி செய்யவும், மேலிருந்து அல்லாமல் வேர்ப்பகுதியில் நீர் பாய்ச்சவும், கருவிகளைச் சுத்தமாக வைக்கவும்.",
            "hi": "रोगमुक्त बीज या पौध का उपयोग करें, हर मौसम फसल चक्र अपनाएँ, ऊपर से नहीं बल्कि जड़ के पास पानी दें और औज़ार साफ रखें।",
        },
    },
    "blight": {
        "action": {
            "en": "Remove infected leaves and destroy them away from the field. Spray a fungicide recommended by your local agriculture officer, and act quickly because blight spreads fast in humid weather.",
            "ta": "பாதிக்கப்பட்ட இலைகளை அகற்றி வயலுக்கு வெளியே அழிக்கவும். உள்ளூர் வேளாண் அலுவலர் பரிந்துரைக்கும் பூஞ்சைக்கொல்லியைத் தெளிக்கவும். ஈரப்பதமான வானிலையில் கருகல் நோய் வேகமாகப் பரவும் என்பதால் உடனே நடவடிக்கை எடுக்கவும்.",
            "hi": "संक्रमित पत्तियाँ तोड़कर खेत से दूर नष्ट करें। स्थानीय कृषि अधिकारी द्वारा सुझाई गई फफूंदनाशक दवा का छिड़काव करें। नमी वाले मौसम में झुलसा रोग तेज़ी से फैलता है, इसलिए जल्दी कदम उठाएँ।",
        },
        "prevention": {
            "en": "Rotate crops, space plants for good airflow, avoid wetting leaves when watering, remove old crop debris, and choose resistant varieties where available.",
            "ta": "பயிர் சுழற்சி செய்யவும், காற்றோட்டத்திற்காகச் செடிகளுக்கு இடைவெளி விடவும், நீர் பாய்ச்சும்போது இலைகளை நனைக்காதீர்கள், பழைய பயிர்க் கழிவுகளை அகற்றவும், கிடைத்தால் நோய் எதிர்ப்பு ரகங்களைத் தேர்ந்தெடுக்கவும்.",
            "hi": "फसल चक्र अपनाएँ, हवा के लिए पौधों के बीच दूरी रखें, सिंचाई में पत्तियों को गीला न करें, पुराने फसल अवशेष हटाएँ और उपलब्ध हो तो रोग-प्रतिरोधी किस्में चुनें।",
        },
    },
    "leaf_fungus": {
        "action": {
            "en": "Remove affected leaves and destroy them. Improve airflow around the plants and apply a fungicide recommended by your local agriculture officer if the spots keep spreading.",
            "ta": "பாதிக்கப்பட்ட இலைகளை அகற்றி அழிக்கவும். செடிகளைச் சுற்றி காற்றோட்டத்தை மேம்படுத்தவும். புள்ளிகள் தொடர்ந்து பரவினால் உள்ளூர் வேளாண் அலுவலர் பரிந்துரைக்கும் பூஞ்சைக்கொல்லியைப் பயன்படுத்தவும்.",
            "hi": "प्रभावित पत्तियाँ हटाकर नष्ट करें। पौधों के आसपास हवा का प्रवाह बढ़ाएँ और धब्बे फैलते रहें तो स्थानीय कृषि अधिकारी की सलाह से फफूंदनाशक का उपयोग करें।",
        },
        "prevention": {
            "en": "Water at the base, avoid crowding plants, clear fallen leaves, rotate crops, and keep humidity low in greenhouses or dense plantings.",
            "ta": "வேர்ப்பகுதியில் நீர் பாய்ச்சவும், செடிகளை நெருக்கமாக நடுவதைத் தவிர்க்கவும், உதிர்ந்த இலைகளை அகற்றவும், பயிர் சுழற்சி செய்யவும், பசுமைக்குடில் அல்லது அடர்ந்த நடவில் ஈரப்பதத்தைக் குறைவாக வைக்கவும்.",
            "hi": "जड़ के पास पानी दें, पौधों को घना न लगाएँ, गिरी हुई पत्तियाँ साफ करें, फसल चक्र अपनाएँ और ग्रीनहाउस या घनी रोपाई में नमी कम रखें।",
        },
    },
    "mite": {
        "action": {
            "en": "Spray the underside of leaves with water or neem oil solution to reduce mites. If the infestation is heavy, use a miticide recommended by your local agriculture officer.",
            "ta": "இலைகளின் அடிப்பகுதியில் தண்ணீர் அல்லது வேப்பெண்ணெய்க் கரைசலைத் தெளித்துச் சிலந்திப் பேன்களைக் குறைக்கவும். தாக்குதல் அதிகமாக இருந்தால் உள்ளூர் வேளாண் அலுவலர் பரிந்துரைக்கும் சிலந்திப்பேன் கொல்லியைப் பயன்படுத்தவும்.",
            "hi": "माइट कम करने के लिए पत्तियों की निचली सतह पर पानी या नीम तेल के घोल का छिड़काव करें। संक्रमण अधिक हो तो स्थानीय कृषि अधिकारी द्वारा सुझाया गया माइटनाशक उपयोग करें।",
        },
        "prevention": {
            "en": "Check the underside of leaves regularly, avoid dusty and water-stressed conditions, remove heavily infested leaves, and protect natural predators of mites.",
            "ta": "இலைகளின் அடிப்பகுதியை அடிக்கடி பரிசோதிக்கவும், தூசி மற்றும் நீர்ப் பற்றாக்குறையைத் தவிர்க்கவும், அதிகம் தாக்கப்பட்ட இலைகளை அகற்றவும், சிலந்திப் பேன்களின் இயற்கை எதிரிகளைப் பாதுகாக்கவும்.",
            "hi": "पत्तियों की निचली सतह नियमित जाँचें, धूल और पानी की कमी से बचें, अधिक प्रभावित पत्तियाँ हटाएँ और माइट के प्राकृतिक शत्रुओं की रक्षा करें।",
        },
    },
    "virus_curl": {
        "action": {
            "en": "There is no cure. Remove and destroy infected plants, and control whiteflies with yellow sticky traps, neem oil or an insecticide recommended by your local agriculture officer.",
            "ta": "இதற்குக் குணப்படுத்தும் மருந்து இல்லை. பாதிக்கப்பட்ட செடிகளை அகற்றி அழிக்கவும். மஞ்சள் ஒட்டுப் பொறிகள், வேப்பெண்ணெய் அல்லது உள்ளூர் வேளாண் அலுவலர் பரிந்துரைக்கும் பூச்சிக்கொல்லி மூலம் வெள்ளை ஈக்களைக் கட்டுப்படுத்தவும்.",
            "hi": "इसका कोई इलाज नहीं है। संक्रमित पौधे उखाड़कर नष्ट करें और पीले चिपचिपे जाल, नीम तेल या स्थानीय कृषि अधिकारी द्वारा सुझाए कीटनाशक से सफेद मक्खी नियंत्रित करें।",
        },
        "prevention": {
            "en": "Use virus-free or resistant seedlings, cover nurseries with insect-proof net, control weeds, and monitor for whiteflies early.",
            "ta": "வைரஸ் இல்லாத அல்லது எதிர்ப்புத் திறன் கொண்ட நாற்றுகளைப் பயன்படுத்தவும், நாற்றங்காலைப் பூச்சி புகாத வலையால் மூடவும், களைகளைக் கட்டுப்படுத்தவும், வெள்ளை ஈக்களை ஆரம்பத்திலேயே கண்காணிக்கவும்.",
            "hi": "वायरस-मुक्त या प्रतिरोधी पौध लगाएँ, नर्सरी को कीट-रोधी जाली से ढकें, खरपतवार नियंत्रित करें और सफेद मक्खी की शुरुआत में ही निगरानी करें।",
        },
    },
    "virus_mosaic": {
        "action": {
            "en": "There is no cure. Remove and destroy infected plants, wash hands and tools with soap after handling them, and do not touch healthy plants afterwards.",
            "ta": "இதற்குக் குணப்படுத்தும் மருந்து இல்லை. பாதிக்கப்பட்ட செடிகளை அகற்றி அழிக்கவும். அவற்றைக் கையாண்ட பிறகு கைகளையும் கருவிகளையும் சோப்பால் கழுவவும்; பிறகு ஆரோக்கியமான செடிகளைத் தொடாதீர்கள்.",
            "hi": "इसका कोई इलाज नहीं है। संक्रमित पौधे उखाड़कर नष्ट करें, उन्हें छूने के बाद हाथ और औज़ार साबुन से धोएँ और फिर स्वस्थ पौधों को न छुएँ।",
        },
        "prevention": {
            "en": "Use certified virus-free seed, disinfect tools, avoid tobacco contact before handling plants, control weeds and aphids, and choose resistant varieties.",
            "ta": "சான்றளிக்கப்பட்ட வைரஸ் இல்லாத விதைகளைப் பயன்படுத்தவும், கருவிகளைக் கிருமிநீக்கம் செய்யவும், செடிகளைத் தொடும் முன் புகையிலையைத் தொடுவதைத் தவிர்க்கவும், களைகளையும் அசுவினிகளையும் கட்டுப்படுத்தவும், எதிர்ப்புத் திறன் கொண்ட ரகங்களைத் தேர்ந்தெடுக்கவும்.",
            "hi": "प्रमाणित वायरस-मुक्त बीज उपयोग करें, औज़ार कीटाणुरहित करें, पौधों को छूने से पहले तंबाकू छूने से बचें, खरपतवार और माहू नियंत्रित करें और प्रतिरोधी किस्में चुनें।",
        },
    },
    "healthy": {
        "action": {
            "en": "No disease detected. Keep up regular watering, balanced nutrition and routine leaf checks.",
            "ta": "நோய் எதுவும் கண்டறியப்படவில்லை. வழக்கமான நீர்ப்பாசனம், சமச்சீர் உரமிடல், இலைகளை அவ்வப்போது பரிசோதித்தல் ஆகியவற்றைத் தொடரவும்.",
            "hi": "कोई रोग नहीं मिला। नियमित सिंचाई, संतुलित पोषण और पत्तियों की नियमित जाँच जारी रखें।",
        },
        "prevention": {
            "en": "Maintain good spacing and airflow, water at the base, rotate crops, and inspect plants weekly so problems are caught early.",
            "ta": "நல்ல இடைவெளியும் காற்றோட்டமும் பராமரிக்கவும், வேர்ப்பகுதியில் நீர் பாய்ச்சவும், பயிர் சுழற்சி செய்யவும், பிரச்சினைகளை முன்கூட்டியே கண்டறிய வாரந்தோறும் செடிகளைப் பரிசோதிக்கவும்.",
            "hi": "पौधों के बीच उचित दूरी और हवा का प्रवाह रखें, जड़ के पास पानी दें, फसल चक्र अपनाएँ और समस्याएँ जल्दी पकड़ने के लिए हर हफ्ते पौधों की जाँच करें।",
        },
    },
}

# Used only if a class name in class_names.json cannot be matched to the table above.
FALLBACK_ADVICE = {
    "action": {
        "en": "Consult your local agriculture officer to confirm this result and choose a suitable treatment.",
        "ta": "இந்த முடிவை உறுதிப்படுத்தவும் பொருத்தமான சிகிச்சையைத் தேர்ந்தெடுக்கவும் உள்ளூர் வேளாண் அலுவலரை அணுகவும்.",
        "hi": "इस परिणाम की पुष्टि और उचित उपचार के लिए स्थानीय कृषि अधिकारी से संपर्क करें।",
    },
    "prevention": {
        "en": "Keep plants well spaced, water at the base, remove infected leaves and rotate crops.",
        "ta": "செடிகளுக்கு இடைவெளி விடவும், வேர்ப்பகுதியில் நீர் பாய்ச்சவும், பாதிக்கப்பட்ட இலைகளை அகற்றவும், பயிர் சுழற்சி செய்யவும்.",
        "hi": "पौधों के बीच दूरी रखें, जड़ के पास पानी दें, संक्रमित पत्तियाँ हटाएँ और फसल चक्र अपनाएँ।",
    },
}

# Matches the class names in class_names.json (any spelling of underscores / case).
DISEASE_TOKENS = [
    ("healthy", ("healthy",)),
    ("bacterial_spot", ("bacterial",)),
    ("early_blight", ("earlyblight",)),
    ("late_blight", ("lateblight",)),
    ("leaf_mold", ("mold",)),
    ("septoria", ("septoria",)),
    ("spider_mites", ("spider", "mite")),
    ("target_spot", ("target",)),
    ("curl_virus", ("curl",)),
    ("mosaic_virus", ("mosaic",)),
]


def identify(class_name):
    """'Tomato_Septoria_leaf_spot' -> ('tomato', 'septoria') or (None, None) parts if unknown."""
    n = re.sub(r"[^a-z0-9]", "", class_name.lower())
    plant = next((p for p in PLANTS if p in n), None)
    disease = next((d for d, toks in DISEASE_TOKENS if any(t in n for t in toks)), None)
    return plant, disease


# ============================================================ MODEL LOADING
validator = None
disease_model = None
class_names = []
validator_scale = VALIDATOR_DIVIDE_BY_255   # final value decided in describe_validator()


def _walk_layers(model):
    for layer in getattr(model, "layers", []):
        yield layer
        yield from _walk_layers(layer)


def describe_validator(model):
    """Print what the validator really contains, and avoid double preprocessing."""
    global validator_scale
    last = model.layers[-1]
    act = getattr(getattr(last, "activation", None), "__name__", "unknown")
    print("=== VALIDATOR MODEL ===")
    print(f"input_shape={model.input_shape} output_shape={model.output_shape} last_layer_activation={act}")
    pre = [l.name for l in _walk_layers(model)
           if type(l).__name__ in ("Rescaling", "Normalization", "TrueDivide", "Subtract")
           or "preprocess" in l.name.lower()]
    validator_scale = VALIDATOR_DIVIDE_BY_255
    if pre and validator_scale:
        validator_scale = False
        print(f"[WARN] Validator already contains preprocessing layers {pre}; NOT dividing by 255 again.")
    print(f"[INFO] Validator built-in preprocessing layers: {pre}")
    print(f"[INFO] Validator input scaling: divide_by_255={validator_scale} (False = raw 0-255 pixels)")
    print(f"[INFO] Single-output meaning = P({VALIDATOR_OUTPUT_MEANS}), threshold={LEAF_THRESHOLD}")


def load_assets():
    global validator, disease_model, class_names
    from tensorflow.keras.models import load_model

    for label, path in (("Leaf validator", VALIDATOR_PATH), ("Disease model", DISEASE_PATH)):
        if not os.path.exists(path):
            print(f"[ERROR] {label} not found: {path}")
    if os.path.exists(VALIDATOR_PATH):
        validator = load_model(VALIDATOR_PATH)
        describe_validator(validator)
    if os.path.exists(DISEASE_PATH):
        disease_model = load_model(DISEASE_PATH)

    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):  # {"0": "name", "1": "name", ...}
            data = [name for _, name in sorted(data.items(), key=lambda kv: int(kv[0]))]
        class_names = list(data)
    else:
        print(f"[ERROR] class_names.json not found: {CLASSES_PATH}")

    if disease_model is not None and class_names:
        outputs = disease_model.output_shape[-1]
        if outputs != len(class_names):
            print(f"[WARN] leaf_model outputs {outputs} classes but class_names.json has {len(class_names)}")
        for name in class_names:
            if None in identify(name):
                print(f"[WARN] Class not in knowledge base, generic advice will be used: {name}")


# ============================================================ HELPERS
def model_size(model):
    shape = getattr(model, "input_shape", None)
    try:
        h, w = int(shape[1]), int(shape[2])
        return (w, h)
    except (TypeError, ValueError, IndexError):
        return DEFAULT_SIZE


def preprocess(image, model, divide):
    img = image.resize(model_size(model))
    arr = np.asarray(img, dtype="float32")
    if divide:
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


def leaf_score_from(output):
    """Raw validator output -> probability that the image is a LEAF (0..1)."""
    out = np.ravel(output).astype("float64")
    if out.size == 1:
        p = float(out[0])
        if not 0.0 <= p <= 1.0:            # raw logit instead of a sigmoid probability
            p = 1.0 / (1.0 + np.exp(-p))
        return (p if VALIDATOR_OUTPUT_MEANS == "leaf" else 1.0 - p), out
    return float(out[LEAF_CLASS_INDEX]), out


def readable(name):
    return re.sub(r"\s+", " ", name.replace("_", " ")).strip()


def error(message, status):
    return jsonify({"error": message}), status


# ============================================================ ROUTES
@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "validator_loaded": validator is not None,
        "disease_model_loaded": disease_model is not None,
        "classes": len(class_names),
    })


@app.post("/predict")
def predict():
    lang = request.form.get("lang", "en")
    if lang not in LANGS:
        lang = "en"

    if validator is None or disease_model is None or not class_names:
        return error("Models are not loaded on the server.", 503)

    file = request.files.get("leaf")
    if file is None or file.filename == "":
        return error("No image received in field 'leaf'.", 400)

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return error("Unsupported or corrupted image.", 400)

    # 1) Leaf validation - the disease model must NEVER see a non-leaf image
    raw = validator.predict(preprocess(image, validator, validator_scale), verbose=0)
    leaf_score, out = leaf_score_from(raw)
    is_leaf = leaf_score >= LEAF_THRESHOLD
    debug_info = {
        "raw_output": out.tolist(),
        "leaf_score": round(leaf_score, 4),
        "threshold": LEAF_THRESHOLD,
        "decision": "LEAF" if is_leaf else "NON-LEAF",
    }
    if DEBUG_VALIDATOR:
        print("VALIDATOR DEBUG:")
        print(f"  raw_output = {out.tolist()}")
        print(f"  leaf_score = {leaf_score:.4f} (threshold {LEAF_THRESHOLD})")
        print(f"  decision   = {'LEAF' if is_leaf else 'NON-LEAF'}")
    if not is_leaf:
        body = {"valid_leaf": False, "message": "This image does not appear to be a plant leaf."}
        if request.values.get("debug") == "1":
            body["validator_debug"] = debug_info
        return jsonify(body)

    # 2) Disease prediction (15 classes)
    probs = np.ravel(
        disease_model.predict(preprocess(image, disease_model, DISEASE_DIVIDE_BY_255), verbose=0)
    )
    idx = int(np.argmax(probs))
    if idx >= len(class_names):
        return error("class_names.json does not match leaf_model.keras.", 500)

    name = class_names[idx]
    plant_id, disease_id = identify(name)
    if plant_id and disease_id:
        category, disease_names = DISEASES[disease_id]
        plant = PLANTS[plant_id][lang]
        disease = disease_names[lang]
        action = ADVICE[category]["action"][lang]
        prevention = ADVICE[category]["prevention"][lang]
    else:
        plant, disease = readable(name), readable(name)
        action = FALLBACK_ADVICE["action"][lang]
        prevention = FALLBACK_ADVICE["prevention"][lang]

    body = {
        "valid_leaf": True,
        "plant": plant,
        "disease": disease,
        "confidence": round(float(probs[idx]) * 100, 2),
        "action": action,
        "prevention": prevention,
    }
    if request.values.get("debug") == "1":
        body["validator_debug"] = debug_info
    return jsonify(body)


@app.errorhandler(413)
def too_large(_):
    return error(f"Image too large. Maximum size is {MAX_UPLOAD_MB} MB.", 413)


# ============================================================ RUN
load_assets()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
