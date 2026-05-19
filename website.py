from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote
import json
import mimetypes

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "house_price_model.pkl"
METRICS_PATH = ROOT / "models" / "metrics.pkl"
RESULT_PLOT_PATH = ROOT / "result_plot.png"
TEMPLATE_PATH = ROOT / "templates" / "index.html"
STATIC_DIR = ROOT / "static"

BASE_USD_RATE = 450
MARKET_SCALE = 0.5
QYZYLORDA_SCALE = 0.4
LAND_PRICE_PER_SOTKA = 1_500_000

DISTRICTS = {
    "Центр": {"multiplier": 1.35, "houses": False, "apartments": True, "note": "деловая часть города"},
    "Сырдария": {"multiplier": 1.28, "houses": False, "apartments": True, "note": "высокий спрос на квартиры"},
    "ЖК Мерей": {"multiplier": 1.30, "houses": False, "apartments": True, "note": "новая застройка"},
    "Левый берег": {"multiplier": 1.32, "houses": False, "apartments": True, "note": "перспективная локация"},
    "Шұғыла": {"multiplier": 1.18, "houses": True, "apartments": True, "note": "смешанный жилой район"},
    "Байтерек": {"multiplier": 1.10, "houses": False, "apartments": True, "note": "спокойный микрорайон"},
    "Универсам": {"multiplier": 1.12, "houses": False, "apartments": True, "note": "удобная инфраструктура"},
    "Арай": {"multiplier": 1.15, "houses": True, "apartments": True, "note": "подходит для частных домов"},
    "Акмаржан": {"multiplier": 1.08, "houses": False, "apartments": True, "note": "средний ценовой уровень"},
    "Саулет": {"multiplier": 0.98, "houses": False, "apartments": True, "note": "доступнее среднего рынка"},
    "Мерей": {"multiplier": 1.05, "houses": False, "apartments": True, "note": "стабильный спрос"},
    "Титов": {"multiplier": 0.85, "houses": True, "apartments": True, "note": "самый доступный коэффициент"},
}

MATERIALS = {
    "Кирпич": 1.15,
    "Панель": 0.95,
    "Бетон": 1.10,
}

REPAIRS = {
    "Черновой": 0.80,
    "Средний": 1.00,
    "Евроремонт": 1.30,
}

FLOORS = {
    1: 0.90,
    2: 1.10,
    3: 1.15,
    4: 1.08,
    5: 0.85,
}

MODEL = joblib.load(MODEL_PATH)


def load_metrics():
    if METRICS_PATH.exists():
        return joblib.load(METRICS_PATH)
    return {"r2": 0.812}


def money(value):
    return f"{int(round(value)):,}".replace(",", " ") + " ₸"


def million(value):
    return f"{value / 1_000_000:.1f} млн ₸"


def to_bool(value):
    return value in (True, "true", "1", 1, "on")


def normalize_payload(payload):
    object_type = payload.get("objectType", "apartment")
    is_house = object_type == "house"
    district = payload.get("district") or "Центр"
    district_data = DISTRICTS.get(district, DISTRICTS["Центр"])

    if is_house and not district_data["houses"]:
        district = next(name for name, item in DISTRICTS.items() if item["houses"])
    if not is_house and not district_data["apartments"]:
        district = next(name for name, item in DISTRICTS.items() if item["apartments"])

    return {
        "object_type": object_type,
        "is_house": is_house,
        "district": district,
        "area": max(20.0, min(float(payload.get("area", 85)), 500.0)),
        "rooms": max(1, min(int(payload.get("rooms", 3)), 10)),
        "age": max(1, min(int(payload.get("age", 20)), 60)),
        "floor": max(1, min(int(payload.get("floor", 3)), 5)),
        "land": max(1, min(int(payload.get("land", 6)), 20)),
        "material": payload.get("material", "Кирпич") if payload.get("material") in MATERIALS else "Кирпич",
        "repair": payload.get("repair", "Средний") if payload.get("repair") in REPAIRS else "Средний",
        "income": max(100_000.0, min(float(payload.get("income", 500_000)), 2_500_000.0)),
        "usd_rate": max(400, min(int(payload.get("usdRate", 480)), 850)),
        "school": to_bool(payload.get("school", True)),
        "shop": to_bool(payload.get("shop", True)),
        "park": to_bool(payload.get("park", False)),
    }


def build_features(params):
    med_inc = (params["income"] * 12) / BASE_USD_RATE / 10_000
    return pd.DataFrame(
        {
            "MedInc": [med_inc],
            "HouseAge": [params["age"]],
            "AveRooms": [max(params["area"] / 25, params["rooms"])],
            "AveBedrms": [max(1.0, params["rooms"] / 2.4)],
            "Population": [1500],
            "AveOccup": [3.5],
            "Latitude": [34.0],
            "Longitude": [-118.0],
        }
    )


def calculate_price(params, district, usd_rate):
    district_data = DISTRICTS[district]
    raw_prediction = MODEL.predict(build_features(params))[0]
    floor_multiplier = 1.0 if params["is_house"] else FLOORS[params["floor"]]
    infra_multiplier = 1.0

    if params["school"]:
        infra_multiplier += 0.02
    if params["shop"]:
        infra_multiplier += 0.01
    if params["park"]:
        infra_multiplier += 0.03

    price = (
        raw_prediction
        * 100_000
        * usd_rate
        * MARKET_SCALE
        * QYZYLORDA_SCALE
        * district_data["multiplier"]
        * MATERIALS[params["material"]]
        * REPAIRS[params["repair"]]
        * floor_multiplier
        * infra_multiplier
    )

    if params["is_house"]:
        price += params["land"] * LAND_PRICE_PER_SOTKA

    return {
        "price": int(price),
        "raw_prediction": float(raw_prediction),
        "district_multiplier": district_data["multiplier"],
        "floor_multiplier": floor_multiplier,
        "infra_multiplier": infra_multiplier,
    }


def available_districts(is_house):
    key = "houses" if is_house else "apartments"
    return [
        {
            "name": name,
            "multiplier": data["multiplier"],
            "note": data["note"],
        }
        for name, data in DISTRICTS.items()
        if data[key]
    ]


def prediction_response(payload):
    params = normalize_payload(payload)
    current = calculate_price(params, params["district"], BASE_USD_RATE)
    scenario = calculate_price(params, params["district"], params["usd_rate"])
    district_rows = []

    for district in available_districts(params["is_house"]):
        row_result = calculate_price(params, district["name"], params["usd_rate"])
        district_rows.append(
            {
                "name": district["name"],
                "note": district["note"],
                "multiplier": district["multiplier"],
                "price": row_result["price"],
                "priceText": million(row_result["price"]),
                "meterText": money(row_result["price"] / params["area"]),
            }
        )

    district_rows.sort(key=lambda item: item["price"], reverse=True)
    delta = scenario["price"] - current["price"]
    delta_percent = ((scenario["price"] / current["price"]) - 1) * 100 if current["price"] else 0

    return {
        "params": {
            "district": params["district"],
            "objectType": params["object_type"],
            "area": params["area"],
            "rooms": params["rooms"],
            "usdRate": params["usd_rate"],
        },
        "currentPrice": current["price"],
        "currentPriceText": money(current["price"]),
        "scenarioPrice": scenario["price"],
        "scenarioPriceText": money(scenario["price"]),
        "pricePerMeterText": money(scenario["price"] / params["area"]),
        "deltaText": f"{'+' if delta >= 0 else '-'}{money(abs(delta))}",
        "deltaPercent": round(delta_percent, 1),
        "rawPrediction": round(scenario["raw_prediction"], 3),
        "districtMultiplier": scenario["district_multiplier"],
        "floorMultiplier": scenario["floor_multiplier"],
        "infraMultiplier": scenario["infra_multiplier"],
        "materialMultiplier": MATERIALS[params["material"]],
        "repairMultiplier": REPAIRS[params["repair"]],
        "landPriceText": money(params["land"] * LAND_PRICE_PER_SOTKA) if params["is_house"] else "не применяется",
        "districts": district_rows,
        "availableDistricts": available_districts(params["is_house"]),
    }


def render_index():
    r2 = load_metrics().get("r2", 0.812) * 100
    return TEMPLATE_PATH.read_text(encoding="utf-8").replace("__R2_SCORE__", f"{r2:.1f}")


class WebsiteHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = unquote(self.path.split("?", 1)[0])
        if path == "/":
            self.send_text(render_index(), "text/html; charset=utf-8")
        elif path == "/result-plot.png" and RESULT_PLOT_PATH.exists():
            self.send_file(RESULT_PLOT_PATH, "image/png")
        elif path.startswith("/static/"):
            self.serve_static(path.removeprefix("/static/"))
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/api/predict":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length).decode("utf-8") if length else "{}"

        try:
            payload = json.loads(raw_body)
            self.send_json(prediction_response(payload))
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)

    def serve_static(self, relative_path):
        target = (STATIC_DIR / relative_path).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.exists():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_file(target, content_type)

    def send_text(self, content, content_type, status=200):
        body = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, content_type):
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 5000), WebsiteHandler)
    print("Сайт запущен: http://127.0.0.1:5000")
    server.serve_forever()
