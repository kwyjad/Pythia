import hashlib, json

def _norm(s: str) -> str:
    return " ".join(str(s).strip().split()).lower()

def digest(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, (dict, list)):
            p = json.dumps(p, sort_keys=True, separators=(",",":"))
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()

def scenario_id(iso3, hazard_code, title, md_json) -> str:
    return digest("scenario", _norm(iso3), _norm(hazard_code), _norm(title), md_json)

def question_id(iso3, hazard_code, metric, target_month, wording) -> str:
    return digest("question", _norm(iso3), _norm(hazard_code), _norm(metric), _norm(target_month), _norm(wording))

def forecast_id(question_id, model_name, horizon_m, run_id) -> str:
    return digest("forecast", question_id, model_name, horizon_m, run_id)
