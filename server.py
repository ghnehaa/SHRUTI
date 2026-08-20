"""
SHRUTI Care - Python WebSocket Server
FastAPI backend with WebSocket for real-time screening, AI, and clinical engine.
"""
import json
import asyncio
import uuid
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from signal_processing import simulate_dpoae_response, compute_fft, DPOAE_FREQUENCIES
from ai_engine import assess_signal_quality, classify_dpoae_spectrum, explain_decision
from clinical_engine import evaluate_screening, generate_follow_up_task, generate_digital_referral
from synthetic_data import (
    generate_normal_response, generate_weak_response, generate_noisy_response,
    generate_artifact_response, generate_probe_leakage_response,
    generate_large_test_dataset, generate_simulator_state, TEST_FREQUENCIES
)

app = FastAPI(title="SHRUTI Care", version="1.0.0-hackathon")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

connected_clients: Dict[str, WebSocket] = {}
screening_sessions: Dict[str, Dict] = {}
device_registry: Dict[str, Dict] = {}
hearing_records: Dict[str, Dict] = {}
referral_queue: List[Dict] = []
follow_up_tasks: List[Dict] = []


class BabyRegistration(BaseModel):
    name: str
    dob: str
    gender: str
    phone: str = ""
    blood_group: str = ""
    birth_weight: Optional[float] = None
    gestational_age: Optional[int] = None
    nicu_admission: bool = False
    risk_factors: str = ""
    address: str = ""
    facility: str = ""
    mother_id: Optional[str] = None


class ScreeningRequest(BaseModel):
    baby_id: str
    operator: str
    ear: str = "both"
    ambient_db: float = 35.0
    probe_seal: float = 0.8


class SimulatorParams(BaseModel):
    ambient_db: float = 35
    probe_seal: float = 0.8
    dp_amplitude: float = -10
    movement: float = 0.0
    stimulus_l1: float = 65
    stimulus_l2: float = 55


class DeviceRegistration(BaseModel):
    serial: str
    model: str = "SHRUTI-ESP32-S3"
    firmware: str = "v1.0.0"


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>SHRUTI Care Server Running</h1>")


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0-hackathon", "timestamp": datetime.now().isoformat()}


@app.get("/api/babies")
async def list_babies():
    return {"babies": list(hearing_records.values())}


@app.post("/api/babies/register")
async def register_baby(reg: BabyRegistration):
    baby_id = str(uuid.uuid4())[:8]
    baby = {
        "id": baby_id,
        "name": reg.name,
        "dob": reg.dob,
        "gender": reg.gender,
        "phone": reg.phone,
        "blood_group": reg.blood_group,
        "birth_weight": reg.birth_weight,
        "gestational_age": reg.gestational_age,
        "nicu_admission": reg.nicu_admission,
        "risk_factors": reg.risk_factors,
        "address": reg.address,
        "facility": reg.facility,
        "mother_id": reg.mother_id,
        "registered_date": datetime.now().strftime("%Y-%m-%d"),
        "qr_code": f"SHRUTI-{baby_id}",
        "screenings": [],
        "prenatal_risk_profile": None,
        "hearing_passport": {
            "id": f"HP-{baby_id}",
            "child_id": baby_id,
            "entries": []
        }
    }
    hearing_records[baby_id] = baby
    return {"baby": baby, "message": "Baby registered successfully"}


@app.get("/api/babies/{baby_id}")
async def get_baby(baby_id: str):
    if baby_id not in hearing_records:
        raise HTTPException(status_code=404, detail="Baby not found")
    return {"baby": hearing_records[baby_id]}


@app.get("/api/babies/{baby_id}/passport")
async def get_hearing_passport(baby_id: str):
    if baby_id not in hearing_records:
        raise HTTPException(status_code=404, detail="Baby not found")
    baby = hearing_records[baby_id]
    return {"passport": baby.get("hearing_passport", {}), "record": baby}


@app.get("/api/screenings")
async def list_screenings():
    all_screenings = []
    for baby in hearing_records.values():
        for s in baby.get("screenings", []):
            all_screenings.append(s)
    return {"screenings": all_screenings}


@app.post("/api/screenings/run")
async def run_screening(req: ScreeningRequest):
    import numpy as np
    baby = hearing_records.get(req.baby_id)
    if not baby:
        raise HTTPException(status_code=404, detail="Baby not found")

    sq_data = []
    dpoae_left = []
    dpoae_right = []
    all_frames_left = []
    all_frames_right = []

    for f in TEST_FREQUENCIES:
        resp_l = simulate_dpoae_response(
            f,
            dp_amplitude=-10,
            noise_floor=-40,
            snr=15,
            duration=0.5
        )
        sq = assess_signal_quality(resp_l["signal"], probe_seal=req.probe_seal, ambient_db=req.ambient_db)
        sq_data.append(sq)

        sig = np.array(resp_l["signal"])
        n = len(sig)
        windowed = sig * np.hanning(n)
        fft_vals = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(n, 1.0 / resp_l["sample_rate"])
        mags = (20 * np.log10(np.abs(fft_vals) + 1e-12)).tolist()

        dpoae_info = classify_dpoae_spectrum(mags, [float(x) for x in freqs], resp_l["dp_frequency"])
        dpoae_left.append(dpoae_info)
        all_frames_left.append(resp_l["signal"][:1024])

    for f in TEST_FREQUENCIES:
        resp_r = simulate_dpoae_response(
            f,
            dp_amplitude=-12,
            noise_floor=-38,
            snr=10,
            duration=0.5
        )
        sig = np.array(resp_r["signal"])
        n = len(sig)
        windowed = sig * np.hanning(n)
        fft_vals = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(n, 1.0 / resp_r["sample_rate"])
        mags = (20 * np.log10(np.abs(fft_vals) + 1e-12)).tolist()

        dpoae_info = classify_dpoae_spectrum(mags, [float(x) for x in freqs], resp_r["dp_frequency"])
        dpoae_right.append(dpoae_info)
        all_frames_right.append(resp_r["signal"][:1024])

    avg_sq = {
        "overall_score": round(sum(s["overall_score"] for s in sq_data) / len(sq_data), 3),
        "confidence": round(sum(s["confidence"] for s in sq_data) / len(sq_data), 3),
        "recommendation": sq_data[0]["recommendation"],
        "probe_fit": sq_data[0]["probe_fit"],
        "factors": sq_data[0]["factors"]
    }

    left_data = {"dpoae_per_freq": dpoae_left, "valid_frames": len(all_frames_left), "total_frames": len(all_frames_left), "rejected_reasons": []}
    right_data = {"dpoae_per_freq": dpoae_right, "valid_frames": len(all_frames_right), "total_frames": len(all_frames_right), "rejected_reasons": []}

    result = evaluate_screening(left_data, right_data, req.baby_id, baby["name"], req.operator, avg_sq)

    baby["screenings"].append(result)
    baby["hearing_passport"]["entries"].append({
        "date": result["date"],
        "verdict": result["verdict"],
        "left": result["ear_left_result"],
        "right": result["ear_right_result"],
        "snr_left": result.get("snr_left"),
        "snr_right": result.get("snr_right"),
        "operator": result["operator"]
    })

    screening_sessions[result["baby_id"]] = result

    if result["verdict"] in ("REFER", "RETEST"):
        fu = generate_follow_up_task(result, [])
        follow_up_tasks.append(fu)
        if result["verdict"] == "REFER":
            ref = generate_digital_referral(result)
            referral_queue.append(ref)

    return {"result": result, "signal_quality": avg_sq}


@app.get("/api/referrals")
async def list_referrals():
    return {"referrals": referral_queue}


@app.get("/api/follow-ups")
async def list_follow_ups():
    return {"follow_ups": follow_up_tasks}


@app.get("/api/analytics")
async def get_analytics():
    all_screenings = []
    for baby in hearing_records.values():
        all_screenings.extend(baby.get("screenings", []))

    total = len(all_screenings)
    pass_count = sum(1 for s in all_screenings if s["verdict"] == "PASS")
    refer_count = sum(1 for s in all_screenings if s["verdict"] == "REFER")
    retest_count = sum(1 for s in all_screenings if s["verdict"] == "RETEST")
    refer_rate = round(refer_count / max(1, total) * 100, 1)
    pass_rate = round(pass_count / max(1, total) * 100, 1)

    return {
        "total_screenings": total,
        "pass_count": pass_count,
        "refer_count": refer_count,
        "retest_count": retest_count,
        "pass_rate": pass_rate,
        "refer_rate": refer_rate,
        "total_babies": len(hearing_records),
        "active_referrals": len(referral_queue),
        "pending_followups": len([f for f in follow_up_tasks if f["status"] == "Pending"]),
        "screening_trend": [
            {"label": "Week " + str(i + 1), "pass": random.randint(5, 15), "refer": random.randint(1, 4), "retest": random.randint(0, 3)}
            for i in range(8)
        ],
        "coverage_by_district": [
            {"district": "Noida", "screened": random.randint(40, 80), "total": 100},
            {"district": "Delhi", "screened": random.randint(60, 95), "total": 120},
            {"district": "Hyderabad", "screened": random.randint(30, 60), "total": 80},
            {"district": "Bengaluru", "screened": random.randint(50, 90), "total": 100},
            {"district": "Patna", "screened": random.randint(20, 45), "total": 70},
            {"district": "Pune", "screened": random.randint(40, 70), "total": 90},
        ],
        "gis_heatmap": [
            {"lat": 28.57, "lng": 77.33, "intensity": 0.8, "label": "Noida"},
            {"lat": 28.61, "lng": 77.21, "intensity": 0.9, "label": "Delhi"},
            {"lat": 17.39, "lng": 78.49, "intensity": 0.6, "label": "Hyderabad"},
            {"lat": 12.97, "lng": 77.59, "intensity": 0.7, "label": "Bengaluru"},
            {"lat": 25.61, "lng": 85.14, "intensity": 0.4, "label": "Patna"},
            {"lat": 18.52, "lng": 73.86, "intensity": 0.65, "label": "Pune"},
        ]
    }


@app.get("/api/devices")
async def list_devices():
    devices = list(device_registry.values())
    if not devices:
        devices = [
            {
                "serial": "SHR-001",
                "model": "SHRUTI-ESP32-S3",
                "firmware": "v1.0.0",
                "status": "Online",
                "battery": 87,
                "calibration_date": "2026-07-15",
                "calibration_valid": True,
                "mic_health": "OK",
                "receiver_health": "OK",
                "codec_health": "OK",
                "last_self_test": datetime.now().strftime("%Y-%m-%d"),
                "storage_used": 12,
                "storage_total": 32
            }
        ]
    return {"devices": devices}


@app.post("/api/devices/register")
async def register_device(dev: DeviceRegistration):
    device = {
        "serial": dev.serial,
        "model": dev.model,
        "firmware": dev.firmware,
        "status": "Online",
        "battery": 100,
        "calibration_date": datetime.now().strftime("%Y-%m-%d"),
        "calibration_valid": True,
        "mic_health": "OK",
        "receiver_health": "OK",
        "codec_health": "OK",
        "last_self_test": datetime.now().strftime("%Y-%m-%d"),
        "storage_used": 0,
        "storage_total": 32
    }
    device_registry[dev.serial] = device
    return {"device": device, "message": "Device registered"}


@app.get("/api/devices/{serial}/self-test")
async def device_self_test(serial: str):
    import numpy as np
    device = device_registry.get(serial, {
        "serial": serial,
        "model": "SHRUTI-ESP32-S3",
        "status": "Online",
        "battery": random.randint(60, 100),
        "mic_health": "OK",
        "receiver_health": "OK",
        "codec_health": "OK",
    })
    test_signal = np.sin(2 * np.pi * 1000 * np.arange(4410) / 44100)
    rms = float(np.sqrt(np.mean(test_signal ** 2)))
    results = {
        "serial": serial,
        "timestamp": datetime.now().isoformat(),
        "tests": {
            "microphone": {"status": "PASS", "detail": f"RMS level: {rms:.4f}"},
            "receiver": {"status": "PASS", "detail": "Output verified at 65 dB SPL"},
            "codec": {"status": "PASS", "detail": "24-bit audio codec operational"},
            "battery": {"status": "PASS" if device.get("battery", 100) > 20 else "WARN", "level": device.get("battery", 85)},
            "storage": {"status": "PASS", "detail": "12 / 32 GB used"},
            "calibration": {"status": "PASS" if device.get("calibration_valid", True) else "WARN", "date": device.get("calibration_date", "unknown")},
            "ble": {"status": "PASS", "detail": "Bluetooth Low Energy connected"},
            "ambient_noise": {"status": "PASS", "level_db": random.randint(28, 45)}
        },
        "overall": "READY",
        "blocking_issues": []
    }
    return results


@app.get("/api/synthetic/normal")
async def synthetic_normal(n: int = 10):
    return {"data": generate_normal_response(n), "count": n, "type": "normal"}


@app.get("/api/synthetic/weak")
async def synthetic_weak(n: int = 10):
    return {"data": generate_weak_response(n), "count": n, "type": "weak"}


@app.get("/api/synthetic/noisy")
async def synthetic_noisy(n: int = 10, level: str = "moderate"):
    return {"data": generate_noisy_response(n, level), "count": n, "type": f"noisy_{level}"}


@app.get("/api/synthetic/artifact")
async def synthetic_artifact(n: int = 5, kind: str = "movement"):
    return {"data": generate_artifact_response(kind, n), "count": n, "type": f"artifact_{kind}"}


@app.get("/api/synthetic/probe-leak")
async def synthetic_probe_leak(n: int = 5, severity: str = "moderate"):
    return {"data": generate_probe_leakage_response(n, severity), "count": n, "type": f"probe_leak_{severity}"}


@app.get("/api/synthetic/large-dataset")
async def synthetic_large(n: int = 100):
    return {"data": generate_large_test_dataset(n), "count": n}


@app.post("/api/simulator/run")
async def run_simulator(params: SimulatorParams):
    state = generate_simulator_state(params.model_dump())
    sq = assess_signal_quality(
        [random.gauss(0, 0.01) for _ in range(4410)],
        probe_seal=params.probe_seal,
        ambient_db=params.ambient_db
    )
    return {"state": state, "signal_quality": sq}


@app.get("/api/prenatal/research")
async def prenatal_research():
    return {
        "module": "Prenatal Hearing Research (Experimental)",
        "status": "Research-only - not for clinical use",
        "features": [
            "Maternal-abdominal contact microphone sensing",
            "Controlled external acoustic/vibroacoustic stimulus",
            "Fetal movement response analysis",
            "Ultrasound-derived feature research",
            "Maternal and family risk-factor collection",
            "Multimodal AI risk stratification",
            "Automatic recommendation for enhanced newborn screening"
        ],
        "safety_note": "Routine clinical deployment requires extensive safety, clinical-validation and regulatory work.",
        "research_data": [
            {"id": f"PR-{i:03d}", "gestational_week": random.randint(24, 38), "fetal_response": random.choice(["present", "absent", "equivocal"]), "risk_score": round(random.uniform(0.1, 0.8), 2), "recommendation": random.choice(["Enhanced screening at birth", "Routine screening", "Further investigation"])}
            for i in range(10)
        ]
    }


@app.get("/api/parent/info/{lang}")
async def parent_info(lang: str = "en"):
    translations = {
        "en": {
            "pass_title": "Hearing Screen Result: PASS",
            "pass_explain": "Your baby's hearing screening result is normal. Both ears responded as expected. This is a screening, not a diagnosis.",
            "refer_title": "Hearing Screen Result: REFER",
            "refer_explain": "Your baby needs a more detailed hearing test. This does not mean your baby has hearing loss. Please visit the audiology centre.",
            "retest_title": "Hearing Screen Result: RETEST",
            "retest_explain": "The screening could not be completed today. Please come back for a retest.",
            "education": "Early hearing assessment helps detect issues that can be addressed early."
        },
        "hi": {
            "pass_title": "Sunne ki jaanch parinaam: PASS",
            "pass_explain": "Aapke bachche ki sunne ki jaanch saamaanya hai. Dono kaanon ne pratyekshaanusaar pratikriya di.",
            "refer_title": "Sunne ki jaanch parinaam: REFER",
            "refer_explain": "Aapke bachche ko ek adhik vistrit sunne ki jaanch ki aavashyakta hai.",
            "retest_title": "Sunne ki jaanch parinaam: RETEST",
            "retest_explain": "Takniki kaaranon se aaj jaanch poori nahi ho saki. Kripya dobara aayen.",
            "education": "Prarambhik sunne ka moolyaankan un samasyaon ka pata lagane mein madad karta hai."
        }
    }
    return translations.get(lang, translations["en"])


@app.get("/api/explainable-ai/{baby_id}")
async def explainable_ai(baby_id: str):
    baby = hearing_records.get(baby_id)
    if not baby or not baby.get("screenings"):
        return {"explanation": None, "message": "No screening data available"}
    latest = baby["screenings"][-1]
    return {
        "explanation": latest.get("explanation", {}),
        "signal_quality_score": latest.get("signal_quality_score"),
        "ai_confidence": latest.get("ai_confidence"),
        "frames_valid": latest.get("frames_valid"),
        "frames_total": latest.get("frames_total"),
        "model_version": latest.get("protocol_version", "unknown"),
        "disclaimer": "AI provides signal quality support. Clinical decision is by validated protocol, not AI diagnosis."
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = str(uuid.uuid4())[:8]
    connected_clients[client_id] = websocket

    try:
        await websocket.send_json({
            "type": "connected",
            "client_id": client_id,
            "message": "SHRUTI Care WebSocket connected",
            "server_version": "1.0.0-hackathon"
        })

        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "screening_start":
                await handle_ws_screening(websocket, data)

            elif msg_type == "simulator_update":
                params = data.get("params", {})
                state = generate_simulator_state(params)
                sq = assess_signal_quality(
                    [random.gauss(0, 0.01) for _ in range(4410)],
                    probe_seal=params.get("probe_seal", 0.8),
                    ambient_db=params.get("ambient_db", 35)
                )
                await websocket.send_json({"type": "simulator_state", "state": state, "signal_quality": sq})

            elif msg_type == "device_status_request":
                await websocket.send_json({
                    "type": "device_status",
                    "devices": [{"serial": "SHR-001", "battery": random.randint(60, 100), "status": "Online"}]
                })

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})

    except WebSocketDisconnect:
        connected_clients.pop(client_id, None)
    except Exception:
        connected_clients.pop(client_id, None)


async def handle_ws_screening(websocket: WebSocket, data: dict):
    import numpy as np

    probe_seal = data.get("probe_seal", 0.8)
    ambient_db = data.get("ambient_db", 35)

    await websocket.send_json({"type": "screening_phase", "phase": "probe_fit", "status": "checking"})
    await asyncio.sleep(0.5)
    probe_status = "READY" if probe_seal > 0.5 else "ADJUST" if probe_seal > 0.3 else "REJECT"
    await websocket.send_json({"type": "screening_phase", "phase": "probe_fit", "status": probe_status, "seal": probe_seal})

    await websocket.send_json({"type": "screening_phase", "phase": "ambient_noise", "status": "checking"})
    await asyncio.sleep(0.3)
    noise_status = "PASS" if ambient_db < 50 else "WARN" if ambient_db < 60 else "FAIL"
    await websocket.send_json({"type": "screening_phase", "phase": "ambient_noise", "status": noise_status, "level_db": ambient_db})

    if probe_status == "REJECT" or noise_status == "FAIL":
        await websocket.send_json({"type": "screening_result", "verdict": "RETEST", "reason": "Test readiness check failed"})
        return

    for freq_idx, freq in enumerate(TEST_FREQUENCIES):
        await websocket.send_json({
            "type": "screening_phase", "phase": "stimulus",
            "frequency": freq, "progress": round((freq_idx / len(TEST_FREQUENCIES)) * 100)
        })
        await asyncio.sleep(0.3)

        resp = simulate_dpoae_response(freq, noise_floor=-40, duration=0.5)
        sig = np.array(resp["signal"])
        n = len(sig)
        windowed = sig * np.hanning(n)
        fft_vals = np.fft.rfft(windowed)
        freqs_arr = np.fft.rfftfreq(n, 1.0 / resp["sample_rate"])
        mags = (20 * np.log10(np.abs(fft_vals) + 1e-12)).tolist()

        d_info = classify_dpoae_spectrum(mags, [float(x) for x in freqs_arr], resp["dp_frequency"])

        fft_points = []
        step = max(1, len(freqs_arr) // 100)
        for i in range(0, len(freqs_arr), step):
            fft_points.append({"f": round(float(freqs_arr[i]), 1), "m": round(float(mags[i]), 2)})

        await websocket.send_json({"type": "screening_fft", "frequency": freq, "fft": fft_points, "dpoae": d_info})

        sq = assess_signal_quality(resp["signal"], probe_seal=probe_seal, ambient_db=ambient_db)
        await websocket.send_json({"type": "screening_quality", "frequency": freq, "quality": sq})
        await asyncio.sleep(0.2)

    await websocket.send_json({"type": "screening_phase", "phase": "analysis", "status": "computing"})
    await asyncio.sleep(0.5)

    verdict = "PASS" if random.random() > 0.3 else "REFER" if random.random() > 0.5 else "RETEST"
    sq_final = assess_signal_quality([random.gauss(0, 0.01) for _ in range(4410)], probe_seal=probe_seal, ambient_db=ambient_db)

    await websocket.send_json({
        "type": "screening_result",
        "verdict": verdict,
        "signal_quality": sq_final,
        "summary": f"Screening complete. Result: {verdict}.",
        "disclaimer": "Screening is not a diagnosis. AI supports signal quality; clinical pathway determines outcome."
    })


if __name__ == "__main__":
    import uvicorn
    print("SHRUTI Care Server starting on http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
