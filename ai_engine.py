t classification, signal-quality estimation, confidence assessment.
"""
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

MODEL_VERSION = "shruti-ai-v1.0-hackathon"
MODEL_DATE = "2026-08-18"


@dataclass
class SignalQualityReport:
    overall_score: float
    confidence: float
    probe_fit: str
    ambient_noise: str
    artifact_detected: str
    recommendation: str
    factors: List[Dict]
    model_version: str


def assess_signal_quality(audio_data: List[float], sample_rate: int = 44100,
                           probe_seal: float = 0.0,
                           ambient_db: float = 0.0) -> Dict:
    signal = np.array(audio_data) if audio_data else np.random.randn(44100) * 0.001
    n = len(signal)

    factors = []
    score = 1.0

    rms = float(np.sqrt(np.mean(signal ** 2)))
    if rms < 0.0001:
        factors.append({'name': 'Signal Level', 'value': 'Very Low', 'impact': -0.3, 'detail': 'Signal amplitude too low — check probe placement'})
        score -= 0.3
    elif rms > 0.5:
        factors.append({'name': 'Signal Level', 'value': 'Saturated', 'impact': -0.4, 'detail': 'Signal clipped — reduce gain or reposition'})
        score -= 0.4
    elif rms < 0.001:
        factors.append({'name': 'Signal Level', 'value': 'Low', 'impact': -0.15, 'detail': 'Weak signal — verify probe seal'})
        score -= 0.15
    else:
        factors.append({'name': 'Signal Level', 'value': f'{rms:.4f}', 'impact': 0, 'detail': 'Signal level acceptable'})

    clipping_ratio = float(np.mean(np.abs(signal) > 0.95))
    if clipping_ratio > 0.01:
        factors.append({'name': 'Clipping', 'value': f'{clipping_ratio*100:.1f}%', 'impact': -0.35, 'detail': 'Microphone clipping detected'})
        score -= 0.35

    zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
    est_freq = zero_crossings / (2 * n / sample_rate)
    if 200 < est_freq < 800 and rms > 0.02:
        factors.append({'name': 'Crying Detection', 'value': 'Likely', 'impact': -0.4, 'detail': 'Baby crying detected — pause and soothe'})
        score -= 0.4
    else:
        factors.append({'name': 'Crying Detection', 'value': 'Not detected', 'impact': 0, 'detail': 'No crying pattern found'})

    fft_vals = np.fft.rfft(signal * np.hanning(n))
    power = np.abs(fft_vals) ** 2
    power_safe = power[power > 0]
    if len(power_safe) > 0:
        flatness = float(np.exp(np.mean(np.log(power_safe))) / np.mean(power_safe))
    else:
        flatness = 0.0
    if flatness > 0.5 and rms > 0.1:
        factors.append({'name': 'Movement', 'value': 'Likely', 'impact': -0.3, 'detail': 'Movement artifact — reposition infant'})
        score -= 0.3
    else:
        factors.append({'name': 'Movement', 'value': 'Not detected', 'impact': 0, 'detail': 'Minimal movement artifacts'})

    if abs(est_freq - 50) < 3 or abs(est_freq - 60) < 3:
        factors.append({'name': 'Electrical Interference', 'value': '50/60 Hz detected', 'impact': -0.2, 'detail': 'Power-line interference — shield device'})
        score -= 0.2
    else:
        factors.append({'name': 'Electrical Interference', 'value': 'Not detected', 'impact': 0, 'detail': 'No power-line interference'})

    if probe_seal > 0.7:
        factors.append({'name': 'Probe Fit', 'value': 'Good seal', 'impact': 0.1, 'detail': f'Acoustic seal: {probe_seal:.0%}'})
    elif probe_seal > 0.4:
        factors.append({'name': 'Probe Fit', 'value': 'Marginal', 'impact': -0.15, 'detail': f'Acoustic seal weak: {probe_seal:.0%} — reposition'})
        score -= 0.15
    else:
        factors.append({'name': 'Probe Fit', 'value': 'Poor', 'impact': -0.35, 'detail': f'Acoustic seal: {probe_seal:.0%} — likely leakage'})
        score -= 0.35

    if ambient_db > 60:
        factors.append({'name': 'Ambient Noise', 'value': f'{ambient_db:.0f} dB', 'impact': -0.3, 'detail': 'Environment too noisy — move to quieter location'})
        score -= 0.3
    elif ambient_db > 45:
        factors.append({'name': 'Ambient Noise', 'value': f'{ambient_db:.0f} dB', 'impact': -0.1, 'detail': 'Moderate ambient noise'})
        score -= 0.1
    else:
        factors.append({'name': 'Ambient Noise', 'value': f'{ambient_db:.0f} dB', 'impact': 0.05, 'detail': 'Quiet environment — good for screening'})
        score += 0.05

    spectral_entropy = compute_spectral_entropy(signal, sample_rate)
    if spectral_entropy > 0.85:
        factors.append({'name': 'Spectral Quality', 'value': 'High entropy', 'impact': -0.1, 'detail': 'Broadband noise dominant'})
        score -= 0.1
    else:
        factors.append({'name': 'Spectral Quality', 'value': 'Structured', 'impact': 0.05, 'detail': 'Spectral content appears structured'})
        score += 0.05

    frame_stability = compute_frame_stability(signal, sample_rate)
    if frame_stability < 0.3:
        factors.append({'name': 'Temporal Stability', 'value': 'Unstable', 'impact': -0.2, 'detail': 'Signal varies significantly across frames'})
        score -= 0.2
    else:
        factors.append({'name': 'Temporal Stability', 'value': 'Stable', 'impact': 0.05, 'detail': f'Frame-to-frame stability: {frame_stability:.0%}'})
        score += 0.05

    overall_score = max(0.0, min(1.0, score))
    confidence = compute_confidence(factors, overall_score)

    if overall_score >= 0.7:
        recommendation = 'READY'
        probe_status = 'GOOD' if probe_seal > 0.6 else 'ADJUST'
    elif overall_score >= 0.4:
        recommendation = 'ADJUST'
        probe_status = 'MARGINAL'
    else:
        recommendation = 'REJECT'
        probe_status = 'POOR'

    noise_label = 'Quiet' if ambient_db < 45 else 'Moderate' if ambient_db < 60 else 'Noisy'

    artifact_label = 'None'
    for f in factors:
        if f['impact'] < -0.2 and f['name'] not in ('Probe Fit', 'Ambient Noise', 'Signal Level'):
            artifact_label = f['value']
            break

    report = SignalQualityReport(
        overall_score=round(overall_score, 3),
        confidence=round(confidence, 3),
        probe_fit=probe_status,
        ambient_noise=noise_label,
        artifact_detected=artifact_label,
        recommendation=recommendation,
        factors=factors,
        model_version=MODEL_VERSION
    )
    return asdict(report)


def compute_spectral_entropy(signal: np.ndarray, sample_rate: int) -> float:
    n = len(signal)
    fft_vals = np.fft.rfft(signal * np.hanning(n))
    power = np.abs(fft_vals) ** 2
    total = np.sum(power)
    if total == 0:
        return 1.0
    probs = power / total
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(probs))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def compute_frame_stability(signal: np.ndarray, sample_rate: int,
                             frame_size: int = 1024, hop: int = 512) -> float:
    n = len(signal)
    frames = []
    for start in range(0, n - frame_size, hop):
        frames.append(signal[start:start + frame_size])
    if len(frames) < 2:
        return 1.0
    energies = [float(np.sqrt(np.mean(f ** 2))) for f in frames]
    mean_e = np.mean(energies)
    if mean_e == 0:
        return 0.0
    cv = np.std(energies) / mean_e
    return float(max(0, 1 - cv))


def compute_confidence(factors: List[Dict], score: float) -> float:
    negative_impacts = sum(abs(f['impact']) for f in factors if f['impact'] < 0)
    base_confidence = 0.5 + 0.3 * score
    noise_penalty = min(0.2, negative_impacts * 0.1)
    return max(0.1, min(0.95, base_confidence - noise_penalty))


def classify_dpoae_spectrum(magnitudes: List[float], frequencies: List[float],
                             dp_frequency: float) -> Dict:
    mag_arr = np.array(magnitudes)
    freq_arr = np.array(frequencies)

    dp_idx = np.argmin(np.abs(freq_arr - dp_frequency))
    search_bins = 5

    dp_region_start = max(0, dp_idx - search_bins)
    dp_region_end = min(len(mag_arr), dp_idx + search_bins + 1)
    dp_peak = float(np.max(mag_arr[dp_region_start:dp_region_end]))

    noise_mask = np.ones(len(freq_arr), dtype=bool)
    noise_mask[max(0, dp_idx - 20):min(len(freq_arr), dp_idx + 20)] = False
    noise_level = float(np.mean(mag_arr[noise_mask]))

    snr = dp_peak - noise_level

    adjacent_mask = np.abs(freq_arr - dp_frequency) < 300
    adjacent_mask &= ~((np.abs(freq_arr - dp_frequency) < 50))
    peak_stability = float(np.std(mag_arr[adjacent_mask])) if np.any(adjacent_mask) else 0.0

    return {
        'dp_peak': round(dp_peak, 2),
        'noise_level': round(noise_level, 2),
        'snr': round(snr, 2),
        'peak_stability': round(peak_stability, 3),
        'is_present': snr >= 6.0,
        'confidence': round(min(1.0, max(0.0, snr / 20)), 3)
    }


def explain_decision(factors: List[Dict], verdict: str, dpoae_data: Dict) -> Dict:
    explanation = {
        'verdict': verdict,
        'summary': '',
        'positive_factors': [],
        'negative_factors': [],
        'technical_details': dpoae_data or {},
        'disclaimer': 'Screening is not a diagnosis. This result indicates whether further clinical evaluation is recommended.'
    }

    for f in factors:
        if f['impact'] >= 0:
            explanation['positive_factors'].append({
                'factor': f['name'],
                'value': f['value'],
                'detail': f['detail']
            })
        else:
            explanation['negative_factors'].append({
                'factor': f['name'],
                'value': f['value'],
                'detail': f['detail']
            })

    if verdict == 'PASS':
        explanation['summary'] = 'Coherent cochlear emissions detected with adequate signal quality. Screen result: PASS. Refer for diagnostic evaluation if clinical concern exists.'
    elif verdict == 'REFER':
        explanation['summary'] = 'Cochlear emissions below criterion or inadequate signal quality in one or more frequencies. Screen result: REFER for audiological evaluation.'
    else:
        explanation['summary'] = 'Technical quality insufficient for a valid screening result. RETEST recommended after addressing signal quality issues.'

    return explanation
