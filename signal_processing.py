"""
SHRUTI Signal Processing Module
DPOAE extraction, FFT, synchronous averaging, filtering, noise estimation.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional

SAMPLE_RATE = 44100
FRAME_SIZE = 1024
HOP_SIZE = 512
DPOAE_FREQUENCIES = [2000, 3000, 4000, 5000]
FREQ_RATIO = 1.22
L1_DB = 65
L2_DB = 55
SNR_THRESHOLD = 6.0
MIN_VALID_FRAMES = 3
PASS_CRITERION = 3


def generate_tone(frequency: float, duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(sample_rate * duration)) / sample_rate
    return np.sin(2 * np.pi * frequency * t)


def generate_dpoae_stimulus(f2: float, duration: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    f1 = f2 / FREQ_RATIO
    t = np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE
    stim1 = 10 ** (L1_DB / 20) * np.sin(2 * np.pi * f1 * t)
    stim2 = 10 ** (L2_DB / 20) * np.sin(2 * np.pi * f2 * t)
    return stim1, stim2


def simulate_dpoae_response(f2: float, dp_amplitude: float = -10.0,
                            noise_floor: float = -40.0, snr: float = 15.0,
                            duration: float = 1.0, add_artifacts: bool = False,
                            artifact_type: str = None) -> Dict:
    n_samples = int(SAMPLE_RATE * duration)
    t = np.arange(n_samples) / SAMPLE_RATE
    f_dp = 2 * f2 / FREQ_RATIO - f2

    dp_signal = 10 ** (dp_amplitude / 20) * np.sin(2 * np.pi * f_dp * t)
    noise_std = 10 ** (noise_floor / 20)
    noise = noise_std * np.random.randn(n_samples)

    if add_artifacts and artifact_type:
        if artifact_type == 'crying':
            cry_freq = 400 + 200 * np.sin(2 * np.pi * 3 * t)
            artifact = 0.3 * np.sin(2 * np.pi * cry_freq * t * 0.01)
            noise += artifact * noise_std * 50
        elif artifact_type == 'movement':
            burst = np.exp(-((t - duration / 2) ** 2) / (2 * 0.01 ** 2))
            artifact = burst * np.random.randn(n_samples) * noise_std * 100
            noise += artifact
        elif artifact_type == 'electrical':
            noise += 0.01 * np.sin(2 * np.pi * 50 * t)
        elif artifact_type == 'clipping':
            dp_signal = np.clip(dp_signal, -0.5, 0.5)

    recorded = dp_signal + noise
    return {
        'signal': recorded.tolist(),
        'dp_amplitude': dp_amplitude,
        'noise_floor': noise_floor,
        'snr': snr,
        'frequency': f2,
        'dp_frequency': f_dp,
        'duration': duration,
        'sample_rate': SAMPLE_RATE,
        'n_samples': n_samples,
        'has_artifacts': add_artifacts,
        'artifact_type': artifact_type or 'none'
    }


def compute_fft(signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Dict:
    n = len(signal)
    windowed = signal * np.hanning(n)
    fft_vals = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    magnitudes = 20 * np.log10(np.abs(fft_vals) + 1e-12)
    return {
        'frequencies': freqs.tolist(),
        'magnitudes': magnitudes.tolist(),
        'n_points': n
    }


def extract_dpoae_snir(signal: np.ndarray, f2: float,
                        sample_rate: int = SAMPLE_RATE) -> Dict:
    n = len(signal)
    windowed = signal * np.hanning(n)
    fft_vals = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)

    f_dp = 2 * f2 / FREQ_RATIO - f2
    freq_resolution = sample_rate / n
    dp_idx = np.argmin(np.abs(freqs - f_dp))
    search_range = max(1, int(50 / freq_resolution))

    dp_region = magnitudes_windowed(fft_vals, dp_idx, search_range)
    dp_amplitude = float(np.max(dp_region))

    noise_bins = []
    for i in range(len(freqs)):
        if abs(freqs[i] - f_dp) > 100 and abs(freqs[i] - f2) > 100:
            noise_bins.append(20 * np.log10(np.abs(fft_vals[i]) + 1e-12))
    noise_floor = float(np.mean(noise_bins)) if noise_bins else -60.0
    snr = dp_amplitude - noise_floor

    return {
        'frequency': f2,
        'dp_frequency': float(f_dp),
        'dp_amplitude': round(dp_amplitude, 2),
        'noise_floor': round(noise_floor, 2),
        'snr': round(snr, 2),
        'pass': snr >= SNR_THRESHOLD
    }


def magnitudes_windowed(fft_vals, center_idx, half_range):
    start = max(0, center_idx - half_range)
    end = min(len(fft_vals), center_idx + half_range + 1)
    return 20 * np.log10(np.abs(fft_vals[start:end]) + 1e-12)


def synchronous_average(frames: List[np.ndarray]) -> np.ndarray:
    if not frames:
        return np.array([])
    min_len = min(len(f) for f in frames)
    aligned = [f[:min_len] for f in frames]
    return np.mean(aligned, axis=0)


def bandpass_filter(signal: np.ndarray, low_freq: float, high_freq: float,
                     sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    mask = np.zeros_like(freqs)
    mask[(freqs >= low_freq) & (freqs <= high_freq)] = 1.0
    transition = 0.05 * (high_freq - low_freq)
    for i in range(len(freqs)):
        if low_freq - transition < freqs[i] < low_freq:
            mask[i] = (freqs[i] - (low_freq - transition)) / (2 * transition)
        elif high_freq < freqs[i] < high_freq + transition:
            mask[i] = (high_freq + transition - freqs[i]) / (2 * transition)
    filtered_fft = fft_vals * mask
    return np.fft.irfft(filtered_fft, n)


def estimate_noise_floor(signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    n = len(signal)
    fft_vals = np.fft.rfft(signal * np.hanning(n))
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    noise_regions = []
    for i in range(len(freqs)):
        is_notch = False
        for f in DPOAE_FREQUENCIES:
            f_dp = 2 * f / FREQ_RATIO - f
            if abs(freqs[i] - f_dp) < 80 or abs(freqs[i] - f) < 80:
                is_notch = True
                break
        if not is_notch:
            noise_regions.append(20 * np.log10(np.abs(fft_vals[i]) + 1e-12))
    return float(np.mean(noise_regions)) if noise_regions else -60.0


def detect_artifacts(signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Dict:
    n = len(signal)
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))

    clipping = bool(np.sum(np.abs(signal) > 0.95 * np.max(np.abs(signal))) > n * 0.01)

    zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
    freq_content = zero_crossings / (2 * n / sample_rate)

    spectral_flatness = compute_spectral_flatness(signal, sample_rate)

    artifacts = {
        'rms': round(float(rms), 6),
        'peak': round(float(peak), 6),
        'clipping': clipping,
        'estimated_frequency': round(float(freq_content), 1),
        'spectral_flatness': round(float(spectral_flatness), 4),
        'movement_likely': bool(rms > 0.1 and spectral_flatness > 0.5),
        'crying_likely': bool(200 < freq_content < 800 and rms > 0.02),
        'electrical_noise': bool(abs(freq_content - 50) < 5 or abs(freq_content - 60) < 5),
        'valid': not clipping and rms < 0.5 and not (rms > 0.1 and spectral_flatness > 0.5)
    }
    return artifacts


def compute_spectral_flatness(signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    n = len(signal)
    fft_vals = np.fft.rfft(signal * np.hanning(n))
    power = np.abs(fft_vals) ** 2
    power = power[power > 0]
    if len(power) == 0:
        return 0.0
    geometric_mean = np.exp(np.mean(np.log(power)))
    arithmetic_mean = np.mean(power)
    return float(geometric_mean / arithmetic_mean) if arithmetic_mean > 0 else 0.0


def full_screening_analysis(frames: List[np.ndarray], f2: float,
                             sample_rate: int = SAMPLE_RATE) -> Dict:
    valid_frames = []
    rejected_reasons = []

    for i, frame in enumerate(frames):
        artifacts = detect_artifacts(frame, sample_rate)
        if artifacts['valid']:
            valid_frames.append(frame)
        else:
            reason = []
            if artifacts['clipping']:
                reason.append('clipping')
            if artifacts['movement_likely']:
                reason.append('movement')
            if artifacts['crying_likely']:
                reason.append('crying')
            if artifacts['electrical_noise']:
                reason.append('electrical')
            rejected_reasons.append({'frame': i, 'reasons': reason})

    if len(valid_frames) < MIN_VALID_FRAMES:
        return {
            'status': 'RETEST',
            'reason': f'Only {len(valid_frames)} valid frames out of {len(frames)}',
            'valid_frames': len(valid_frames),
            'total_frames': len(frames),
            'rejected': rejected_reasons,
            'dpoae': None
        }

    averaged = synchronous_average(filtered_frames(valid_frames, f2, sample_rate))
    dpoae = extract_dpoae_snir(averaged, f2, sample_rate)

    n_pass = sum(1 for r in [dpoae] if r['pass'])

    return {
        'status': 'PASS' if n_pass >= PASS_CRITERION else 'REFER',
        'valid_frames': len(valid_frames),
        'total_frames': len(frames),
        'rejected': rejected_reasons,
        'dpoae': dpoae,
        'noise_floor': round(estimate_noise_floor(averaged, sample_rate), 2)
    }


def filtered_frames(frames: List[np.ndarray], f2: float,
                     sample_rate: int) -> List[np.ndarray]:
    f_dp = 2 * f2 / FREQ_RATIO - f2
    low = max(500, f_dp - 200)
    high = min(sample_rate / 2 - 100, f_dp + 200)
    return [bandpass_filter(f, low, high, sample_rate) for f in frames]
