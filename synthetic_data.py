"""
SHRUTI Synthetic Data Laboratory
Python-based research environment for generating synthetic DPOAE-like signals.
"""
import numpy as np
from typing import Dict, List, Optional
import json

SAMPLE_RATE = 44100
FREQ_RATIO = 1.22
TEST_FREQUENCIES = [2000, 3000, 4000, 5000]


def generate_normal_response(n_babies: int = 10) -> List[Dict]:
    datasets = []
    for i in range(n_babies):
        dp_amplitudes = np.random.uniform(-15, -5, len(TEST_FREQUENCIES))
        noise_floors = np.random.uniform(-45, -35, len(TEST_FREQUENCIES))
        snrs = dp_amplitudes - noise_floors

        per_freq = []
        for j, f2 in enumerate(TEST_FREQUENCIES):
            f_dp = 2 * f2 / FREQ_RATIO - f2
            duration = 1.0
            n_samples = int(SAMPLE_RATE * duration)
            t = np.arange(n_samples) / SAMPLE_RATE
            signal = (10 ** (dp_amplitudes[j] / 20) * np.sin(2 * np.pi * f_dp * t) +
                      10 ** (noise_floors[j] / 20) * np.random.randn(n_samples))

            per_freq.append({
                'frequency': f2,
                'dp_amplitude': round(float(dp_amplitudes[j]), 2),
                'noise_floor': round(float(noise_floors[j]), 2),
                'snr': round(float(snrs[j]), 2),
                'pass': bool(snrs[j] >= 6.0),
                'signal': signal.tolist()[:500]
            })

        datasets.append({
            'id': f'normal_{i+1}',
            'type': 'normal',
            'baby_name': f'Synthetic Baby N{i+1:03d}',
            'per_frequency': per_freq,
            'overall_verdict': 'PASS',
            'notes': 'Synthetic normal DPOAE response'
        })
    return datasets


def generate_weak_response(n_babies: int = 10) -> List[Dict]:
    datasets = []
    for i in range(n_babies):
        dp_amplitudes = np.random.uniform(-25, -18, len(TEST_FREQUENCIES))
        noise_floors = np.random.uniform(-40, -32, len(TEST_FREQUENCIES))
        snrs = dp_amplitudes - noise_floors

        per_freq = []
        for j, f2 in enumerate(TEST_FREQUENCIES):
            f_dp = 2 * f2 / FREQ_RATIO - f2
            duration = 1.0
            n_samples = int(SAMPLE_RATE * duration)
            t = np.arange(n_samples) / SAMPLE_RATE
            signal = (10 ** (dp_amplitudes[j] / 20) * np.sin(2 * np.pi * f_dp * t) +
                      10 ** (noise_floors[j] / 20) * np.random.randn(n_samples))

            per_freq.append({
                'frequency': f2,
                'dp_amplitude': round(float(dp_amplitudes[j]), 2),
                'noise_floor': round(float(noise_floors[j]), 2),
                'snr': round(float(snrs[j]), 2),
                'pass': bool(snrs[j] >= 6.0),
                'signal': signal.tolist()[:500]
            })

        n_pass = sum(1 for f in per_freq if f['pass'])
        datasets.append({
            'id': f'weak_{i+1}',
            'type': 'weak',
            'baby_name': f'Synthetic Baby W{i+1:03d}',
            'per_frequency': per_freq,
            'overall_verdict': 'REFER' if n_pass < 3 else 'PASS',
            'notes': 'Synthetic weak DPOAE response — likely REFER'
        })
    return datasets


def generate_noisy_response(n_babies: int = 10, noise_level: str = 'moderate') -> List[Dict]:
    noise_multipliers = {'mild': 2, 'moderate': 5, 'severe': 15}
    mult = noise_multipliers.get(noise_level, 5)

    datasets = []
    for i in range(n_babies):
        dp_amplitudes = np.random.uniform(-12, -6, len(TEST_FREQUENCIES))
        noise_floors = np.random.uniform(-35 + mult, -25, len(TEST_FREQUENCIES))
        snrs = dp_amplitudes - noise_floors

        per_freq = []
        for j, f2 in enumerate(TEST_FREQUENCIES):
            f_dp = 2 * f2 / FREQ_RATIO - f2
            duration = 1.0
            n_samples = int(SAMPLE_RATE * duration)
            t = np.arange(n_samples) / SAMPLE_RATE
            signal = (10 ** (dp_amplitudes[j] / 20) * np.sin(2 * np.pi * f_dp * t) +
                      10 ** (noise_floors[j] / 20) * np.random.randn(n_samples))

            per_freq.append({
                'frequency': f2,
                'dp_amplitude': round(float(dp_amplitudes[j]), 2),
                'noise_floor': round(float(noise_floors[j]), 2),
                'snr': round(float(snrs[j]), 2),
                'pass': bool(snrs[j] >= 6.0),
                'signal': signal.tolist()[:500]
            })

        datasets.append({
            'id': f'noisy_{i+1}',
            'type': f'noisy_{noise_level}',
            'baby_name': f'Synthetic Baby X{i+1:03d}',
            'per_frequency': per_freq,
            'overall_verdict': 'RETEST',
            'notes': f'Synthetic noisy signal ({noise_level}) — likely RETEST'
        })
    return datasets


def generate_artifact_response(artifact_type: str = 'movement',
                                n_babies: int = 5) -> List[Dict]:
    datasets = []
    for i in range(n_babies):
        per_freq = []
        for j, f2 in enumerate(TEST_FREQUENCIES):
            f_dp = 2 * f2 / FREQ_RATIO - f2
            duration = 1.0
            n_samples = int(SAMPLE_RATE * duration)
            t = np.arange(n_samples) / SAMPLE_RATE

            dp_amp = np.random.uniform(-12, -7)
            signal = 10 ** (dp_amp / 20) * np.sin(2 * np.pi * f_dp * t)
            noise = 10 ** (-40 / 20) * np.random.randn(n_samples)

            if artifact_type == 'movement':
                burst_center = np.random.uniform(0.2, 0.8) * duration
                burst = np.exp(-((t - burst_center) ** 2) / (2 * 0.02 ** 2))
                noise += burst * np.random.randn(n_samples) * 0.1
            elif artifact_type == 'crying':
                cry_freq = 400 + 200 * np.sin(2 * np.pi * 3 * t)
                noise += 0.05 * np.sin(2 * np.pi * cry_freq * t * 0.01)
            elif artifact_type == 'electrical':
                noise += 0.005 * np.sin(2 * np.pi * 50 * t)
            elif artifact_type == 'clipping':
                signal = np.clip(signal, -0.3, 0.3)

            recorded = signal + noise
            noise_floor = float(20 * np.log10(np.std(noise) + 1e-12))
            dp_meas = float(20 * np.log10(np.max(np.abs(signal)) + 1e-12))

            per_freq.append({
                'frequency': f2,
                'dp_amplitude': round(dp_meas, 2),
                'noise_floor': round(noise_floor, 2),
                'snr': round(dp_meas - noise_floor, 2),
                'pass': bool((dp_meas - noise_floor) >= 6.0),
                'signal': recorded.tolist()[:500],
                'artifact_injected': artifact_type
            })

        datasets.append({
            'id': f'artifact_{artifact_type}_{i+1}',
            'type': f'artifact_{artifact_type}',
            'baby_name': f'Synthetic Baby A{i+1:03d}',
            'per_frequency': per_freq,
            'overall_verdict': 'RETEST',
            'notes': f'Injected {artifact_type} artifact'
        })
    return datasets


def generate_probe_leakage_response(n_babies: int = 5,
                                     leakage_severity: str = 'moderate') -> List[Dict]:
    attenuation = {'mild': 0.7, 'moderate': 0.4, 'severe': 0.1}
    att = attenuation.get(leakage_severity, 0.4)

    datasets = []
    for i in range(n_babies):
        per_freq = []
        for j, f2 in enumerate(TEST_FREQUENCIES):
            f_dp = 2 * f2 / FREQ_RATIO - f2
            duration = 1.0
            n_samples = int(SAMPLE_RATE * duration)
            t = np.arange(n_samples) / SAMPLE_RATE

            dp_amp = np.random.uniform(-10, -5) * att
            signal = 10 ** (dp_amp / 20) * np.sin(2 * np.pi * f_dp * t)
            noise = 10 ** (-38 / 20) * np.random.randn(n_samples)
            recorded = signal + noise

            noise_floor = float(20 * np.log10(np.std(noise) + 1e-12))
            dp_meas = float(20 * np.log10(np.max(np.abs(signal)) + 1e-12))

            per_freq.append({
                'frequency': f2,
                'dp_amplitude': round(dp_meas, 2),
                'noise_floor': round(noise_floor, 2),
                'snr': round(dp_meas - noise_floor, 2),
                'pass': bool((dp_meas - noise_floor) >= 6.0),
                'signal': recorded.tolist()[:500],
                'probe_leakage': leakage_severity
            })

        datasets.append({
            'id': f'leak_{leakage_severity}_{i+1}',
            'type': f'probe_leak_{leakage_severity}',
            'baby_name': f'Synthetic Baby L{i+1:03d}',
            'per_frequency': per_freq,
            'overall_verdict': 'RETEST',
            'notes': f'Probe leakage simulation ({leakage_severity})',
            'probe_seal': round(att, 2)
        })
    return datasets


def generate_large_test_dataset(n_total: int = 100) -> List[Dict]:
    all_data = []
    all_data.extend(generate_normal_response(n_total // 5))
    all_data.extend(generate_weak_response(n_total // 5))
    all_data.extend(generate_noisy_response(n_total // 5, 'moderate'))
    all_data.extend(generate_artifact_response('movement', n_total // 10))
    all_data.extend(generate_artifact_response('crying', n_total // 10))
    all_data.extend(generate_probe_leakage_response(n_total // 10, 'moderate'))
    np.random.shuffle(all_data)
    return all_data


def generate_simulator_state(params: Dict) -> Dict:
    ambient_db = params.get('ambient_db', 35)
    probe_seal = params.get('probe_seal', 0.8)
    dp_amplitude = params.get('dp_amplitude', -10)
    movement = params.get('movement', 0.0)
    stimulus_level_l1 = params.get('stimulus_l1', 65)
    stimulus_level_l2 = params.get('stimulus_l2', 55)

    per_freq = []
    for f2 in TEST_FREQUENCIES:
        f_dp = 2 * f2 / FREQ_RATIO - f2
        effective_dp = dp_amplitude + 10 * np.log10(max(0.01, probe_seal))
        effective_noise = -35 + ambient_db * 0.3 + movement * 20

        snr = effective_dp - effective_noise
        freq_pass = snr >= 6.0

        per_freq.append({
            'frequency': f2,
            'dp_frequency': round(f_dp, 1),
            'dp_amplitude': round(effective_dp, 2),
            'noise_floor': round(effective_noise, 2),
            'snr': round(snr, 2),
            'pass': bool(freq_pass)
        })

    n_pass = sum(1 for f in per_freq if f['pass'])
    if n_pass >= 3:
        verdict = 'PASS'
    elif probe_seal < 0.3 or ambient_db > 60 or movement > 0.7:
        verdict = 'RETEST'
    else:
        verdict = 'REFER'

    return {
        'per_frequency': per_freq,
        'verdict': verdict,
        'probe_seal': probe_seal,
        'ambient_db': ambient_db,
        'dp_amplitude': dp_amplitude,
        'movement': movement,
        'stimulus_l1': stimulus_level_l1,
        'stimulus_l2': stimulus_level_l2,
        'passing_frequencies': n_pass,
        'total_frequencies': len(TEST_FREQUENCIES)
    }


EXPORT_FORMATS = {
    'description': 'Synthetic data lab for algorithm development and demonstrations.',
    'disclaimer': 'Synthetic data should NOT be presented as clinical validation data.',
    'available_types': [
        'normal — Normal DPOAE responses',
        'weak — Weak DPOAE responses (likely REFER)',
        'noisy_mild — Mild ambient noise',
        'noisy_moderate — Moderate ambient noise',
        'noisy_severe — Severe ambient noise',
        'artifact_movement — Movement artifact injection',
        'artifact_crying — Baby crying simulation',
        'artifact_electrical — 50/60 Hz electrical interference',
        'artifact_clipping — Microphone clipping',
        'probe_leak_mild — Mild probe leakage',
        'probe_leak_moderate — Moderate probe leakage',
        'probe_leak_severe — Severe probe leakage'
    ]
}
