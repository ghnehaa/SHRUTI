"""
SHRUTI Clinical Decision Engine
Converts validated signal measurements into PASS, REFER, or RETEST.
Separates measurement from decision logic.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

PROTOCOL_VERSION = "SHRUTI-CDR-v1.0"
PROTOCOL_STANDARD = "IEC 60645-6"
FREQ_RATIO = 1.22
L1_DB = 65
L2_DB = 55
SNR_THRESHOLD = 6.0
PASS_FREQUENCIES_REQUIRED = 3
TOTAL_TEST_FREQUENCIES = 4
TEST_FREQUENCIES = [2000, 3000, 4000, 5000]


@dataclass
class ScreeningResult:
    baby_id: str
    baby_name: str
    date: str
    ear: str
    ear_left_result: str
    ear_right_result: str
    snr_left: Optional[float]
    snr_right: Optional[float]
    verdict: str
    follow_up_date: str
    status: str
    operator: str
    protocol: str
    protocol_version: str
    software_version: str
    signal_quality_score: Optional[float]
    ai_confidence: Optional[float]
    frames_valid: Optional[int]
    frames_total: Optional[int]
    frames_rejected_reasons: List[Dict]
    dpoae_details: Optional[Dict]
    explanation: Dict
    notes: str


def evaluate_screening(ear_left_data: Dict, ear_right_data: Dict,
                        baby_id: str, baby_name: str, operator: str,
                        signal_quality: Optional[Dict] = None) -> Dict:
    left_result = evaluate_ear(ear_left_data, 'Left')
    right_result = evaluate_ear(ear_right_data, 'Right')

    overall = compute_overall_verdict(left_result, right_result)

    follow_up_days = determine_follow_up(overall)
    follow_up_date = (datetime.now() + timedelta(days=follow_up_days)).strftime('%Y-%m-%d')

    status = determine_status(overall, left_result, right_result)

    sq_score = signal_quality.get('overall_score') if signal_quality else None
    ai_conf = signal_quality.get('confidence') if signal_quality else None

    explanation = build_explanation(left_result, right_result, overall, signal_quality)

    return {
        'baby_id': baby_id,
        'baby_name': baby_name,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'ear': 'Both',
        'ear_left_result': left_result['verdict'],
        'ear_right_result': right_result['verdict'],
        'snr_left': left_result.get('representative_snr'),
        'snr_right': right_result.get('representative_snr'),
        'verdict': overall,
        'follow_up_date': follow_up_date,
        'status': status,
        'operator': operator,
        'protocol': PROTOCOL_STANDARD,
        'protocol_version': PROTOCOL_VERSION,
        'software_version': 'shruti-v1.0-hackathon',
        'signal_quality_score': sq_score,
        'ai_confidence': ai_conf,
        'frames_valid': left_result.get('valid_frames'),
        'frames_total': left_result.get('total_frames'),
        'frames_rejected_reasons': left_result.get('rejected_reasons', []),
        'dpoae_details': {'left': left_result.get('dpoae_per_freq'), 'right': right_result.get('dpoae_per_freq')},
        'explanation': explanation,
        'notes': ''
    }


def evaluate_ear(data: Dict, ear_label: str) -> Dict:
    if not data or 'dpoae_per_freq' not in data:
        return {
            'verdict': 'RETEST',
            'reason': 'No data available',
            'passing_frequencies': 0,
            'total_frequencies': 0,
            'representative_snr': None,
            'dpoae_per_freq': [],
            'valid_frames': data.get('valid_frames', 0) if data else 0,
            'total_frames': data.get('total_frames', 0) if data else 0,
            'rejected_reasons': data.get('rejected_reasons', [])
        }

    dpoae_per_freq = data.get('dpoae_per_freq', [])
    valid_frames = data.get('valid_frames', 0)
    total_frames = data.get('total_frames', 0)

    if valid_frames < 3:
        return {
            'verdict': 'RETEST',
            'reason': f'Insufficient valid frames ({valid_frames}/{total_frames})',
            'passing_frequencies': 0,
            'total_frequencies': len(dpoae_per_freq),
            'representative_snr': None,
            'dpoae_per_freq': dpoae_per_freq,
            'valid_frames': valid_frames,
            'total_frames': total_frames,
            'rejected_reasons': data.get('rejected_reasons', [])
        }

    passing_freqs = sum(1 for d in dpoae_per_freq if d.get('pass', False))
    snrs = [d.get('snr', -999) for d in dpoae_per_freq if d.get('snr') is not None]
    representative_snr = round(max(snrs), 2) if snrs else None

    if len(dpoae_per_freq) == 0:
        verdict = 'RETEST'
        reason = 'No frequency data'
    elif passing_freqs >= PASS_FREQUENCIES_REQUIRED:
        verdict = 'PASS'
        reason = f'{passing_freqs}/{TOTAL_TEST_FREQUENCIES} frequencies met criterion (SNR >= {SNR_THRESHOLD} dB)'
    else:
        verdict = 'REFER'
        reason = f'Only {passing_freqs}/{TOTAL_TEST_FREQUENCIES} frequencies met criterion'

    return {
        'verdict': verdict,
        'reason': reason,
        'passing_frequencies': passing_freqs,
        'total_frequencies': len(dpoae_per_freq),
        'representative_snr': representative_snr,
        'dpoae_per_freq': dpoae_per_freq,
        'valid_frames': valid_frames,
        'total_frames': total_frames,
        'rejected_reasons': data.get('rejected_reasons', [])
    }


def compute_overall_verdict(left: Dict, right: Dict) -> str:
    lv = left.get('verdict', 'RETEST')
    rv = right.get('verdict', 'RETEST')

    if lv == 'RETEST' or rv == 'RETEST':
        return 'RETEST'
    if lv == 'REFER' or rv == 'REFER':
        return 'REFER'
    return 'PASS'


def determine_follow_up(verdict: str) -> int:
    if verdict == 'PASS':
        return 90
    elif verdict == 'REFER':
        return 7
    else:
        return 14


def determine_status(verdict: str, left: Dict, right: Dict) -> str:
    if verdict == 'PASS':
        return 'Cleared'
    elif verdict == 'REFER':
        return 'ABR referral'
    else:
        return 'Re-screen due'


def build_explanation(left: Dict, right: Dict, overall: str,
                       signal_quality: Optional[Dict]) -> Dict:
    explanation = {
        'verdict': overall,
        'protocol': f'{PROTOCOL_STANDARD} — f2/f1 = {FREQ_RATIO}, L1/L2 = {L1_DB}/{L2_DB} dB SPL',
        'criterion': f'SNR >= {SNR_THRESHOLD} dB at {PASS_FREQUENCIES_REQUIRED} of {TOTAL_TEST_FREQUENCIES} frequencies',
        'left_ear': {
            'verdict': left.get('verdict'),
            'passing': f'{left.get("passing_frequencies", 0)}/{left.get("total_frequencies", 0)}',
            'snr': left.get('representative_snr')
        },
        'right_ear': {
            'verdict': right.get('verdict'),
            'passing': f'{right.get("passing_frequencies", 0)}/{right.get("total_frequencies", 0)}',
            'snr': right.get('representative_snr')
        },
        'disclaimer': 'Screening result — not a diagnosis. PASS does not rule out all hearing conditions. REFER indicates the need for audiological evaluation.',
        'bias': 'This engine is biased toward safe referral (REFER) rather than false reassurance.',
        'model_version': PROTOCOL_VERSION
    }

    if signal_quality:
        explanation['signal_quality'] = {
            'score': signal_quality.get('overall_score'),
            'confidence': signal_quality.get('confidence'),
            'recommendation': signal_quality.get('recommendation')
        }

    return explanation


def generate_follow_up_task(result: Dict, immunization_dates: List[Dict]) -> Dict:
    follow_up_date = result.get('follow_up_date')
    nearest_immunization = None

    for imm in immunization_dates:
        if imm.get('due_date') and imm.get('status') != 'Given':
            if follow_up_date and imm['due_date'] >= follow_up_date:
                nearest_immunization = imm
                break

    task = {
        'baby_id': result['baby_id'],
        'baby_name': result['baby_name'],
        'type': result['verdict'],
        'created_date': datetime.now().strftime('%Y-%m-%d'),
        'due_date': follow_up_date,
        'linked_immunization': nearest_immunization,
        'priority': 'High' if result['verdict'] == 'REFER' else 'Medium',
        'status': 'Pending',
        'escalation_level': 0,
        'notes': f"Screening result: {result['verdict']}. Follow-up scheduled."
    }

    if nearest_immunization:
        task['notes'] += f" Linked to {nearest_immunization.get('vaccine', '')} visit."
        task['type'] += ' (immunization-linked)'

    return task


def generate_digital_referral(result: Dict) -> Dict:
    return {
        'referral_id': f"REF-{result['baby_id']}-{datetime.now().strftime('%Y%m%d')}",
        'baby_id': result['baby_id'],
        'baby_name': result['baby_name'],
        'date': datetime.now().strftime('%Y-%m-%d'),
        'screening_result': result['verdict'],
        'left_ear': result['ear_left_result'],
        'right_ear': result['ear_right_result'],
        'snr_left': result.get('snr_left'),
        'snr_right': result.get('snr_right'),
        'referred_to': 'Audiological Evaluation (ABR)',
        'reason': 'Hearing screening did not meet PASS criterion',
        'protocol': result.get('protocol'),
        'operator': result.get('operator'),
        'status': 'Pending',
        'digital_qr': f"QR-{result['baby_id']}-REF"
    }
