"""Parse HL7/FDA Annotated ECG (aECG) XML files.

Extracts:
- 12-lead waveform data
- Sampling frequency
- Beat annotations (P/QRS/T onset/offset, RR intervals, beat types)
- Global measurements (PR, QRS, QT intervals, axis, amplitudes)
"""

import xml.etree.ElementTree as ET
import numpy as np
import os
import json


def strip_ns(tag):
    return tag.split('}')[-1] if '}' in tag else tag


def find_value(el, code, code_system=None):
    """Find a <value> element with a given @code."""
    for v in el.iter():
        if strip_ns(v.tag) == 'value':
            attrs = dict(v.attrib)
            c = attrs.get('code', '')
            if code in c:
                return v
    return None


def get_text_value(el):
    """Get text or attribute value."""
    a = dict(el.attrib)
    val = a.get('value', '')
    if val:
        return val
    return (el.text or '').strip()


class AECGParser:
    """Parse HL7 aECG XML files into NumPy arrays."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.fs = None          # sampling frequency
        self.n_leads = 0        # number of leads
        self.lead_names = []    # lead names
        self.signals = {}       # {lead_name: np.ndarray}
        self.duration_sec = 0
        self.annotations = []   # [{beat_idx, P_on, P_off, QRS_on, QRS_off, T_on, T_off, type, ...}]
        self.global_measurements = {}  # PR, QRS, QT, etc.
        self.patient_info = {}

    def parse(self):
        tree = ET.parse(self.filepath)
        root = tree.getroot()
        ns_uri = 'urn:hl7-org:v3'

        # ---- 1. Find sampling frequency ----
        for el in root.iter():
            tag = strip_ns(el.tag)
            if tag == 'value':
                a = dict(el.attrib)
                if 'unit' in a and a['unit'] == 'Hz' and 'value' in a:
                    if a['value'] == '500.0':
                        # Sampling rate indicators: sampleRate or clock frequency
                        pass
                    # Check if this is "sampleRate" or similar
                    val = float(a['value'])
                    code = a.get('code', '')
                    if 'RATE' in code.upper() or 'FREQ' in code.upper() or val > 100:
                        # Could be sampling rate
                        pass

        # More systematic: find the sequenceSet containing waveform data
        # and count samples per lead
        sequences = list(root.iter())
        seq_sets = [el for el in sequences if strip_ns(el.tag) == 'sequenceSet']

        # ---- 2. Extract all <digits> elements with waveforms ----
        waveform_digits = []
        measurement_info = []  # (lead_code, scale, unit)

        for el in root.iter():
            tag = strip_ns(el.tag)
            if tag == 'digits':
                text = (el.text or '').strip()
                if text and len(text) > 40:  # Real waveform data
                    # Check parent context - is this a lead waveform?
                    # Walk up to find code and scale
                    parent = el
                    code = ''
                    scale_val = 1.0
                    lead_unit = 'uV'
                    while parent is not None:
                        ptag = strip_ns(parent.tag)
                        if ptag == 'sequence':
                            # Look for code and scale in this sequence
                            for child in parent:
                                ctag = strip_ns(child.tag)
                                if ctag == 'code':
                                    code = child.get('code', '')
                                elif ctag == 'value':
                                    a = dict(child.attrib)
                                    if 'unit' in a and a['unit'] in ('uV', 'mV', 'mm'):
                                        scale_val = float(a.get('value', '1.0'))
                                        lead_unit = a['unit']
                        parent = None if 'component' in ptag.lower() else (
                            {None} if parent is None else
                            type(parent, (), {})  # break after sequence
                        )
                        if ptag == 'sequenceSet':
                            break
                        # Go up
                        # Actually let's do this differently

                    # Parse digits
                    try:
                        samples = np.array([float(x) for x in text.split()], dtype=np.float64)
                        if len(samples) > 10:
                            waveform_digits.append(samples)
                    except ValueError:
                        pass

        # ---- 3. More robust: find all leads by looking at component structure ----
        # In aECG, each lead is typically a <component> with <sequenceSet>
        leads = []
        for comp in root.iter():
            if strip_ns(comp.tag) == 'component':
                seq_set = comp.find('.//{' + ns_uri + '}sequenceSet')
                if seq_set is not None:
                    # Find lead code
                    lead_code = ''
                    lead_name = ''
                    scale = 1.0
                    fs_lead = None

                    for el in comp.iter():
                        t = strip_ns(el.tag)
                        if t == 'code':
                            c = el.get('code', '')
                            dn = el.get('displayName', '')
                            if 'LEAD' in c.upper() or 'MDC_ECG_LEAD' in c:
                                lead_code = c
                                lead_name = dn if dn else c.split('_')[-1]
                        elif t == 'value':
                            a = dict(el.attrib)
                            if a.get('unit') == 'uV':
                                scale = float(a.get('value', '1.0'))
                            elif a.get('unit') == 'Hz' and 'RATE' in a.get('code', '').upper():
                                fs_lead = float(a['value'])

                    if fs_lead and not self.fs:
                        self.fs = fs_lead

                    # Find digits in this component's sequenceSet
                    for d in seq_set.iter():
                        if strip_ns(d.tag) == 'digits':
                            text = (d.text or '').strip()
                            try:
                                samples = np.array([float(x) for x in text.split()], dtype=np.float64)
                                if len(samples) > 100:
                                    leads.append({
                                        'name': lead_name or f'lead_{len(leads)}',
                                        'code': lead_code,
                                        'signal': samples * scale if scale else samples,
                                        'n_samples': len(samples),
                                    })
                            except ValueError:
                                pass

        # ---- 4. If no structured leads found, use raw digits approach ----
        if not leads:
            print(f"  WARNING: No structured leads found, trying raw extraction")
            # Try to find the sampling rate from PQ values
            for el in root.iter():
                t = strip_ns(el.tag)
                if t == 'value':
                    a = dict(el.attrib)
                    if a.get('unit') == 'Hz' and float(a.get('value', '0')) > 100:
                        self.fs = float(a['value'])
                        break

            # Grab all digit sequences
            for el in root.iter():
                if strip_ns(el.tag) == 'digits':
                    text = (el.text or '').strip()
                    try:
                        samples = np.array([float(x) for x in text.split()], dtype=np.float64)
                        if len(samples) > 100:
                            leads.append({
                                'name': f'lead_{len(leads)}',
                                'code': '',
                                'signal': samples,
                                'n_samples': len(samples),
                            })
                    except ValueError:
                        pass

        # Store
        for i, ld in enumerate(leads):
            self.signals[ld['name']] = ld['signal']

        self.lead_names = [ld['name'] for ld in leads]
        self.n_leads = len(leads)

        if self.leads_available and len(leads[0]['signal']) > 0:
            if not self.fs:
                self.fs = 250.0  # default
            self.duration_sec = len(leads[0]['signal']) / self.fs

        # ---- 5. Extract beat annotations ----
        # P wave: MDC_ECG_WAVC_PWAVE with IVL_PQ (onset/offset)
        # QRS: MDC_ECG_WAVC_QRSWAVE
        # T wave: MDC_ECG_WAVC_TWAVE

        for comp in root.iter():
            if strip_ns(comp.tag) != 'component':
                continue

            beat_ann = {}
            for el in comp.iter():
                t = strip_ns(el.tag)
                if t == 'value':
                    a = dict(el.attrib)
                    code = a.get('code', '')

                    if 'PWAVE' in code:
                        # Look for IVL_PQ with low/high
                        ivl = comp.find('.//{' + ns_uri + '}value[@{http://www.w3.org/2001/XMLSchema-instance}type=\'IVL_PQ\']')
                        if ivl is not None:
                            lo = ivl.find('.//{' + ns_uri + '}low')
                            hi = ivl.find('.//{' + ns_uri + '}high')
                            if lo is not None and hi is not None:
                                beat_ann['P_on'] = float(lo.get('value', '0'))
                                beat_ann['P_off'] = float(hi.get('value', '0'))

                    elif 'QRSWAVE' in code:
                        ivl = None
                        # Find sibling IVL_PQ
                        parent = el.getparent() if hasattr(el, 'getparent') else None
                        # Scan nearby
                        for sibling in comp.iter():
                            st = strip_ns(sibling.tag)
                            if st == 'value' and sibling.get('{http://www.w3.org/2001/XMLSchema-instance}type') == 'IVL_PQ':
                                ivl = sibling
                                break
                        if ivl is not None:
                            lo = ivl.find('.//{' + ns_uri + '}low')
                            hi = ivl.find('.//{' + ns_uri + '}high')
                            if lo is not None and hi is not None:
                                beat_ann['QRS_on'] = float(lo.get('value', '0'))
                                beat_ann['QRS_off'] = float(hi.get('value', '0'))

                    elif 'TWAVE' in code:
                        ivl = None
                        for sibling in comp.iter():
                            st = strip_ns(sibling.tag)
                            if st == 'value' and sibling.get('{http://www.w3.org/2001/XMLSchema-instance}type') == 'IVL_PQ':
                                ivl = sibling
                                break
                        if ivl is not None:
                            lo = ivl.find('.//{' + ns_uri + '}low')
                            hi = ivl.find('.//{' + ns_uri + '}high')
                            if lo is not None and hi is not None:
                                beat_ann['T_on'] = float(lo.get('value', '0'))
                                beat_ann['T_off'] = float(hi.get('value', '0'))

            if beat_ann:
                beat_ann['beat_idx'] = len(self.annotations)
                self.annotations.append(beat_ann)

        # ---- 6. Global measurements ----
        for el in root.iter():
            t = strip_ns(el.tag)
            if t == 'value':
                a = dict(el.attrib)
                unit = a.get('unit', '')
                val = a.get('value', '')
                code = a.get('code', '')
                if val and unit:
                    if unit == 'ms':
                        self.global_measurements[code.split('.')[-1] if '.' in code else code] = f'{val}ms'
                    elif unit == 'bpm':
                        self.global_measurements['heart_rate'] = f'{val}bpm'
                    elif unit == 'deg':
                        self.global_measurements[code.split('.')[-1] if '.' in code else code] = f'{val}deg'
                    elif unit == 'mV':
                        self.global_measurements[code.split('.')[-1] if '.' in code else code] = f'{val}mV'

        return self

    @property
    def leads_available(self):
        return self.n_leads > 0

    def get_lead(self, idx=0):
        """Get lead signal by index."""
        if self.lead_names:
            name = self.lead_names[idx]
            return self.signals[name]
        return None

    def summarize(self):
        s = {
            'file': os.path.basename(self.filepath),
            'n_leads': self.n_leads,
            'lead_names': self.lead_names,
            'fs': self.fs,
            'duration_sec': round(self.duration_sec, 1),
            'n_beats_annotated': len(self.annotations),
            'global_measurements': self.global_measurements,
            'first_beat': self.annotations[0] if self.annotations else None,
            'annotation_keys': sorted(set().union(*[a.keys() for a in self.annotations])) if self.annotations else [],
        }
        return s


# =====================================================================
# Test
# =====================================================================
if __name__ == '__main__':
    import sys
    test_path = 'C:/LoyaltyLo/datasets/RA-LA_Reversal/aECG/1805185J6U.aECG'
    if len(sys.argv) > 1:
        test_path = sys.argv[1]

    print(f"Parsing: {test_path}")
    parser = AECGParser(test_path)
    parser.parse()
    summary = parser.summarize()

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # Sample values
    if parser.leads_available:
        lead0 = parser.get_lead(0)
        print(f"\nLead 0 ({parser.lead_names[0]}): {len(lead0)} samples")
        print(f"  First 10: {lead0[:10]}")
        print(f"  Range: [{lead0.min():.1f}, {lead0.max():.1f}]")
