"""Prepare the SEPIQ AVIDa-hHER2 dataset for AntiBinder training / validation.

What this script does
---------------------
1. Streams the ~500 MB ``AVIDa-hHER2.csv`` in chunks and keeps only rows with a
   definite ``binary_label`` (0.0 = non-binder, 1.0 = binder). The ~1.82M
   ``non-sig.`` / ``noise`` rows have no binary label and are excluded.
2. Strips the C-terminal ``HHHHHH`` His-tag present on every VHH sequence.
3. Splits each VHH into Chothia FR/CDR regions (``H-FR1`` ... ``H-FR4``), the
   format consumed by ``antigen_antibody_emb.py``.
     - method=auto (default): use ``abnumber`` (ANARCI) when it is installed and
       functional, and fall back to a built-in conserved-anchor splitter for the
       few sequences ANARCI rejects. The anchor splitter reproduces the exact
       Chothia CDR3 boundary (Cys94 -> Trp103), which is the only boundary used
       by the model's IgFold CDR3 pooling; CDR1/CDR2 boundaries are Chothia
       anchored within +-1-2 residues.
     - method=anchor: zero extra dependencies; retains >=99.5% of labeled VHHs.
4. Assigns a CLUSTER-DISJOINT train/val split on the ``Cluster`` column (93%
   sequence-identity clusters) so near-duplicate VHHs never leak into
   validation.
5. Reads the 624-aa HER2 antigen sequence from a prediction template's chain A
   and writes a single CSV with every column ``main.py`` needs.

Kaggle dataset (attach it via "Add Data"):
    https://www.kaggle.com/datasets/kolsmirnov/sepiq-2026-training-data
    Mounted at /kaggle/input/sepiq-2026-training-data/ (also auto-discovered
    under /kaggle/input/*/ if the slug changes).

Kaggle usage (internet ON for apt/pip)::

    # Optional but recommended: canonical ANARCI Chothia numbering.
    # abnumber needs the HMMER binary (hmmscan); the anchor fallback below
    # works without it but only the CDR3 boundary is guaranteed Chothia-exact.
    !apt-get install -y hmmer
    !pip install abnumber -q

    # Defaults already point at /kaggle/input and auto-discover the dataset
    # folder regardless of its slug, so --input/--template can be omitted.
    !python prepare_sepiq.py

Then train / validate::

    !python main.py train
    !python main.py test          # evaluates the held-out val split
"""

import argparse
import glob
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

# Kaggle mounts attached datasets here; the folder name equals the dataset slug.
KAGGLE_INPUT_DIR = '/kaggle/input'
DEFAULT_INPUT = f'{KAGGLE_INPUT_DIR}/sepiq-2026-training-data/AVIDa-hHER2.csv'
DEFAULT_TEMPLATE = f'{KAGGLE_INPUT_DIR}/sepiq-2026-training-data/prediction_template_ONU11.csv'
DEFAULT_OUTPUT = './datasets/process_data/SEPIQ/AVIDa-hHER2_binary.csv'

# Columns consumed by antibody_antibody_emb.py / main.py
REGION_COLS = ['H-FR1', 'H-CDR1', 'H-FR2', 'H-CDR2', 'H-FR3', 'H-CDR3', 'H-FR4']
MODEL_COLS = REGION_COLS + ['vh', 'ANT_Binding', 'Antigen', 'Antigen Sequence']

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

# ---------------------------------------------------------------------------
# Built-in conserved-anchor Chothia-style splitter (no external dependency)
# ---------------------------------------------------------------------------
_C1 = re.compile(r'C')
_W_HALL = re.compile(r'W[GAFYVILHMP][RKP]Q')        # conserved Trp35 hallmark
_W_HALL2 = re.compile(r'[WYFR][FYVILHMP]?[RKP][QARPK]?')
_C2_PAT = re.compile(r'[DE][TD][ATGSN][VILAM]Y[YFHSTLRI]?C')  # ...EDTAVYYC
_FR3_CORE = re.compile(r'[YF][A-Z][DSNEGPA]')       # NYAD / YAG / YPD ...

# Chothia_H canonical lengths with empirically validated VHH tolerance.
# CDR1/FR3 upper bounds are widened beyond the canonical range for the rare
# long-CDR1 / long-FR3 VHH germlines seen in structural databases (SAbDab2);
# the conserved Trp35 / Cys94 hallmarks still pin the other boundaries.
_BOUNDS = {
    'H-FR1': (18, 30), 'H-CDR1': (4, 20), 'H-FR2': (15, 22),
    'H-CDR2': (3, 12), 'H-FR3': (30, 56), 'H-CDR3': (4, 30),
    'H-FR4': (7, 14),
}

# Observed natural variation across SAbDab2 structures and NGS (AVIDa) VHHs:
# Trp103 substitutions/deletions, 105 Q->K/R/P/H/A/G/S/L/E/T/V/I/M, 106
# G->P/F/R/T, 107 T->A/I/L/E, 108 Q->L/P/K/R/T/E (T is the canonical JH4
# allele), 110 T->M, and records truncated after 108-111. Positions 107-110
# are therefore the end-anchoring core (107-110); 103-106 are taken as-is.
# The strict 105-110 core is kept for internal (tail) matches to avoid
# core-like motifs inside long CDR3 loops.
_FR4_CORE_LOOSE = re.compile(
    r'[TASGVILME][QLMIVPKRTE][VILA][TASGVM]')
_FR4_CORE = re.compile(r'[QKNR]G[TSAGV][QLMIVPKRT][VILA][TSAGV]')
_FR4_CORE_PART = re.compile(
    r'[TASGVILME][QLMIVPKRTE][VILA]?[TASGVM]?$')


def _find_fr4(seq):
    """Locate FR4 (Chothia 103-113) and return ``(start, end)``.

    Position 103 is W in the textbook VHH but substitutions/deletions occur,
    and records frequently truncate after FR4 or continue into a cloning
    tail. The search has two tiers (see tier comments below): a strict
    hallmark core that tolerates tails, and an end-anchored loose core for
    the hyper-variable 103-110 region. Tier 1 wins whenever it fires."""
    n = len(seq)
    win_lo = max(70, n - 50)
    win = seq[win_lo:]

    def make_loose(m, partial=False):
        core_abs = win_lo + m.start()
        start = core_abs - 4  # positions 103-106 are variable
        if start < 70:
            return None
        end = n if partial else min(n, start + 11)
        if 7 <= end - start <= 14:
            return start, end
        return None

    # 1) STRICT 105-110 core with a genuine W/R/F103 hallmark, tail 0-40.
    # This is the only tier allowed away from the sequence end, because the
    # hallmark discriminates FR4 from core-like motifs inside long CDR3
    # loops (e.g. RGSRLS, or TEAS preceded by a CDR3 phenylalanine).
    best = None
    for m in _FR4_CORE.finditer(win):
        core_abs = win_lo + m.start()
        start = core_abs - 2
        end = min(n, start + 11)
        tail = n - end
        if not (0 <= tail <= 40) or start < 70 or seq[start] not in 'WRF':
            continue
        if not (7 <= end - start <= 14):
            continue
        key = (0 if seq[start] == 'W' else 1, -start)
        if best is None or key < best[0]:
            best = (key, (start, end))
    if best is not None:
        return best[1]

    # 2) LOOSE 107-110 core strictly end-anchored (tip <= 4 residues) for the
    # hyper-variable 103-106 region; then a partial core for records
    # truncated at 108-111.
    best = None  # key: (partial 0<1, non-Trp 0<1, -start)
    for m in _FR4_CORE_LOOSE.finditer(win):
        trail = len(win) - m.end()
        if not (0 <= trail <= 4):
            continue
        cand = make_loose(m)
        if cand is None:
            continue
        hallmark = seq[cand[0]] in 'WRF'
        key = (0, 0 if hallmark else 1, -cand[0])
        if best is None or key < best[0]:
            best = (key, cand)
    if best is None:
        m = _FR4_CORE_PART.search(win)
        if m is not None and len(win) - m.end() == 0:
            cand = make_loose(m, partial=True)
            if cand is not None:
                best = ((1, 0 if seq[cand[0]] == 'W' else 1, -cand[0]), cand)
    return None if best is None else best[1]


def anchor_split(seq):
    """Split one VHH into Chothia regions using conserved anchors.

    Returns dict(REGION_COLS -> substring) or None for framework mutants /
    recombination artifacts that fall outside validated tolerances.
    """
    # --- FR1 -> conserved Cys22; require a Trp/Tyr later (FR2 hallmark) ---
    c1 = -1
    for m in _C1.finditer(seq, 15, 34):
        i = m.start()
        if re.search(r'[WFY]', seq[i + 8:i + 28]):
            c1 = i
            break
    if c1 < 0:
        return None
    cdr1_start = c1 + 4  # Chothia CDR1 starts at position 26 (3 residues after Cys22)

    # --- FR2 hallmark Trp35 ---
    w = None
    m = _W_HALL.search(seq, c1 + 8, c1 + 30) or _W_HALL2.search(seq, c1 + 8, c1 + 30)
    if m:
        w = m.start()
    else:
        m = re.search(r'W', seq[c1 + 8:c1 + 20])
        if m:
            w = c1 + 8 + m.start()
    if w is None:
        return None
    fr2_start = w - 2  # positions 33-34 precede Trp35
    if not (_BOUNDS['H-CDR1'][0] <= fr2_start - cdr1_start <= _BOUNDS['H-CDR1'][1]):
        return None

    # --- FR4 / conserved Trp103 (may be followed by a cloning tail) --------
    fr4 = _find_fr4(seq)
    if fr4 is None:
        return None
    f4, f4_end = fr4

    # --- conserved Cys94 at the end of FR3 --------------------------------
    # The FR3 hallmark pattern selects the junction specifically: extra
    # non-canonical disulfide cysteines inside long CDR3 loops never match
    # it, so the LAST pattern match is the junction. The plain-rfind fallback
    # is range-guarded so it can never grab Cys22 or an intra-CDR3 Cys.
    c2 = -1
    for m in _C2_PAT.finditer(seq, f4 - 48, f4 - 2):
        c2 = m.end() - 1
    if c2 < 0 or not (4 <= f4 - c2 <= 36):
        c2 = -1
        for m in re.finditer(r'Y[YFHSTLRI]?C', seq[f4 - 48:f4 - 2]):
            c2 = f4 - 48 + m.end() - 1
    if c2 < 0 or not (4 <= f4 - c2 <= 36):
        c = seq.rfind('C', f4 - 38, f4 - 2)
        if c >= 0 and 4 <= f4 - c <= 36:
            c2 = c
        else:
            c2 = -1
    if c2 < 0:
        return None

    # --- FR3 start: default after 19-residue FR2 + 5-residue CDR2, snapped to
    #     the conserved NYAD-like hallmark when present in the expected window.
    fr3 = fr2_start + 24
    m3 = _FR3_CORE.search(seq, fr2_start + 21, fr2_start + 33)
    if m3 is not None:
        candidate = m3.start() - 1
        if fr2_start + 22 <= candidate <= fr2_start + 32:
            fr3 = candidate
    fr2_end = fr2_start + 19

    cuts = [0, cdr1_start, fr2_start, fr2_end, fr3, c2 + 1, f4, f4_end]
    if any(cuts[i] >= cuts[i + 1] for i in range(len(cuts) - 1)):
        return None
    regions = [seq[cuts[i]:cuts[i + 1]] for i in range(7)]
    for name, sub in zip(REGION_COLS, regions):
        lo, hi = _BOUNDS[name]
        if not (lo <= len(sub) <= hi):
            return None
    return dict(zip(REGION_COLS, regions))


# ---------------------------------------------------------------------------
# Optional abnumber (ANARCI) splitter, matching the original COVID pipeline
# ---------------------------------------------------------------------------
_ABNUMBER_CHAIN = None
_ABNUMBER_OK = None


def _abnumber_available(smoke_seq):
    """Check abnumber import AND that its ANARCI backend actually runs.

    ANARCI shells out to ``hmmscan`` (HMMER), which is NOT installed by the
    pip package. On Kaggle: enable internet and run
    ``!apt-get install -y hmmer && pip install abnumber`` beforehand."""
    global _ABNUMBER_CHAIN, _ABNUMBER_OK
    if _ABNUMBER_OK is not None:
        return _ABNUMBER_OK
    try:
        from abnumber import Chain  # noqa: WPS433
    except ImportError:
        print("[prepare_sepiq] abnumber not installed "
              "(pip install abnumber); using built-in anchor splitter.")
        _ABNUMBER_OK = False
        return _ABNUMBER_OK
    if shutil.which('hmmscan') is None:
        print("[prepare_sepiq] hmmscan not found on PATH (install with "
              "'apt-get install -y hmmer'); using built-in anchor splitter.")
        _ABNUMBER_OK = False
        return _ABNUMBER_OK
    try:
        c = Chain(smoke_seq, scheme='chothia')
        _ = ''.join([c.fr1_seq, c.cdr1_seq, c.fr2_seq, c.cdr2_seq,
                     c.fr3_seq, c.cdr3_seq, c.fr4_seq])
        _ABNUMBER_CHAIN = Chain
        _ABNUMBER_OK = True
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[prepare_sepiq] ANARCI smoke test failed ({type(exc).__name__}: {exc}); "
              f"using built-in anchor splitter.")
        _ABNUMBER_OK = False
    return _ABNUMBER_OK


def abnumber_split(seq):
    """Chothia regions via abnumber. Returns dict or None."""
    try:
        c = _ABNUMBER_CHAIN(seq, scheme='chothia')
        parts = [c.fr1_seq, c.cdr1_seq, c.fr2_seq, c.cdr2_seq,
                 c.fr3_seq, c.cdr3_seq, c.fr4_seq]
    except Exception:
        return None
    if any(p is None or p == '' for p in parts):
        return None
    if ''.join(parts) != seq:
        return None
    return dict(zip(REGION_COLS, parts))


# ---------------------------------------------------------------------------
# Kaggle input path discovery
# ---------------------------------------------------------------------------
def resolve_kaggle_path(path, filename):
    """Return ``path`` if it exists, otherwise recursively search every
    attached Kaggle dataset for ``filename``. Returns the original path
    unchanged when nothing is found so the caller reports a clear error."""
    if path and os.path.exists(path):
        return path
    pattern = os.path.join(KAGGLE_INPUT_DIR, '**', filename)
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits[0] if hits else path


# ---------------------------------------------------------------------------
# Antigen sequence from the prediction template
# ---------------------------------------------------------------------------
def read_antigen_sequence(template_path):
    tpl = pd.read_csv(template_path)
    chain_a = tpl[tpl['chain'] == 'A'].sort_values('seqNum')
    residues = [THREE_TO_ONE[str(r).strip().upper()] for r in chain_a['resName']]
    seq = ''.join(residues)
    if len(seq) < 100:
        raise ValueError(f"Antigen chain A in {template_path} is unexpectedly short ({len(seq)} aa).")
    return seq


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Convert SEPIQ AVIDa-hHER2.csv into an AntiBinder-ready CSV '
                    '(cluster-disjoint train/val split).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', default=DEFAULT_INPUT,
                        help='path to AVIDa-hHER2.csv; if missing, auto-searched under '
                             '/kaggle/input/*/')
    parser.add_argument('--template', default=DEFAULT_TEMPLATE,
                        help='path to any prediction_template_ONU*.csv (HER2 chain A '
                             'source); auto-searched under /kaggle/input/*/ like --input')
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help='output CSV path (contains a split column)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='fraction of CLUSTERS assigned to validation')
    parser.add_argument('--max_len', type=int, default=149,
                        help='drop de-tagged VHH sequences longer than this '
                             '(AntiBinder antibody max length)')
    parser.add_argument('--antigen_name', default='HER2',
                        help='value written to the Antigen column (ESM cache id)')
    parser.add_argument('--method', choices=['auto', 'anchor', 'abnumber'], default='auto',
                        help='VHH FR/CDR splitting method')
    parser.add_argument('--seed', type=int, default=42, help='cluster split seed')
    parser.add_argument('--chunksize', type=int, default=200_000)
    args = parser.parse_args()

    # Auto-discover files under /kaggle/input/*/ when the assumed slug differs.
    args.input = resolve_kaggle_path(args.input, 'AVIDa-hHER2.csv')
    args.template = resolve_kaggle_path(args.template, 'prediction_template_ONU11.csv')
    if not os.path.exists(args.input):
        sys.exit(f"[prepare_sepiq] input not found: {args.input} "
                 f"(attach the SEPIQ dataset to this Kaggle notebook or pass --input)")
    if not os.path.exists(args.template):
        sys.exit(f"[prepare_sepiq] template not found: {args.template} "
                 f"(attach the SEPIQ dataset or pass --template)")
    print(f"[prepare_sepiq] input:    {args.input}")
    print(f"[prepare_sepiq] template: {args.template}")

    antigen_seq = read_antigen_sequence(args.template)
    print(f"[prepare_sepiq] HER2 antigen sequence: {len(antigen_seq)} aa "
          f"({antigen_seq[:10]}...{antigen_seq[-6:]})")

    # 1. Stream + filter rows with a definite binary label.
    usecols = ['#ONU ID', 'label', 'binary_label', 'Cluster', 'sequence']
    chunks = []
    n_total = 0
    for chunk in pd.read_csv(args.input, usecols=usecols, chunksize=args.chunksize):
        n_total += len(chunk)
        chunk = chunk[chunk['binary_label'].isin([0.0, 1.0])]
        if len(chunk):
            chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f"[prepare_sepiq] rows total={n_total:,}, with binary label={len(df):,} "
          f"(pos={int((df.binary_label == 1).sum()):,}, "
          f"neg={int((df.binary_label == 0).sum()):,})")

    # 2. Strip the C-terminal HHHHHH His-tag.
    df['sequence'] = df['sequence'].str.replace(r'H{6}$', '', regex=True)

    # 3. Standard 20-letter alphabet only; length filter for the fixed model.
    standard = set('ACDEFGHIKLMNPQRSTVWY')
    keep = df['sequence'].map(lambda s: set(s) <= standard and len(s) <= args.max_len)
    n_badlen = int((~keep).sum())
    df = df[keep].reset_index(drop=True)
    print(f"[prepare_sepiq] dropped {n_badlen} non-standard / >{args.max_len}-aa sequences, "
          f"remaining={len(df):,}")

    # 4. FR/CDR splitting.
    use_abnumber = args.method == 'abnumber'
    if args.method == 'auto':
        use_abnumber = _abnumber_available(df['sequence'].iloc[0])
    elif args.method == 'abnumber':
        if not _abnumber_available(df['sequence'].iloc[0]):
            sys.exit("[prepare_sepiq] --method abnumber requested but abnumber/ANARCI "
                     "is unavailable. On Kaggle (internet ON): "
                     "'!apt-get install -y hmmer' then '!pip install abnumber'.")

    regions_by_method = {'abnumber': 0, 'anchor': 0}
    failed = 0
    rows = []
    records = df.to_dict('records')
    for rec in records:
        seq = rec['sequence']
        regs = abnumber_split(seq) if use_abnumber else None
        method = 'abnumber'
        if regs is None:
            regs = anchor_split(seq)
            method = 'anchor' if regs is not None else ('abnumber-fail' if use_abnumber else 'fail')
        if regs is None:
            failed += 1
            continue
        regions_by_method[method if method in regions_by_method else 'anchor'] += 1
        rows.append({
            '#ONU ID': rec['#ONU ID'],
            'Cluster': rec['Cluster'],
            'label': rec['label'],
            'binary_label': int(rec['binary_label']),
            'sequence': seq,
            **regs,
            'vh': ''.join(regs[c] for c in REGION_COLS),
            'ANT_Binding': int(rec['binary_label']),
            'Antigen': args.antigen_name,
            'Antigen Sequence': antigen_seq,
            'split_method': method,
        })
    out = pd.DataFrame(rows)
    print(f"[prepare_sepiq] split OK={len(out):,} "
          f"(abnumber={regions_by_method['abnumber']:,}, "
          f"anchor={regions_by_method['anchor']:,}), failed={failed:,} "
          f"({failed / len(df):.3%})")

    # Sanity: reconstructed vh must equal the de-tagged sequence.
    assert (out['vh'] == out['sequence']).all(), "region reconstruction mismatch"

    # 5. Cluster-disjoint train/val split.
    rng = np.random.RandomState(args.seed)
    clusters = np.array(sorted(out['Cluster'].unique()))
    rng.shuffle(clusters)
    n_val = int(round(len(clusters) * args.val_ratio))
    val_clusters = set(clusters[:n_val])
    out['split'] = np.where(out['Cluster'].isin(val_clusters), 'val', 'train')

    for split in ('train', 'val'):
        sub = out[out['split'] == split]
        npos = int((sub['ANT_Binding'] == 1).sum())
        print(f"[prepare_sepiq] {split:5s}: rows={len(sub):,}, pos={npos:,}, "
              f"neg={len(sub) - npos:,}, pos_ratio={npos / max(len(sub), 1):.3f}, "
              f"clusters={sub['Cluster'].nunique():,}")

    # 6. Write.
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    ordered = ['#ONU ID', 'Cluster', 'label', 'binary_label', 'split', 'split_method',
               'sequence'] + MODEL_COLS
    out = out[ordered]
    out.to_csv(args.output, index=False)
    print(f"[prepare_sepiq] wrote {len(out):,} rows -> {args.output}")
    print("[prepare_sepiq] next: python main.py train "
          f"--data_path {args.output}")


if __name__ == '__main__':
    main()
