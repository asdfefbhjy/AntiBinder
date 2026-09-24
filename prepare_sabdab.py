"""Stage-1 data preparation: per-residue interface labels from the official
SAbDab2 Machine Learning Dataset (Zenodo record 22019991, v0.2.0).

Why this source
---------------
The legacy OPIg SAbDab(-nano) summary endpoints were retired when SAbDab2 was
rebuilt (https://sabdab2.opig.stats.ox.ac.uk/). OPIG now publish a single
curated ML bundle on Zenodo:

    https://zenodo.org/records/22019991
    splits.tar.gz  (~889 MB, md5 ada2fdd573418877d7eaa87afe92044e)

It contains 15,810 pre-cropped structure files, including 3,318 single-domain
(VHH-like) antibodies (~2,083 with an antigen), plus standardised
sequence-clustered train/test splits. We use the antigen-aware single-domain
split (``abag_split_sd.csv``): antibody AND antigen sequence similarity were
considered, so train/test leakage on either side is prevented upstream.

File naming inside the archive::

    splits_final/pdb_<index><pdbid>_<abChain1>_<abChain2>.cif
    e.g. pdb_00007zml_E_+.cif   -> VHH (single heavy) chain E, '+' = no light
         pdb_00002a6j_H_L.cif  -> conventional Fv (ignored here)

Each file keeps the antibody variable region and the (possibly cropped)
antigen chain coordinates; apo VHH files simply have no other polymer chain.

Outputs (under --out_dir):
  sabdab_nano.csv          columns consumed by `main.py train --task struct`
                           (H-FR1..H-FR4, vh, Antigen, Antigen Sequence,
                            sample_id, pdb, split=train/val)
  res_labels/<sample_id>.npz
                           ab_label: (len(vh),) float32 0/1 paratope
                           ag_label: (len(ag),) float32 0/1 epitope

The validation split is carved out of the OFFICIAL train structures by
unique antigen sequence (seeded), so no antigen (or near-identical antigen,
thanks to abag clustering) crosses train/val. Official test structures are
excluded by default (pass --include_official_test to fold them into train for
a final submission model).

Kaggle usage (internet ON, biopython installed)::

    !pip install biopython -q
    !python prepare_sabdab.py --out_dir ./datasets/process_data/SAbDab
    !python main.py train --task struct --model_name sabdabstruct \
        --data_path ./datasets/process_data/SAbDab/sabdab_nano.csv \
        --res_label_dir ./datasets/process_data/SAbDab/res_labels

Offline mode: pre-download splits.tar.gz yourself and pass --archive.
"""
import argparse
import hashlib
import io
import os
import random
import re
import shutil
import subprocess
import tarfile
import time
import urllib.request

import numpy as np
import pandas as pd

from prepare_sepiq import anchor_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ZENODO_URL = 'https://zenodo.org/records/22019991/files/splits.tar.gz?download=1'
ZENODO_MD5 = 'ada2fdd573418877d7eaa87afe92044e'  # v0.2.0 (20 Aug 2026)

REGION_COLS = ['H-FR1', 'H-CDR1', 'H-FR2', 'H-CDR2', 'H-FR3', 'H-CDR3', 'H-FR4']
MAX_VH_LEN = 149          # model antibody max length
MIN_VH_LEN = 90
MAX_AG_LEN = 1000         # keep margin below the 1024 ESM window
MIN_AG_LEN = 20
MIN_CONTACTS = 3          # require a real interface on each side
CONTACT_DIST = 5.0        # heavy-atom contact cutoff (Angstrom)

# pdb_00007zml_E_+.cif  ->  ('7zml', 'E', '+')
MEMBER_RE = re.compile(
    r'^pdb_(?P<idx>\d{4})(?P<pdb>[0-9a-z]{4})_(?P<c1>[^_]+)_(?P<c2>[^_]+)\.cif$')

THREE2ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V', 'MSE': 'M', 'SEC': 'U', 'PYL': 'O',
}


# ---------------------------------------------------------------------------
# Download (stdlib-first; curl fallback) with resume + md5 verification
# ---------------------------------------------------------------------------
def _md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def _download_urllib(url, dst, pos):
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; AntiBinder-prep/1.0)',
               'Accept': 'application/octet-stream,*/*;q=0.8'}
    if pos:
        headers['Range'] = f'bytes={pos}-'
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, 'ab') as f:
        shutil.copyfileobj(r, f, length=1 << 20)


def _download_curl(url, dst, pos):
    cmd = ['curl', '-fL', '--retry', '3', '--connect-timeout', '60',
           '--output', dst, url]
    if pos:
        cmd[-1:-1] = ['-C', '-']  # resume
    subprocess.run(cmd, check=True)


def download_archive(dst):
    """Download (resumably) the Zenodo bundle and verify the md5."""
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f'[download] found existing archive ({os.path.getsize(dst)/1e6:.0f} MB), '
              f'verifying md5...')
        if _md5(dst) == ZENODO_MD5:
            print('[download] md5 OK, reusing.')
            return dst
        print('[download] md5 mismatch / partial file; resuming download.')

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    part = dst + '.part'
    pos = os.path.getsize(part) if os.path.exists(part) else 0
    backoff = 10
    for attempt in range(6):
        try:
            print(f'[download] urllib GET {url_clip(ZENODO_URL)} (resume @ {pos/1e6:.0f} MB)')
            _download_urllib(ZENODO_URL, part, pos)
            break
        except Exception as exc:
            print(f'[download] urllib failed ({type(exc).__name__}: {exc}); '
                  f'trying curl in {backoff}s ...')
            try:
                _download_curl(ZENODO_URL, part, pos)
                break
            except Exception as exc2:
                print(f'[download] curl failed ({type(exc2).__name__}: {exc2}); '
                      f'retry {attempt+1}/6 in {backoff}s')
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)
                pos = os.path.getsize(part) if os.path.exists(part) else 0
    else:
        raise RuntimeError('Could not download splits.tar.gz from Zenodo after retries.')

    if _md5(part) != ZENODO_MD5:
        raise RuntimeError(f'Downloaded archive md5 mismatch (expected {ZENODO_MD5}). '
                           f'Delete {part} and retry.')
    shutil.move(part, dst)
    print(f'[download] complete: {dst} ({os.path.getsize(dst)/1e6:.0f} MB)')
    return dst


def url_clip(u):
    return u.split('?')[0]


# ---------------------------------------------------------------------------
# Structure parsing / contacts
# ---------------------------------------------------------------------------
def chain_sequence_and_atoms(chain):
    """Sequence (one-letter, file order) + per-residue heavy-atom coordinate
    arrays for a Bio.PDB chain. Water/ligand HETATM residues are skipped
    (MSE -> M)."""
    seq, coords = [], []
    for res in chain:
        hetflag = res.id[0]
        if hetflag.strip() and res.get_resname() != 'MSE':
            continue
        three = res.get_resname().upper()
        if three not in THREE2ONE:
            continue
        atoms = [a for a in res if a.element != 'H']
        if not atoms:
            continue
        seq.append(THREE2ONE[three])
        coords.append(np.array([a.coord for a in atoms], dtype=np.float32))
    return ''.join(seq), coords


def contact_pairs(coords_a, coords_b, dist=CONTACT_DIST):
    """Index sets of residues in A/B with any heavy-atom pair < dist."""
    from scipy.spatial import cKDTree
    tree_b = cKDTree(np.concatenate(coords_b, axis=0))
    sizes_b = [len(c) for c in coords_b]
    b_res_of_atom = np.concatenate(
        [np.full(n, i, dtype=np.int32) for i, n in enumerate(sizes_b)])

    contact_a, contact_b = set(), set()
    for i, ca in enumerate(coords_a):
        pairs = tree_b.query_ball_point(ca, r=dist)
        hits = {b_res_of_atom[j] for sub in pairs for j in sub}
        if hits:
            contact_a.add(i)
            contact_b.update(hits)
    return contact_a, contact_b


def extract_vhh(text, ab_chain):
    """Parse one SAbDab2 cif. Returns (vh_seq, candidates) where candidates is
    a list of (chain_id, ag_seq, ab_contact_idx, ag_contact_idx) computed for
    EVERY polymer chain other than the VHH (the curated subset is selected at
    join time). Returns (None, reason) when the VHH chain is unusable."""
    from Bio.PDB import MMCIFParser
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('x', io.StringIO(text))
    model = next(structure.get_models())

    vh = None
    raw_cands = []
    for ch in model:
        seq, coords = chain_sequence_and_atoms(ch)
        if not seq:
            continue
        if ch.id == ab_chain:
            vh = (seq, coords)
        elif 5 <= len(seq) <= MAX_AG_LEN:
            # README: retained polymer antigen chains are >= 5 standard,
            # contiguous residues (includes short PEPTIDE antigens).
            raw_cands.append((ch.id, seq, coords))

    if vh is None or not (MIN_VH_LEN <= len(vh[0]) <= MAX_VH_LEN):
        return None, f'no VHH chain {ab_chain!r} in {MIN_VH_LEN}-{MAX_VH_LEN} aa range'
    vh_seq, vh_coords = vh
    candidates = []
    for ch_id, ag_seq, ag_coords in raw_cands:
        ab_idx, ag_idx = contact_pairs(vh_coords, ag_coords)
        candidates.append((ch_id, ag_seq, ab_idx, ag_idx))
    return (vh_seq, candidates), None


# ---------------------------------------------------------------------------
# Official split metadata (abag_split_sd.csv; schema documented in README.md)
# ---------------------------------------------------------------------------
SD_CSV = 'abag_split_sd.csv'
META_COLS = ['INSTANCE', 'type', 'holo', 'Hchain', 'agchains', 'agtypes',
             'agresolvedseqs', 'ab_ag_split']
POLYMER_TYPES = {'PROTEIN', 'PEPTIDE'}  # residue-level epitope labels need a
#                                       polymer antigen (DNA/RNA/sugar/hapten/
#                                       ion complexes are excluded)


def load_sd_metadata(csv_blobs):
    """Parse the single-domain ab-ag split into {INSTANCE: meta}.

    The CSV ships heavy python-list numbering columns (26 MB), so read only
    the documented columns as strings. See README.md in the archive."""
    if SD_CSV not in csv_blobs:
        raise RuntimeError(f'{SD_CSV} missing from archive (found: '
                           f'{sorted(csv_blobs)})')
    df = pd.read_csv(io.StringIO(csv_blobs[SD_CSV]), usecols=META_COLS,
                     dtype=str).fillna('')
    meta = {}
    counts = {'sdh_holo_polymer': 0, 'sdh_other': 0, 'sdl': 0}
    for r in df.to_dict('records'):
        inst = r['INSTANCE'].strip()
        stype = r['type'].strip().upper()
        if stype == 'SD-L':
            counts['sdl'] += 1
            continue
        holo = r['holo'].strip().upper() == 'TRUE'
        ag_chains = [c.strip() for c in r['agchains'].split('/') if c.strip()]
        ag_types = [t.strip().upper() for t in r['agtypes'].split('/')]
        poly = [(c, t) for c, t in zip(ag_chains, ag_types)
                if t in POLYMER_TYPES]
        split = r['ab_ag_split'].strip().lower()
        meta[inst] = {
            'split': split,
            'holo': holo,
            'poly': poly,  # [(author_chain_id, 'PROTEIN'|'PEPTIDE'), ...]
            'all_ag_chains': ag_chains,
        }
        if stype == 'SD-H' and holo and poly:
            counts['sdh_holo_polymer'] += 1
        elif stype == 'SD-H':
            counts['sdh_other'] += 1
    print(f'[split] {SD_CSV}: {counts}')
    n_train = sum(1 for m in meta.values()
                  if m['split'] == 'train' and m['holo'] and m['poly'])
    n_test = sum(1 for m in meta.values()
                 if m['split'] == 'test' and m['holo'] and m['poly'])
    print(f'[split] holo SD-H + polymer antigen: train={n_train}, test={n_test}')
    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--archive', default='./sabdab_structures/splits.tar.gz',
                    help='path to splits.tar.gz (downloaded here if missing)')
    ap.add_argument('--out_dir', default='./datasets/process_data/SAbDab',
                    help='output directory (CSV + res_labels/*.npz)')
    ap.add_argument('--skip_download', action='store_true',
                    help='do not attempt any download (--archive must exist)')
    ap.add_argument('--include_official_test', action='store_true',
                    help='fold the official TEST structures into training data '
                         '(for a final submission model; they are excluded by default)')
    ap.add_argument('--val_ratio', type=float, default=0.15,
                    help='fraction of unique antigens from OFFICIAL train used '
                         'for the validation split')
    ap.add_argument('--max_structures', type=int, default=None,
                    help='optional cap on kept complexes (debugging)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    res_dir = os.path.join(args.out_dir, 'res_labels')
    os.makedirs(res_dir, exist_ok=True)

    if not args.skip_download:
        archive = download_archive(args.archive)
    else:
        archive = args.archive
        if not os.path.exists(archive):
            raise FileNotFoundError(f'--archive not found: {archive}')

    # ---- one streaming pass over the gzip tar -----------------------------
    # All single-domain-heavy cifs are parsed (the split CSV sits at the END
    # of the archive, so metadata is not available yet); only tiny sequences
    # and contact index sets are retained, not the raw cif text.
    records = {}          # INSTANCE (exact case) -> (pdb, ab_chain, vh_seq, candidates)
    csv_blobs = {}        # basename -> text
    skipped = {}
    n_seen = n_vhh = 0

    with tarfile.open(archive, mode='r:gz') as tf:
        for member in tf:
            if not member.isfile():
                continue
            base = os.path.basename(member.name)
            n_seen += 1
            if n_seen % 2000 == 0:
                print(f'... scanned {n_seen} files, VHH parsed {len(records)} '
                      f'(skipped {sum(skipped.values())})')

            if base.endswith(('.csv', '.md', '.txt')):
                f = tf.extractfile(member)
                if f is not None:
                    csv_blobs[base] = f.read().decode('utf-8', errors='replace')
                continue

            m = MEMBER_RE.match(base)
            if m is None or m.group('c2') != '+':
                continue  # conventional Fv, single-domain light, or other file
            n_vhh += 1
            f = tf.extractfile(member)
            if f is None:
                continue
            text = f.read().decode('utf-8', errors='replace')
            try:
                result, reason = extract_vhh(text, m.group('c1'))
            except Exception as exc:
                result, reason = None, f'{type(exc).__name__}: {exc}'
            if result is None:
                key = reason.split(' ')[0]
                skipped[key] = skipped.get(key, 0) + 1
                continue
            vh_seq, candidates = result
            records[base[:-4]] = (m.group('pdb'), m.group('c1'),
                                  vh_seq, candidates)

    print(f'\n[tar] scanned {n_seen} files; {n_vhh} single-domain-heavy members, '
          f'{len(records)} parsed successfully.')
    print(f'[tar] parse skips: {skipped}')
    print(f'[tar] metadata files: {sorted(csv_blobs)}')

    # ---- join the official ab-ag single-domain metadata -------------------
    meta = load_sd_metadata(csv_blobs)
    rows = []
    join_skip = {}
    n_test = 0
    for inst, info in meta.items():
        if not info['holo'] or not info['poly']:
            continue  # apo VHH or non-polymer antigen only (sugar/hapten/...)
        if info['split'] not in ('train', 'test'):
            join_skip[f"split={info['split']}"] = join_skip.get(
                f"split={info['split']}", 0) + 1
            continue
        if info['split'] == 'test' and not args.include_official_test:
            n_test += 1
            continue
        rec = records.get(inst)
        if rec is None:
            join_skip['cif_not_parsed'] = join_skip.get('cif_not_parsed', 0) + 1
            continue
        pdb, ab_chain, vh_seq, candidates = rec
        type_by_chain = dict(info['poly'])
        # Restrict to curated PROTEIN/PEPTIDE chains; proteins >= MIN_AG_LEN,
        # short PEPTIDE antigens allowed down to 5 residues.
        eligible = [
            (cid, seq, ab_idx, ag_idx)
            for cid, seq, ab_idx, ag_idx in candidates
            if cid in type_by_chain
            and (type_by_chain[cid] == 'PEPTIDE' or len(seq) >= MIN_AG_LEN)]
        if not eligible:
            join_skip['no_curated_chain_coords'] = join_skip.get(
                'no_curated_chain_coords', 0) + 1
            continue
        # Primary antigen: most contacting residues, then most paratope
        # contacts (deterministic tie-break by chain id).
        cid, ag_seq, ab_idx, ag_idx = sorted(
            eligible, key=lambda x: (-len(x[3]), -len(x[2]), x[0]))[0]
        if len(ab_idx) < MIN_CONTACTS or len(ag_idx) < MIN_CONTACTS:
            join_skip['too_few_contacts'] = join_skip.get('too_few_contacts', 0) + 1
            continue

        regions = anchor_split(vh_seq)
        if regions is None:
            join_skip['anchor_split'] = join_skip.get('anchor_split', 0) + 1
            continue
        # anchor_split tolerates resolved cloning tails AFTER FR4; the
        # concatenated regions are the clean VHH. FR1 always starts at
        # position 0, so indices need only a right-side filter (no shift).
        vh_clean = ''.join(regions[k] for k in REGION_COLS)
        ab_idx = {i for i in ab_idx if i < len(vh_clean)}
        if len(ab_idx) < MIN_CONTACTS:
            join_skip['too_few_contacts'] = join_skip.get('too_few_contacts', 0) + 1
            continue
        ag_name = 'AG' + hashlib.md5(ag_seq.encode()).hexdigest()[:10]
        rows.append({
            **regions,
            'vh': vh_clean,
            'Antigen': ag_name,
            'Antigen Sequence': ag_seq,
            '_sid': f'{pdb}_{ab_chain}',
            '_inst': inst,
            'pdb': pdb,
            '_official': info['split'],
            '_ab_idx': ab_idx,
            '_ag_idx': ag_idx,
        })

    if not rows:
        raise RuntimeError('No usable VHH-antigen complexes after join. '
                           f'Skip counts: {join_skip}')
    print(f'[join] kept {len(rows)}; skips: {join_skip}')
    if n_test:
        print(f'[join] {n_test} official-test complexes excluded '
              f'(--include_official_test to keep them).')

    # ---- sample ids, safe even on case-insensitive filesystems ------------
    # PDB chain letters are case-sensitive: 7nvm_n and 7nvm_N are different
    # chains, but on Windows/macOS 7nvm_n.npz == 7nvm_N.npz. Colliding groups
    # get a short hash of the full INSTANCE stem as suffix.
    sid_groups = {}
    for r in rows:
        sid_groups.setdefault(r['_sid'].lower(), []).append(r)
    n_coll = 0
    for grp in sid_groups.values():
        if len(grp) == 1:
            grp[0]['sample_id'] = grp[0]['_sid']
        else:
            n_coll += len(grp)
            for r in grp:
                # Encode chain-letter case with digits (0 lower / 1 upper) so
                # ids stay distinct on case-insensitive filesystems.
                chain = r['_sid'].rsplit('_', 1)[1]
                enc = ''.join(f'{c}0' if c.islower() else f'{c.lower()}1'
                              for c in chain)
                suffix = hashlib.md5(r['_inst'].encode()).hexdigest()[:4]
                r['sample_id'] = f"{r['pdb']}_{enc}_{suffix}"
    if n_coll:
        print(f'[join] {n_coll} rows given case-collision-safe sample ids.')

    # ---- val carved from OFFICIAL train by unique antigen -----------------
    # Official-test rows (only present with --include_official_test) always
    # stay 'train'; they never define or join the validation set.
    antigens = sorted({r['Antigen Sequence'] for r in rows
                       if r['_official'] == 'train'})
    rng = random.Random(args.seed)
    rng.shuffle(antigens)
    n_val = max(1, int(round(len(antigens) * args.val_ratio)))
    val_ags = set(antigens[:n_val])
    for r in rows:
        if r['_official'] == 'train' and r['Antigen Sequence'] in val_ags:
            r['split'] = 'val'
        else:
            r['split'] = 'train'

    if args.max_structures and len(rows) > args.max_structures:
        rng2 = random.Random(args.seed)
        rows = rng2.sample(rows, args.max_structures)

    # ---- write labels + CSV ------------------------------------------------
    out_rows = []
    for r in rows:
        npz_path = os.path.join(res_dir, f"{r['sample_id']}.npz")
        ab_label = np.zeros(len(r['vh']), dtype=np.float32)
        ab_label[sorted(r['_ab_idx'])] = 1.0
        ag_label = np.zeros(len(r['Antigen Sequence']), dtype=np.float32)
        ag_label[sorted(r['_ag_idx'])] = 1.0
        np.savez_compressed(npz_path, ab_label=ab_label, ag_label=ag_label)
        out_rows.append({k: r[k] for k in
                         REGION_COLS + ['vh', 'Antigen', 'Antigen Sequence',
                                        'sample_id', 'pdb', 'split']})

    out = pd.DataFrame(out_rows)
    csv_path = os.path.join(args.out_dir, 'sabdab_nano.csv')
    out.to_csv(csv_path, index=False)

    n_val_rows = int((out['split'] == 'val').sum())
    print(f'\nDone: {len(out)} complexes (train {len(out)-n_val_rows} / val {n_val_rows}, '
          f'{len(antigens)-n_val} train antigens / {n_val} val antigens)')
    print(f'CSV    -> {csv_path}')
    print(f'Labels -> {res_dir}/')
    print('\nNext (stage-1):')
    print(f'  python main.py train --task struct --model_name sabdabstruct '
          f'--data_path {csv_path} --res_label_dir {res_dir}')


if __name__ == '__main__':
    main()
