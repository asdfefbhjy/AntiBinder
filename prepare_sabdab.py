"""Stage-1 data preparation: per-residue interface labels from SAbDab-nano.

Builds the residue-supervised (task=struct) training set for SEPIQ Track A:
for each VHH-antigen structure we compute inter-chain heavy-atom contacts and
store per-residue binary interface labels, aligned with the sequence positions
the model actually consumes.

Outputs (under --out_dir):
  sabdab_nano.csv   master CSV (H-FR1..H-FR4, vh, Antigen, Antigen Sequence,
                    sample_id, split) consumable by `main.py train --task struct`
  res_labels/<sample_id>.npz   per-sample labels:
                    ab_label: (len(vh),)  float32 0/1 paratope
                    ag_label: (len(ag),)  float32 0/1 epitope

Usage (Kaggle, internet on):
    !pip install biopython -q
    # 1) try to fetch the SAbDab-nano summary (server can be slow; several min)
    !python prepare_sabdab.py --fetch-summary --out_dir ./datasets/process_data/SAbDab
    # 2) download structures + build labels
    !python prepare_sabdab.py --out_dir ./datasets/process_data/SAbDab

Offline mode: place a SAbDab summary CSV (columns pdb, Hchain, Lchain,
antigen_chain) at --summary and pre-downloaded *.cif[.gz] files in --pdb_dir.
"""
import argparse
import gzip
import hashlib
import os
import random
import shutil
import sys
import urllib.request
import zlib

import numpy as np
import pandas as pd

from prepare_sepiq import anchor_split

REGION_COLS = ['H-FR1', 'H-CDR1', 'H-FR2', 'H-CDR2', 'H-FR3', 'H-CDR3', 'H-FR4']
MAX_VH_LEN = 149          # model antibody max length
MAX_AG_LEN = 1000         # keep margin below the 1024 ESM window
CONTACT_DIST = 5.0        # heavy-atom contact cutoff (Angstrom)

SUMMARY_URLS = [
    'https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/nanobodies/summary/all/',
    'https://opig.stats.ox.ac.uk/webapps/sabdab/summary/all',
]

THREE2ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V', 'MSE': 'M', 'SEC': 'U', 'PYL': 'O',
}
NO_LIGHT = {'', 'NA', 'N/A', 'NAN', '-', 'NONE', '.'}


# ---------------------------------------------------------------------------
# summary handling
# ---------------------------------------------------------------------------
def fetch_summary(out_path):
    """Try known SAbDab summary endpoints; the OPIg server generates the file
    on demand and can take several minutes, hence the long timeout."""
    for url in SUMMARY_URLS:
        try:
            print(f'[summary] downloading {url} (this can take minutes)...')
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=1800) as r:
                data = r.read()
            with open(out_path, 'wb') as f:
                f.write(data)
            print(f'[summary] saved {len(data)/1e6:.1f} MB -> {out_path}')
            return out_path
        except Exception as exc:
            print(f'[summary] FAILED {url}: {exc}')
    print('[summary] All endpoints failed. Download the SAbDab-nano summary manually from '
          'https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/nanobodies/ '
          '(or https://opig.stats.ox.ac.uk/webapps/sabdab/) and pass it via --summary.')
    return None


def pick_col(df, candidates):
    lower = {c.lower().replace('_', ''): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace('_', '')
        if key in lower:
            return lower[key]
    return None


def is_missing(value):
    return str(value).strip().upper() in NO_LIGHT or pd.isna(value)


def load_complex_list(summary_path):
    """Filter the summary down to single-domain-antibody (nanobody/VHH) +
    single-antigen-chain complexes."""
    df = pd.read_csv(summary_path, sep=None, engine='python', nrows=None)
    df.columns = [str(c).strip() for c in df.columns]
    c_pdb = pick_col(df, ['pdb', 'pdbcode'])
    c_h = pick_col(df, ['Hchain', 'heavychain'])
    c_l = pick_col(df, ['Lchain', 'lightchain'])
    c_ag = pick_col(df, ['antigenchain', 'agchain', 'antigenchains'])
    if not all([c_pdb, c_h, c_ag]):
        raise ValueError(f'Summary {summary_path} lacks pdb/Hchain/antigen_chain columns; '
                         f'found: {list(df.columns)[:20]}')
    c_l = c_l or '__none__'
    df[c_pdb] = df[c_pdb].astype(str).str.strip()
    keep = []
    for _, r in df.iterrows():
        pdb = r[c_pdb]
        hchain = r.get(c_h)
        ag = r.get(c_ag)
        if is_missing(pdb) or is_missing(hchain) or is_missing(ag):
            continue
        lchain = r.get(c_l, 'NA') if c_l != '__none__' else 'NA'
        if not is_missing(lchain):
            continue  # paired light chain -> conventional antibody, skip
        ag_chains = [c for c in str(ag).replace(';', ',').split(',') if c.strip()]
        ag_chains = [c.strip() for c in ag_chains if not is_missing(c)]
        if len(ag_chains) != 1:
            continue  # multi-chain antigen: keep the label space unambiguous
        keep.append({'pdb': pdb.lower(), 'Hchain': str(hchain).strip(), 'ag_chain': ag_chains[0]})
    entries = pd.DataFrame(keep).drop_duplicates(subset=['pdb']).reset_index(drop=True)
    print(f'[summary] {len(df)} rows -> {len(entries)} nanobody + single-antigen complexes')
    return entries


# ---------------------------------------------------------------------------
# structure parsing / contacts
# ---------------------------------------------------------------------------
def chain_sequence_and_atoms(chain):
    """Sequence (one-letter, file order) + per-residue heavy-atom coordinate
    arrays for a Bio.PDB chain. Non-standard residues are skipped (MSE->M)."""
    seq, coords = [], []
    for res in chain:
        hetflag, resnum, icode = res.id
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
    """Index sets of residues in A/B that have any heavy-atom pair < dist."""
    from scipy.spatial import cKDTree  # scipy ships with most ML environments
    tree_b = cKDTree(np.concatenate(coords_b, axis=0))
    sizes_b = [len(c) for c in coords_b]
    offsets_b = np.concatenate([[0], np.cumsum(sizes_b)])
    b_res_of_atom = np.concatenate([np.full(n, i, dtype=np.int32) for i, n in enumerate(sizes_b)])

    contact_a, contact_b = set(), set()
    for i, ca in enumerate(coords_a):
        pairs = tree_b.query_ball_point(ca, r=dist)
        hits = sorted({b_res_of_atom[j] for sub in pairs for j in sub})
        if hits:
            contact_a.add(i)
            contact_b.update(hits)
    return contact_a, contact_b


def download_structure(pdb_id, pdb_dir):
    """Download (and cache) the mmcif for a PDB entry. Returns a local path or
    None."""
    os.makedirs(pdb_dir, exist_ok=True)
    for suffix in ('.cif.gz', '.cif'):
        path = os.path.join(pdb_dir, pdb_id + suffix)
        if os.path.exists(path):
            return path
    url = f'https://files.rcsb.org/download/{pdb_id}.cif.gz'
    path = os.path.join(pdb_dir, pdb_id + '.cif.gz')
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=120) as r, open(path, 'wb') as f:
            shutil.copyfileobj(r, f)
        return path
    except Exception as exc:
        print(f'  [download] {pdb_id} failed: {exc}')
        return None


def parse_contacts(cif_path, hchain_id, ag_chain_id):
    """(vh_seq, ag_seq, ab_contact_idx, ag_contact_idx) or (None, reason)."""
    from Bio.PDB import MMCIFParser, PDBParser
    if cif_path.endswith('.gz'):
        with open(cif_path, 'rb') as fh:
            raw = fh.read()
        try:
            raw = gzip.decompress(raw)
        except (gzip.BadGzipFile, zlib.error):
            pass  # already plain text
        tmp = cif_path[:-3] if cif_path.endswith('.gz') else cif_path
        if not os.path.exists(tmp):
            with open(tmp, 'wb') as f:
                f.write(raw)
        cif_path = tmp
        parser = MMCIFParser(QUIET=True)
    else:
        ext = os.path.splitext(cif_path)[1].lower()
        parser = MMCIFParser(QUIET=True) if ext == '.cif' else PDBParser(QUIET=True)
    structure = parser.get_structure('x', cif_path)
    model = next(structure.get_models())

    chains = {ch.id.strip(): ch for ch in model}
    if hchain_id not in chains or ag_chain_id not in chains:
        return None, f'chains {hchain_id}/{ag_chain_id} not in {list(chains)[:6]}'

    vh_seq, vh_coords = chain_sequence_and_atoms(chains[hchain_id])
    ag_seq, ag_coords = chain_sequence_and_atoms(chains[ag_chain_id])
    if not (90 <= len(vh_seq) <= MAX_VH_LEN):
        return None, f'vh length {len(vh_seq)} out of range'
    if len(ag_seq) < 20 or len(ag_seq) > MAX_AG_LEN:
        return None, f'antigen length {len(ag_seq)} out of range'

    ab_idx, ag_idx = contact_pairs(vh_coords, ag_coords)
    if len(ab_idx) < 3 or len(ag_idx) < 3:
        return None, f'too few contacts (ab={len(ab_idx)}, ag={len(ag_idx)})'
    return (vh_seq, ag_seq, ab_idx, ag_idx), None


# ---------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--summary', default=None,
                    help='path to a SAbDab(-nano) summary CSV; combined with --fetch-summary '
                         'this is where the download is stored')
    ap.add_argument('--fetch-summary', action='store_true',
                    help='download the SAbDab-nano summary before processing')
    ap.add_argument('--pdb_dir', default='./sabdab_structures',
                    help='cache directory for downloaded mmcif files')
    ap.add_argument('--out_dir', default='./datasets/process_data/SAbDab',
                    help='output directory (CSV + res_labels/npz)')
    ap.add_argument('--max_structures', type=int, default=3000,
                    help='cap on complexes processed (None-safe memory/time budget)')
    ap.add_argument('--val_ratio', type=float, default=0.15,
                    help='fraction of unique ANTIGENS held out for validation')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    res_dir = os.path.join(args.out_dir, 'res_labels')
    os.makedirs(res_dir, exist_ok=True)

    summary_path = args.summary or os.path.join(args.pdb_dir, 'sabdab_summary.csv')
    if args.fetch_summary:
        got = fetch_summary(summary_path)
        if got is None:
            sys.exit(1)
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f'Summary CSV not found: {summary_path}. Run with --fetch-summary (internet) '
            f'or download it manually and pass --summary.')

    entries = load_complex_list(summary_path)
    if args.max_structures and len(entries) > args.max_structures:
        entries = entries.sample(n=args.max_structures, random_state=args.seed)
        print(f'[cap] sampled {len(entries)} complexes (--max_structures)')

    rows, skipped = [], {}
    for n, ent in enumerate(entries.to_dict('records')):
        pdb, hchain, ag_chain = ent['pdb'], ent['Hchain'], ent['ag_chain']
        sample_id = f"{pdb}_{hchain}"
        if (n + 1) % 100 == 0:
            print(f'... {n+1}/{len(entries)} (kept {len(rows)}, skipped {sum(skipped.values())})')
        npz_path = os.path.join(res_dir, sample_id + '.npz')
        if os.path.exists(npz_path):
            pass  # recompute anyway is cheap only w/o download; keep simple: reuse cache below
        cif = download_structure(pdb, args.pdb_dir)
        if cif is None:
            skipped['download'] = skipped.get('download', 0) + 1
            continue
        try:
            result, reason = parse_contacts(cif, hchain, ag_chain)
        except Exception as exc:
            result, reason = None, f'{type(exc).__name__}: {exc}'
        if result is None:
            skipped['parse'] = skipped.get('parse', 0) + 1
            if reason and (n < 10 or skipped['parse'] <= 10):
                print(f'  [skip] {sample_id}: {reason}')
            continue
        vh_seq, ag_seq, ab_idx, ag_idx = result
        regions = anchor_split(vh_seq)
        if regions is None:
            skipped['anchor_split'] = skipped.get('anchor_split', 0) + 1
            continue
        ab_label = np.zeros(len(vh_seq), dtype=np.float32)
        ab_label[list(ab_idx)] = 1.0
        ag_label = np.zeros(len(ag_seq), dtype=np.float32)
        ag_label[list(ag_idx)] = 1.0
        np.savez_compressed(npz_path, ab_label=ab_label, ag_label=ag_label)
        ag_name = 'AG' + hashlib.md5(ag_seq.encode()).hexdigest()[:10]
        rows.append({**regions, 'vh': vh_seq, 'Antigen': ag_name, 'Antigen Sequence': ag_seq,
                     'sample_id': sample_id, 'pdb': pdb})

    if not rows:
        raise RuntimeError('No usable complexes produced. Check the skip log above.')

    # Split by unique ANTIGEN so all complexes against one antigen stay on the
    # same side (prevents antigen-level leakage between train and val).
    out = pd.DataFrame(rows)
    antigens = out['Antigen Sequence'].unique().tolist()
    rng = random.Random(args.seed)
    rng.shuffle(antigens)
    n_val = max(1, int(round(len(antigens) * args.val_ratio)))
    val_ags = set(antigens[:n_val])
    out['split'] = ['val' if ag in val_ags else 'train' for ag in out['Antigen Sequence']]
    out = out[['H-FR1', 'H-CDR1', 'H-FR2', 'H-CDR2', 'H-FR3', 'H-CDR3', 'H-FR4',
               'vh', 'Antigen', 'Antigen Sequence', 'sample_id', 'pdb', 'split']]
    csv_path = os.path.join(args.out_dir, 'sabdab_nano.csv')
    out.to_csv(csv_path, index=False)

    n_val_rows = int((out['split'] == 'val').sum())
    print(f'\nDone: {len(out)} complexes '
          f'(train {len(out) - n_val_rows} / val {n_val_rows}, '
          f'{len(antigens) - n_val} train antigens / {n_val} val antigens)')
    print(f'Skipped: {skipped}')
    print(f'CSV -> {csv_path}')
    print(f'Labels -> {res_dir}/')
    print('\nNext (stage-1):')
    print(f'  python main.py train --task struct --model_name sabdabstruct '
          f'--data_path {csv_path} --res_label_dir {res_dir}')


if __name__ == '__main__':
    main()
