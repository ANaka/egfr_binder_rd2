import os
import subprocess
import logging
import json
from datetime import datetime
from pathlib import Path
import tempfile
from typing import List, Union
from modal import Image, App, method, enter, Dict, Volume, Mount, interact
from egfr_binder_rd2 import (
    EGFS, 
    EGFR, 
    OFFICIAL_EGFR,
    COLABFOLD_GPU_CONCURRENCY_LIMIT, 
    MODAL_VOLUME_NAME,
    MSA_QUERY_HOST_URL,
    OUTPUT_DIRS,
    LOGGING_CONFIG,
    MODAL_VOLUME_PATH
)
from egfr_binder_rd2.utils import (
    get_mutation_diff, 
    hash_seq,
    get_fasta_path,
    get_folded_dir,
    swap_binder_seq_into_a3m
)
import io
import numpy as np
import pandas as pd
import re

from egfr_binder_rd2.solubility import calculate_solubility
EGFR_EPITOPE_RESIDUES = [11, 12, 13, 15, 16, 17, 18, 356, 440, 441]
INTERACTION_CUTOFF = 4.0  # Angstroms

# Define constants for the modal volume path


# Set up logging using configuration
logging.basicConfig(level=LOGGING_CONFIG["level"], format=LOGGING_CONFIG["format"])
logger = logging.getLogger(__name__)

app = App("simplefold")

# Initialize the Volume
volume = Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

image = (
    Image
    .debian_slim(python_version="3.11")
    .micromamba(python_version="3.11")
    .apt_install("wget", "git", "curl")
    .run_commands('wget https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/install_colabbatch_linux.sh')
    .run_commands('bash install_colabbatch_linux.sh', gpu="a100")
    .pip_install(
        'biopython==1.81',
        'pandas',
        'numpy'
    )
)

with image.imports():
    from Bio import PDB
    import numpy as np
    import pandas as pd
    from Bio.PDB import PDBParser, NeighborSearch
    from Bio.PDB import Selection
    from Bio.PDB.Polypeptide import is_aa


@app.cls(
    image=image,
    gpu="A10G",
    timeout=9600,
    concurrency_limit=COLABFOLD_GPU_CONCURRENCY_LIMIT,
    volumes={MODAL_VOLUME_PATH: volume},
)
class LocalColabFold:
    @staticmethod
    def three_to_one(three_letter_code):
        aa_dict = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        return aa_dict.get(three_letter_code, 'X')

    @enter()
    def setup(self):
        # Set up the environment when the container starts
        os.environ["PATH"] = "/localcolabfold/colabfold-conda/bin:" + os.environ["PATH"]

    @method()
    def colabfold_batch(self, input_path, out_dir, **kwargs):
        # Convert Path objects to strings
        input_path = str(input_path)
        out_dir = str(out_dir)

        cmd = ["colabfold_batch", input_path, out_dir]
        
        # Handle arguments
        for key, value in kwargs.items():
            key = key.replace('_', '-')
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            elif value is not None:
                cmd.extend([f"--{key}", str(value)])
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Command output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with error: {e}")
            logger.error(f"Error output: {e.stderr}")
        volume.commit()
        
    @staticmethod
    def extract_sequence_from_pdb(pdb_content):
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein", io.StringIO(pdb_content))
        
        sequences = {'A': '', 'B': ''}
        for model in structure:
            for chain in model:
                chain_id = chain.id
                if chain_id in sequences:
                    for residue in chain:
                        if PDB.is_aa(residue, standard=True):
                            sequences[chain_id] += LocalColabFold.three_to_one(residue.get_resname())
        
        return {
            'binder': sequences.get('A', ''),
            'target': sequences.get('B', ''),
        }

@app.cls(
    image=image,
    gpu="A10G",  # A10G should be sufficient for MSA generation
    volumes={MODAL_VOLUME_PATH: volume},
    timeout=9600,
)
class MSAQuery:

    @enter()
    def setup(self):
        os.environ["PATH"] = "/localcolabfold/colabfold-conda/bin:" + os.environ["PATH"]

    @method()
    def save_sequences_as_fasta(self, sequences: list[str]) -> Path:
        """Save sequences to a FASTA file, using sequence hashes as headers."""
        fasta_path = get_fasta_path(sequences)
        fasta_path = Path(MODAL_VOLUME_PATH) / fasta_path
        
        # Create parent directories if they don't exist
        fasta_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write sequences to FASTA
        with open(fasta_path, "w") as f:
            for seq in sequences:
                seq_hash = hash_seq(seq)
                f.write(f">{seq_hash}\n{seq}\n")
        
        logger.info(f"Saved {len(sequences)} sequences to {fasta_path}")
        return fasta_path

    @method()
    def query_msa_server(self, input_path: str | Path, out_dir: str | Path, host_url=MSA_QUERY_HOST_URL):
        """Query the MSA server to generate alignments."""
        # Convert paths to Path objects
        input_path = Path(input_path)
        out_dir = Path(out_dir)
        
        # Get expected output a3m path
        a3m_path = out_dir / f"{input_path.stem}.a3m"
        
        # Check if MSA already exists
        if a3m_path.exists():
            logger.info(f"Found existing MSA at {a3m_path}")
            return a3m_path
        
        # Create output directory if it doesn't exist
        out_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "colabfold_batch",
            str(input_path),
            str(out_dir),
            "--msa-only",
            "--host-url", host_url
        ]
        
        logger.info(f"Querying MSA server with command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"MSA generation output: {result.stdout}")

            a3m_path = out_dir / f"{input_path.stem}.a3m"
            volume.commit()
            return a3m_path
        except subprocess.CalledProcessError as e:
            logger.error(f"MSA generation failed with error: {e}")
            logger.error(f"Error output: {e.stderr}")
            raise

    @method()
    def run_msa_generation(self, sequences: List[str]) -> List[Path]:
        """Run complete MSA generation pipeline for multiple sequences.
        
        Args:
            sequences: List of sequences to process (each can be single sequence or paired like 'seq1:seq2')
            
        Returns:
            List of paths to generated MSA files
        """
        fasta_path = self.save_sequences_as_fasta.remote(sequences)
        volume.commit()
        volume.reload()
        out_dir = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["msa_results"]
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # query_msa_server returns a single a3m path, not a directory
        a3m_path = self.query_msa_server.remote(fasta_path, out_dir)
        volume.commit()
        return [a3m_path]  # Return as list to maintain interface


@app.function(
    image=image,
    timeout=9600,
    volumes={MODAL_VOLUME_PATH: volume},
)
def get_msa_for_binder(binder_seqs: List[str], target_seq: str=EGFR) -> List[Path]:
    """Get the MSA results for given binder sequences.
    
    Args:
        binder_seqs: List of binder sequences to process
        target_seq: Target sequence to pair with each binder
        
    Returns:
        List of paths to generated MSA files
    """
    paired_seqs = [f'{binder}:{target_seq}' for binder in binder_seqs]
    msa = MSAQuery()
    msa_paths = msa.run_msa_generation.remote(paired_seqs)
    volume.commit()
    return msa_paths



@app.function(
    image=image,
    timeout=9600,
    volumes={MODAL_VOLUME_PATH: volume},
)
def fold(input_path:str, num_models:int=1, num_recycle:int=1) -> Path:
    """Run folding on a directory containing MSA results."""
    # Ensure paths are within Modal volume
    input_path = Path(MODAL_VOLUME_PATH) / input_path
    output_dir = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["folded"]
    
    # Create output directory if it doesn't exist
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    
    colabfold = LocalColabFold()
    
    logger.info(f"Starting folding for MSA results in {input_path}")
    return colabfold.colabfold_batch.remote(input_path, output_dir, num_models=num_models, num_recycle=num_recycle)


@app.local_entrypoint()
def test_fold():
    """Test the folding functionality."""
    fold.remote('msa_results/5b353a/5b353a.a3m')


@app.local_entrypoint()
def test_msa_server_query():
    """Test the MSA server query functionality."""
    test_sequences = [
        f'{EGFS}:{EGFR}',
        # Add more test sequences as needed
    ]

    msa = MSAQuery()
    result_files = msa.run_msa_generation.remote(test_sequences)
    print(f"MSA results saved to: {[str(p) for p in result_files]}")

def get_a3m_path(binder_seq: str, target_seq: str=EGFR) -> Path:
    seq_hash = hash_seq(f'{binder_seq}:{target_seq}')
    return Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["msa_results"] / f"{seq_hash}.a3m"


def a3m_from_template(binder_seqs: List[str], parent_binder_seqs: List[str], target_seq: str=EGFR) -> List[Path]:
    """Get the a3m file paths for given binder sequences using their parent sequences as templates.
    
    Args:
        binder_seqs: List of binder sequences to process
        parent_binder_seqs: List of parent sequences to use as templates (must match length of binder_seqs)
        target_seq: Target sequence to pair with each binder
        
    Returns:
        List of paths to generated a3m files
    """
    assert len(binder_seqs) == len(parent_binder_seqs), "Must provide same number of binder and parent sequences"
    result_paths = []
    
    for binder_seq, parent_binder_seq in zip(binder_seqs, parent_binder_seqs):
        assert len(binder_seq) == len(parent_binder_seq), "Binder and parent sequences must be the same length"
        
        a3m_path = get_a3m_path(binder_seq=binder_seq, target_seq=target_seq)
        lineage_json_path = a3m_path.with_suffix('.lineage.json')
        
        if a3m_path.exists():
            logger.info(f"Found existing MSA at {a3m_path}")
            if not lineage_json_path.exists():
                # Record lineage information
                lineage_info = {
                    'seq_hash': hash_seq(f'{binder_seq}:{target_seq}'),
                    'parent_hash': hash_seq(f'{parent_binder_seq}:{target_seq}'),
                    'binder_seq': binder_seq,
                    'parent_seq': parent_binder_seq,
                    'target_seq': target_seq,
                    'mutations': get_mutation_diff(parent_binder_seq, binder_seq),
                    'timestamp': datetime.now().isoformat()
                }
                with open(lineage_json_path, 'w') as f:
                    json.dump(lineage_info, f, indent=2)
        else:
            template_a3m_path = get_a3m_path(binder_seq=parent_binder_seq, target_seq=target_seq)
            
            # Record lineage information
            lineage_info = {
                'seq_hash': hash_seq(f'{binder_seq}:{target_seq}'),
                'parent_hash': hash_seq(f'{parent_binder_seq}:{target_seq}'),
                'binder_seq': binder_seq,
                'parent_seq': parent_binder_seq,
                'mutations': get_mutation_diff(parent_binder_seq, binder_seq),
                'timestamp': datetime.now().isoformat()
            }
            with open(lineage_json_path, 'w') as f:
                json.dump(lineage_info, f, indent=2)
            
            a3m_path = swap_binder_seq_into_a3m(binder_seq=binder_seq, template_a3m_path=template_a3m_path, output_path=a3m_path)
        
        result_paths.append(a3m_path)
    
    return result_paths


@app.function(
    image=image,
    timeout=9600,
    volumes={MODAL_VOLUME_PATH: volume},
)
def fold_binder(binder_seqs: Union[str, List[str]], parent_binder_seqs: Union[str, List[str]]=None, target_seq: str=EGFR) -> List[Path]:
    """Fold one or more mutated binder sequences."""
    # Convert single sequences to lists for consistent handling
    if isinstance(binder_seqs, str):
        binder_seqs = [binder_seqs]
    
    # Handle parent sequences
    if parent_binder_seqs is None:
        # If no parent sequences provided, use binder sequences as their own parents
        parent_binder_seqs = binder_seqs
    elif isinstance(parent_binder_seqs, str):
        # If single parent sequence, replicate it for all binders
        parent_binder_seqs = [parent_binder_seqs] * len(binder_seqs)
    elif len(parent_binder_seqs) == 1:
        # If list with single parent, replicate it for all binders
        parent_binder_seqs = parent_binder_seqs * len(binder_seqs)
    
    logger.info(f"Folding sequences: {binder_seqs}")
    logger.info(f"Using parent sequences: {parent_binder_seqs}")
    
    # First, ensure we have MSAs for all sequences
    if binder_seqs == parent_binder_seqs:
        logger.info("Binder sequences match parent sequences, generating new MSAs")
        a3m_paths = get_msa_for_binder.remote(binder_seqs, target_seq)
    else:
        logger.info("Using template MSAs from parent sequences")
        a3m_paths = a3m_from_template(binder_seqs=binder_seqs, parent_binder_seqs=parent_binder_seqs, target_seq=target_seq)

    logger.info(f"Generated a3m paths: {a3m_paths}")
    volume.commit()
    logger.info("Committed volume")

    volume.reload()
    logger.info("Reloaded volume")
    
    # Check that all a3m files exist
    missing_files = [path for path in a3m_paths if not path.exists()]
    if missing_files:
        logger.error(f"MSA files not found after generation attempt: {missing_files}")
        raise FileNotFoundError(f"MSA generation failed for: {missing_files}")

    # Initialize ColabFold instance
    colabfold = LocalColabFold()
    output_dir = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["folded"]
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Fold all sequences
    folded_paths = []
    for a3m_path in a3m_paths:
        logger.info(f"Starting fold for {a3m_path}")
        try:
            # Get sequence hash from a3m path
            seq_hash = a3m_path.stem
            
            # Run folding
            colabfold.colabfold_batch.remote(a3m_path, output_dir, num_models=1, num_recycle=1)
            
            volume.commit()
            logger.info("Committed volume")
            
            volume.reload()
            logger.info("Reloaded volume")
            
            # Look for the scores files
            logger.info(f"Searching for scores files with pattern: {seq_hash}_scores*.json")
            scores_files = list(output_dir.glob(f"{seq_hash}_scores*.json"))
            
            if not scores_files:
                logger.error(f"No scores files found for {seq_hash} in {output_dir}")
                # List all files in output directory for debugging
                raise FileNotFoundError(f"No scores files found for {seq_hash}")
            
            # Extract metrics from scores files
            metrics = []
            for scores_file in scores_files:
                with open(scores_file, 'r') as f:
                    data = json.load(f)
                    
                rank_match = re.search(r'rank_(\d+)', scores_file.name)
                model_number = int(rank_match.group(1)) if rank_match else 0
                
                metrics.append({
                    'seq_hash': seq_hash,
                    'model_number': model_number,
                    'plddt': np.mean(data.get('plddt', [])),
                    'ptm': data.get('ptm', 0),
                    'i_ptm': data.get('iptm', 0),
                    'pae': np.mean(data.get('pae', [])) if data.get('pae') else None
                })
            
            logger.info(f"Extracted metrics from {len(metrics)} models for {seq_hash}")
            return metrics

        except Exception as e:
            logger.error(f"Folding failed for {a3m_path}: {str(e)}")
            raise
    
    return folded_paths


def calc_percentage_charged(sequence: str) -> float:
    """Calculate the percentage of charged amino acids in a sequence.
    
    Args:
        sequence: Amino acid sequence string
        
    Returns:
        float: Fraction of sequence that is charged (R,K,H,D,E)
    """
    charged = set('RKHDE')
    return sum(1 for aa in sequence if aa in charged) / len(sequence)

def calc_percentage_hydrophobic(sequence: str) -> float:
    """Calculate the percentage of hydrophobic amino acids in a sequence.
    
    Args:
        sequence: Amino acid sequence string
        
    Returns:
        float: Fraction of sequence that is hydrophobic (V,I,L,M,F,Y,W)
    """
    hydrophobic = set('VILMFYW')
    return sum(1 for aa in sequence if aa in hydrophobic) / len(sequence)



def get_metrics_from_hash(seq_hash: str) -> dict:
    """Extract metrics from folding results for a given sequence hash."""
    folded_dir = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["folded"]
    
    # Add check for lineage information
    msa_dir = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["msa_results"]
    lineage_path = msa_dir / f"{seq_hash}.lineage.json"
    
    # Load lineage info if available
    parent_info = {}
    if lineage_path.exists():
        with open(lineage_path, 'r') as f:
            lineage_data = json.load(f)
            parent_info = {
                'parent_hash': lineage_data.get('parent_hash'),
                'parent_sequence': lineage_data.get('parent_seq'),
                'mutations': lineage_data.get('mutations')
            }
    
    # Find relevant files
    base_pattern = f"{seq_hash}_*"
    scores_files = list(folded_dir.glob(f"{base_pattern}scores*.json"))
    pdb_files = list(folded_dir.glob(f"{base_pattern}unrelaxed*.pdb"))
    
    if not scores_files or not pdb_files:
        logger.warning(f"Missing results files for hash {seq_hash}")
        return None
        
    results = []
    for pdb_path in pdb_files:
        # Find corresponding scores file
        scores_path = pdb_path.with_name(pdb_path.name.replace('_unrelaxed_', '_scores_')).with_suffix('.json')
        
        if not scores_path.exists():
            continue
            
        with open(pdb_path, 'r') as pdb_file:
            pdb_content = pdb_file.read()
            sequences = LocalColabFold.extract_sequence_from_pdb(pdb_content)
            
        with open(scores_path, 'r') as json_file:
            data = json.load(json_file)
            
            plddt_array = np.array(data.get('plddt', []))
            pae_array = np.array(data.get('pae', []))
            
            rank_match = re.search(r'rank_(\d+)', pdb_path.name)
            model_number = int(rank_match.group(1)) if rank_match else 0
            
            binder_seq = sequences.get('binder', '')
            binder_length = len(binder_seq)
            target_length = len(sequences.get('target', ''))
            
            # Calculate metrics
            binder_plddt = float(plddt_array[:binder_length].mean()) if plddt_array.size else 0
            binder_pae = float(pae_array[:binder_length, :binder_length].mean()) if pae_array.size else 0
            
            pae_interaction = 0
            if pae_array.size:
                if pae_array.shape[0] >= binder_length and pae_array.shape[1] >= binder_length:
                    pae_interaction = float((
                        pae_array[binder_length:, :binder_length].mean() + 
                        pae_array[:binder_length, binder_length:].mean()
                    ) / 2)
            
            
            # Add sequence property metrics for the binder
            binder_charged_fraction = calc_percentage_charged(binder_seq)
            binder_hydrophobic_fraction = calc_percentage_hydrophobic(binder_seq)
            
            # Calculate sequence indices for binder
            sequence_indices = calculate_sequence_indices(binder_seq)
            
            results.append({
                'seq_hash': seq_hash,
                'binder_sequence': binder_seq,
                'binder_length': binder_length,
                'target_sequence': sequences.get('target', ''),
                'target_length': target_length,
                'model_number': model_number,
                'binder_plddt': binder_plddt,
                'binder_pae': binder_pae,
                'pae_interaction': pae_interaction,
                'ptm': data.get('ptm', 0),
                'i_ptm': data.get('iptm', 0),
                'binder_charged_fraction': binder_charged_fraction,
                'binder_hydrophobic_fraction': binder_hydrophobic_fraction,
                'parent_hash': parent_info.get('parent_hash', None),
                'parent_sequence': parent_info.get('parent_sequence', None),
                'mutations': parent_info.get('mutations', None),
                'binder_hydrophobicity': sequence_indices['avg_hydrophobicity'],
                'binder_hydropathy': sequence_indices['avg_hydropathy'],
                'binder_solubility': sequence_indices['avg_solubility'],
                'p_soluble': calculate_solubility(binder_seq)
            })
    
    return results

@app.function(
    image=image,
    timeout=9600,
    volumes={MODAL_VOLUME_PATH: volume},
)
def update_metrics_for_all_folded(overwrite: bool = False):
    """Update metrics CSV with results from all folded structures.
    
    Args:
        overwrite: If True, recalculate all metrics. If False, only calculate for missing entries
    """
    metrics_csv_path = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["metrics_csv"]
    folded_dir = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["folded"]
    
    # Read existing metrics if file exists and not overwriting
    existing_metrics = {}
    if metrics_csv_path.exists() and not overwrite:
        df = pd.read_csv(metrics_csv_path)
        existing_metrics = df.to_dict('records')  # Convert to list of dicts properly
    
    # Collect all metrics
    all_metrics = []
    pdb_files = list(folded_dir.glob("*.pdb"))
    
    for i, pdb_path in enumerate(pdb_files):
        seq_hash = pdb_path.stem.split('_')[0]  # Extract hash from filename
        
        # Skip if already processed and not overwriting
        if not overwrite:
            existing_metric = next((m for m in existing_metrics if m['seq_hash'] == seq_hash), None)
            if existing_metric:
                all_metrics.append(existing_metric)
                continue

        try:
            metrics = get_metrics_from_hash(seq_hash)
            if metrics:  # Only add if we got valid metrics
                all_metrics.extend(metrics)
        except Exception as e:
            logger.error(f"Error processing {pdb_path}: {e}")
            
        # Log progress every 10 files
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(pdb_files)} PDB files")
    
    # Convert to DataFrame
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        
        # Ensure all required columns exist
        check_columns = ['binder_hydrophobicity', 'binder_hydropathy', 'binder_solubility', 'p_soluble']
        for col in check_columns:
            if col not in df.columns:
                df[col] = pd.NA
        
        # Check for missing values
        missing_indices = df[check_columns].isna().any(axis=1)
        
        if missing_indices.any():
            logger.info(f"Found {missing_indices.sum()} rows with missing indices or p_soluble")
            
            # Calculate missing indices and p_soluble
            for idx in df[missing_indices].index:
                if 'binder_sequence' in df.columns and pd.notna(df.loc[idx, 'binder_sequence']):
                    sequence = df.loc[idx, 'binder_sequence']
                    indices = calculate_sequence_indices(sequence)
                    
                    df.loc[idx, 'binder_hydrophobicity'] = indices['avg_hydrophobicity']
                    df.loc[idx, 'binder_hydropathy'] = indices['avg_hydropathy']
                    df.loc[idx, 'binder_solubility'] = indices['avg_solubility']
                    df.loc[idx, 'p_soluble'] = calculate_solubility(sequence)
            
            logger.info("Updated missing indices and p_soluble values")
        
        # Save updated DataFrame
        df.to_csv(metrics_csv_path, index=False)
        logger.info(f"Updated metrics saved to {metrics_csv_path}")
    else:
        logger.warning("No metrics were collected")
    return df

@app.local_entrypoint()
def update_metrics():
    update_metrics_for_all_folded.remote(overwrite=True)

@app.local_entrypoint()
def benchmark_fold():
    import time
    logger.info("Starting benchmark fold")
    start_time = time.time()

    parent_binder_seq = 'AERMRRRFEHIVEIHEEWAKEVLENLKKQGSKEEDLKFMEEYLEQDVEELRKRAEEMVEEYEKSS'
    binder_seqs = [
        # 'AERMRRRFEHIVEIHEEWAKEVLENLKKQGSKEEDLKFMEEYLEQDVEELRKRAEEMVEEYEKSS',
        # 'AERMRRRFEHIVEIHEEWAKEELENLKKQGSKEEDLKFMEEYLEQDVEELRKRAEEMVEEYEKSS',
        'AERMRRRFEHIVEIHEEWAKEVEENLKKQGSKEEDLKFMEEYLEQDVEELRKRAEEMVEEYEKSS'
        ]

    logger.info("Calling fold_binder")
    result = fold_binder.remote(binder_seqs=binder_seqs, parent_binder_seqs=[parent_binder_seq])
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Benchmark fold completed in {duration:.2f} seconds")

    return result


@app.function(
    image=image,
    timeout=9600,
    volumes={MODAL_VOLUME_PATH: volume},
)
def parallel_fold_binder(binder_seqs: Union[str, List[str]], parent_binder_seqs: Union[str, List[str]]=None, target_seq: str=EGFR) -> List[Path]:
    """Fold multiple binder sequences in parallel."""
    # Convert single sequences to lists for consistent handling
    if isinstance(binder_seqs, str):
        binder_seqs = [binder_seqs]
    if isinstance(parent_binder_seqs, str):
        parent_binder_seqs = [parent_binder_seqs] * len(binder_seqs)
    elif parent_binder_seqs is not None and len(parent_binder_seqs) == 1:
        parent_binder_seqs = parent_binder_seqs * len(binder_seqs)
    
    logger.info(f"Parallel folding {len(binder_seqs)} sequences")
    

    if binder_seqs == parent_binder_seqs:
        logger.info("Binder sequences match parent sequences, generating new MSAs")
        a3m_paths = get_msa_for_binder.remote(binder_seqs, target_seq)
    else:
        logger.info("Using template MSAs from parent sequences")
        a3m_paths = a3m_from_template(binder_seqs=binder_seqs, parent_binder_seqs=parent_binder_seqs, target_seq=target_seq)

    volume.commit()
    volume.reload()
    
    # Check that all a3m files exist
    missing_files = [path for path in a3m_paths if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"MSA generation failed for: {missing_files}")

    # Initialize ColabFold instance
    colabfold = LocalColabFold()
    output_dir = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["folded"]
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Create list of arguments for each folding task
    # Each item should be a tuple of (a3m_path, output_dir)
    folding_args = [(a3m_path, output_dir) for a3m_path in a3m_paths]
    kwargs = {"num_models": 1, "num_recycle": 1}

    # Run folding in parallel using starmap since we have multiple arguments
    all_metrics = []
    for result in colabfold.colabfold_batch.starmap(folding_args, kwargs=kwargs):
        logger.info(f"Completed folding task with result: {result}")
        
        # Process results for this batch
        for a3m_path in a3m_paths:
            seq_hash = a3m_path.stem
            scores_files = list(output_dir.glob(f"{seq_hash}_scores*.json"))
            
            if not scores_files:
                logger.error(f"No scores files found for {seq_hash}")
                continue
                
            metrics = []
            for scores_file in scores_files:
                with open(scores_file, 'r') as f:
                    data = json.load(f)
                    
                rank_match = re.search(r'rank_(\d+)', scores_file.name)
                model_number = int(rank_match.group(1)) if rank_match else 0
                
                metrics.append({
                    'seq_hash': seq_hash,
                    'model_number': model_number,
                    'plddt': np.mean(data.get('plddt', [])),
                    'ptm': data.get('ptm', 0),
                    'i_ptm': data.get('iptm', 0),
                    'pae': np.mean(data.get('pae', [])) if data.get('pae') else None
                })
            
            all_metrics.extend(metrics)
    
    return all_metrics

def generate_mutated_sequences(parent_seq: str, num_sequences: int = 10) -> list:
    import random
    """Generate sequences with single random mutations from parent sequence.
    
    Args:
        parent_seq: Parent sequence to mutate
        num_sequences: Number of sequences to generate
        
    Returns:
        List of mutated sequences
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'  # Standard amino acids
    sequences = []
    
    for _ in range(num_sequences):
        # Choose random position to mutate
        pos = random.randint(0, len(parent_seq) - 1)
        
        # Choose random amino acid different from original
        original_aa = parent_seq[pos]
        new_aa = random.choice(amino_acids.replace(original_aa, ''))
        
        # Create mutated sequence
        mutated_seq = parent_seq[:pos] + new_aa + parent_seq[pos + 1:]
        sequences.append(mutated_seq)
    
    return sequences

@app.local_entrypoint()
def benchmark_parallel_fold():
    import time
    logger.info("Starting parallel benchmark fold")
    start_time = time.time()

    parent_binder_seq = 'AERMRRRFEHIVEIHEEWAKEVLENLKKQGSKEEDLKFMEEYLEQDVEELRKRAEEMVEEYEKSS'
    binder_seqs = generate_mutated_sequences(parent_binder_seq)

    logger.info("Calling parallel_fold_binder")
    result = parallel_fold_binder.remote(binder_seqs=binder_seqs, parent_binder_seqs=[parent_binder_seq])
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Parallel benchmark fold completed in {duration:.2f} seconds")

    return result

@app.function(
    image=image,
    timeout=9600,
    volumes={MODAL_VOLUME_PATH: volume},
)
def fold_high_quality(input_path: str, num_recycle: int=3, num_seeds: int=3, num_models: int=5, templates: bool=True) -> Path:
    """Run high quality folding with more recycling, seeds, and models.
    
    Args:
        input_path: Path to input MSA or sequence file
        target_seq: Target sequence to pair with binder (defaults to official EGFR sequence)
    """
    # Ensure paths are within Modal volume
    input_path = Path(MODAL_VOLUME_PATH) / input_path
    output_dir = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["folded_high_quality"]
    
    # Create output directory if it doesn't exist
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    
    colabfold = LocalColabFold()
    
    logger.info(f"Starting high quality folding for {input_path}")
    return colabfold.colabfold_batch.remote(
        input_path, 
        output_dir,
        num_recycle=num_recycle,
        num_seeds=num_seeds,
        num_models=num_models,
        templates=templates
    )

@app.function(
    image=image,
    timeout=9600,
    volumes={MODAL_VOLUME_PATH: volume},
)
def fold_binder_high_quality(binder_seq: str, target_seq: str=OFFICIAL_EGFR) -> dict:
    """Run complete high quality folding pipeline for a single binder sequence.
    
    Args:
        binder_seq: Binder sequence to fold
        target_seq: Target sequence (defaults to official EGFR sequence)
    
    Returns:
        Dictionary containing folding metrics
    """
    logger.info(f"Starting high quality folding pipeline for binder sequence")
    
    # First ensure we have MSA
    msa_paths = get_msa_for_binder.remote([binder_seq], target_seq)
    if not msa_paths:
        raise ValueError("MSA generation failed")
    
    # Run high quality folding
    fold_high_quality.remote(msa_paths[0])
    
    # Extract metrics
    seq_hash = hash_seq(f"{binder_seq}:{target_seq}")
    metrics = get_metrics_from_hash(seq_hash)
    
    return metrics

@app.local_entrypoint()
def test_high_quality_fold():
    """Test the high quality folding functionality."""
    # test_seq = EGFS  # Using the example binder sequence
    test_seq = 'SLFSICPYRYHGICKNNGVCRYAINLRSYTCQCVSGYTGARCQEADIRYLLLRI'
    result = fold_binder_high_quality.remote(test_seq)
    print(f"High quality folding results: {result}")

# Add these constants near the top of the file with other constants
HYDROPHOBICITY_INDICES = {
    'A': {'hydrophobicity': 0.230188679, 'hydropathy': 0.700000000, 'solubility': 0.325669858},
    'R': {'hydrophobicity': 0.226415094, 'hydropathy': 0.000000000, 'solubility': 0.463603847},
    'N': {'hydrophobicity': 0.022641509, 'hydropathy': 0.111111111, 'solubility': 0.274060271},
    'D': {'hydrophobicity': 0.173584906, 'hydropathy': 0.111111111, 'solubility': 0.170907494},
    'C': {'hydrophobicity': 0.403773585, 'hydropathy': 0.777777778, 'solubility': 1.000000000},
    'Q': {'hydrophobicity': 0.000000000, 'hydropathy': 0.111111111, 'solubility': 0.424648204},
    'E': {'hydrophobicity': 0.177358491, 'hydropathy': 0.111111111, 'solubility': 0.000000000},
    'G': {'hydrophobicity': 0.026415094, 'hydropathy': 0.455555556, 'solubility': 0.402625886},
    'H': {'hydrophobicity': 0.230188679, 'hydropathy': 0.144444444, 'solubility': 0.198993339},
    'I': {'hydrophobicity': 0.837735849, 'hydropathy': 1.000000000, 'solubility': 0.662440832},
    'L': {'hydrophobicity': 0.577358491, 'hydropathy': 0.922222222, 'solubility': 0.711681552},
    'K': {'hydrophobicity': 0.433962264, 'hydropathy': 0.066666667, 'solubility': 0.130628199},
    'M': {'hydrophobicity': 0.445283019, 'hydropathy': 0.711111111, 'solubility': 0.766855148},
    'F': {'hydrophobicity': 0.762264151, 'hydropathy': 0.811111111, 'solubility': 0.862558633},
    'P': {'hydrophobicity': 0.735849057, 'hydropathy': 0.322222222, 'solubility': 0.351616012},
    'S': {'hydrophobicity': 0.018867925, 'hydropathy': 0.411111111, 'solubility': 0.521767440},
    'T': {'hydrophobicity': 0.018867925, 'hydropathy': 0.422222222, 'solubility': 0.381261111},
    'W': {'hydrophobicity': 1.000000000, 'hydropathy': 0.400000000, 'solubility': 0.750136006},
    'Y': {'hydrophobicity': 0.709433962, 'hydropathy': 0.355555556, 'solubility': 0.806226306},
    'V': {'hydrophobicity': 0.498113208, 'hydropathy': 0.966666667, 'solubility': 0.539559639}
}

def calculate_sequence_indices(sequence: str) -> dict:
    """Calculate various hydrophobicity indices for a sequence.
    
    Args:
        sequence: Amino acid sequence
        
    Returns:
        Dictionary containing average indices for the sequence
    """
    if not sequence:
        return {
            'avg_hydrophobicity': 0.0,
            'avg_hydropathy': 0.0,
            'avg_solubility': 0.0
        }
    
    total_hydrophobicity = 0.0
    total_hydropathy = 0.0
    total_solubility = 0.0
    valid_residues = 0
    
    for aa in sequence:
        if aa in HYDROPHOBICITY_INDICES:
            indices = HYDROPHOBICITY_INDICES[aa]
            total_hydrophobicity += indices['hydrophobicity']
            total_hydropathy += indices['hydropathy']
            total_solubility += indices['solubility']
            valid_residues += 1
    
    if valid_residues == 0:
        return {
            'avg_hydrophobicity': 0.0,
            'avg_hydropathy': 0.0,
            'avg_solubility': 0.0
        }
    
    return {
        'avg_hydrophobicity': total_hydrophobicity / valid_residues,
        'avg_hydropathy': total_hydropathy / valid_residues,
        'avg_solubility': total_solubility / valid_residues
    }

@app.function(
    image=image,
    timeout=9600,
    volumes={MODAL_VOLUME_PATH: volume},
)
def update_high_quality_metrics(overwrite: bool = False):
    """Update metrics CSV with results from all high-quality folded structures."""
    metrics_csv_path = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["high_quality_metrics_csv"]
    top_ranked_csv_path = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["high_quality_top_ranked_csv"]
    folded_dir = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["folded_high_quality"]
    
    # Read existing metrics if file exists and not overwriting
    existing_metrics = {}
    if metrics_csv_path.exists() and not overwrite:
        df = pd.read_csv(metrics_csv_path)
        existing_metrics = df.to_dict('records')
    
    # Collect all metrics
    all_metrics = []
    score_files = list(folded_dir.glob("*_scores_rank_*.json"))
    
    for i, score_file in enumerate(score_files):
        try:
            # Extract parts from filename
            # Format: hash_scores_rank_001_alphafold2_multimer_v3_model_5_seed_002.json
            parts = score_file.stem.split('_')
            seq_hash = parts[0]
            rank = int(parts[3])
            model_num = parts[8]  # Corrected index for model number
            seed_num = parts[10]  # Corrected index for seed number
            
            # Skip if already processed and not overwriting
            if not overwrite:
                existing_metric = next((m for m in existing_metrics 
                                      if m['seq_hash'] == seq_hash and m['rank'] == rank), None)
                if existing_metric:
                    all_metrics.append(existing_metric)
                    continue

            # Get corresponding PDB file
            pdb_file = folded_dir / f"{seq_hash}_unrelaxed_rank_{rank:03d}_alphafold2_multimer_v3_model_{model_num}_seed_{seed_num}.pdb"
            
            if not pdb_file.exists():
                logger.warning(f"Missing PDB file for {seq_hash} rank {rank}")
                continue
                
            # Extract sequences from PDB
            with open(pdb_file, 'r') as f:
                pdb_content = f.read()
                sequences = LocalColabFold.extract_sequence_from_pdb(pdb_content)
            
            binder_seq = sequences.get('binder', '')
            target_seq = sequences.get('target', '')
            binder_length = len(binder_seq)
            
            # Read scores
            with open(score_file, 'r') as f:
                scores_data = json.load(f)
                
            # Get PAE data
            pae_file = folded_dir / f"{seq_hash}_predicted_aligned_error_v1.json"
            pae_matrix = None
            if pae_file.exists():
                with open(pae_file, 'r') as f:
                    pae_data = json.load(f)
                    pae_matrix = np.array(pae_data.get('predicted_aligned_error', []))
            
            # Calculate metrics
            metric = {
                'seq_hash': seq_hash,
                'rank': rank,
                'model': model_num,  # Now using correct model number
                'seed': seed_num,    # Now using correct seed number
                'binder_sequence': binder_seq,
                'target_sequence': target_seq,
                'binder_length': binder_length,
                'target_length': len(target_seq),
                'plddt': np.mean(scores_data.get('plddt', [])),
                'ptm': scores_data.get('ptm', 0),
                'i_ptm': scores_data.get('iptm', 0),
                'high_quality': True
            }
            
            # Add sequence property metrics
            sequence_indices = calculate_sequence_indices(binder_seq)
            metric.update({
                'binder_hydrophobicity': sequence_indices['avg_hydrophobicity'],
                'binder_hydropathy': sequence_indices['avg_hydropathy'],
                'binder_solubility': sequence_indices['avg_solubility']
            })
            
            # Add PAE metrics if available
            if pae_matrix is not None and pae_matrix.size > 0:
                binder_pae = pae_matrix[:binder_length, :binder_length].mean()
                target_pae = pae_matrix[binder_length:, binder_length:].mean()
                interface_pae = (
                    pae_matrix[:binder_length, binder_length:].mean() +
                    pae_matrix[binder_length:, :binder_length].mean()
                ) / 2
                
                metric.update({
                    'binder_pae': binder_pae,
                    'target_pae': target_pae,
                    'pae_interaction': interface_pae
                })
            
            all_metrics.append(metric)
                
        except Exception as e:
            logger.error(f"Error processing {score_file}: {str(e)}")
            
        # Log progress every 10 files
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(score_files)} structures")
    
    # Convert to DataFrame and save all metrics
    if all_metrics:
        df_all = pd.DataFrame(all_metrics)
        df_all.to_csv(metrics_csv_path, index=False)
        logger.info(f"Updated high-quality metrics saved to {metrics_csv_path}")
        
        # Create top-ranked only DataFrame
        df_top = df_all[df_all['rank'] == 1].copy()
        df_top.to_csv(top_ranked_csv_path, index=False)
        logger.info(f"Top-ranked metrics saved to {top_ranked_csv_path}")
        
        return df_all, df_top
    else:
        logger.warning("No metrics were collected")
        return None, None
    


@app.function(
    image=image,
    timeout=9600,
    volumes={MODAL_VOLUME_PATH: volume},
)
def parallel_fold_binder_high_quality(binder_seqs: List[str], target_seq: str=OFFICIAL_EGFR) -> List[dict]:
    """Run complete high quality folding pipeline for multiple binder sequences in parallel.
    
    Args:
        binder_seqs: List of binder sequences to fold
        target_seq: Target sequence (defaults to official EGFR sequence)
    
    Returns:
        List of dictionaries containing folding metrics for each sequence
    """
    logger.info(f"Starting parallel high quality folding pipeline for {len(binder_seqs)} sequences")
    
    # Use map to run fold_binder_high_quality in parallel for each sequence
    results = list(fold_binder_high_quality.map(binder_seqs, [target_seq] * len(binder_seqs)))
    
    return results

@app.local_entrypoint()
def test_parallel_high_quality_fold():
    """Test the parallel high quality folding functionality."""
    test_seqs = [
        'SYDGKCLNNGKCRYIEDLDSYTCQCESGYTGDRCQTRDLRWLELH',
        'SYEGYCENRGTCQHIESLDSYTCKCLKGYTGDRCQSQDLRYLYLE',
        'KYDGYCNNHGECQHIHSLDSYTCKCLPGYEGDRCQTQDLRWLELR',
        'SYDGYCNNRGVCRHIESLDSYTCKCDQGYEGDRCQTRDLRWLELH',
        'SYNGYCKNGGQCQHIISLDQYTCRCESGYEGDRCQTRDLRWLELR',
        'SYDGYCLNRGECQHIHSLDSYTCKCEPGYTGDRCQTQDLRWLELR',
        'SYDGYCNNRGVCRHIESLDTYTCQCKQGYEGDRCETRDLRWLELY',
        'TYDGYCLNGGKCEHVESLDKYTCNCVSGYTGDRCQERDLRWLEHQ'
    ]
    test_seqs = [
        'GYKGYCLNQGKCEHVESLDSYTCNCVSGYTGDRCQERDLRWLELR',
        'GYKGYCLNQGKCEHVESLDSYTCKCVSGYTGDRCQERDLRWLELR',
        'GYKGYCLNEGKCEHVESLDSYTCKCVSGYTGDRCQERDLRWLELR',
        'GYKGYCLNEGKCEHVESLDSYTCKCVSGYTGDRCQERDLRWLEL',
        'AYKGYCLNEGKCEHVESLDSYTCKCVSGYTGDRCQERDLRWLEL',
        'AYKGYCLNEGKCEHVESLDSYTCKCVSGYTGDRCQERDLRWLELL',
        'SYDGYCLNRGECQHIHSLDSYTCKCEPGYTGDRCQTQDLRWLELR',
        'SYDGYCLNEGECQHIHSLDSYTCKCEPGYTGDRCQTQDLRWLELR',
        'SYDGYCLNEGECQHIHSLDSYTCKCEPGYTGDRCQTQDLRWLELL',
        'SYDGYCLNEGECRHIHSLDSYTCKCEPGYTGDRCQTQDLRWLELL',
        'SYEGYCLNEGECRHIHSLDSYTCKCEPGYTGDRCQTQDLRWLELL',
        'SYEGYCLNEGECRHIHSLDSYTCKCEPGYTGDRCQTRDLRWLELL',
        'SYEGYCLNEGECRHIHSLDSYTCKCEAGYTGDRCQTRDLRWLELL',
        'SYEGYCLNEGECRHVHSLDSYTCKCEAGYTGDRCQTRDLRWLELL',
        'SYEGYCLNEGECRHVKSLDSYTCKCEAGYTGDRCQTRDLRWLELL',
        'SYEGYCLNGGECRHVKSLDSYTCKCEAGYTGDRCQTRDLRWLELL',
        'SYEGYCLNGGECRHVKSLDTYTCKCEAGYTGDRCQTRDLRWLELL',
        'SYEGYCLNGGECRHVKSLDTYTCKCEAGYTGDRCQTRDLRYLELL',
        'NLFSRCPKRYHGICENNGQCRYAINLRTYTCICDSGYTGDRCQELDIRYLLLLN',
        'NLFSRCPKRYAGICENNGQCRYAINLRTYTCICDSGYTGDRCQELDIRYLLLLN',
        'NLFSRCPKRYAGICENNGQCRYAINLRTYTCICKSGYTGDRCQELDIRYLLLLN',
        'NLFSRCPKRYAGICENNGKCRYAINLRTYTCICKSGYTGDRCQELDIRYLLLLN',
    ]
    result = parallel_fold_binder_high_quality.remote(test_seqs)
    print(f"Parallel high quality folding results: {result}")

@app.local_entrypoint()
def get_binder_msa():
    """Get MSA for a binder sequence."""
    binder_seqs = [
        # 'SYEGYCENGGTCVHVEALDSYTCKCLKGYTGDRCQSQDLRYLLLE'
        'RLFSRCPRRYHGICKNNGQCRYAINLRTYTCRCLSGYTGDRCEELDIRYLLLLY',
        'RLFSRCPRRYHGICKNNGQCKYAINLRTYTCRCLSGYTGDRCEELDIRYLLLLY',
    ]
    result = get_msa_for_binder.remote(binder_seqs)
    print(f"MSA for binder sequence: {result}")


@app.local_entrypoint()
def fhq(s: str):
    """Fold a high quality binder sequence."""
    result = fold_binder_high_quality.remote(s)
    print(f"Folded high quality sequence: {result}")

@app.local_entrypoint()
def msa_for_binder(s:str):
    """Get MSA for a binder sequence."""
    result = get_msa_for_binder.remote([s])
    print(f"MSA for binder sequence: {result}")