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
    gpu="H100",
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
            raise
        
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

    def extract_metrics(self, structure_path: Path) -> dict:
        """Extract folding metrics from the structure file."""
        # Implement actual metric extraction logic here
        # Placeholder implementation:
        return {
            "pLDDT": 85.0,
            "i_PAE": 10.0,
            "i_pTM": 0.8
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
        out_dir = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["msa_results"]
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # query_msa_server returns a single a3m path, not a directory
        a3m_path = self.query_msa_server.remote(fasta_path, out_dir)
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
    return msa.run_msa_generation.remote(paired_seqs)



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

def get_a3m_path(binder_seq: str, target_seq: str) -> Path:
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
    if isinstance(parent_binder_seqs, str):
        parent_binder_seqs = [parent_binder_seqs] * len(binder_seqs)
    elif parent_binder_seqs is not None and len(parent_binder_seqs) == 1:
        parent_binder_seqs = parent_binder_seqs * len(binder_seqs)
    
    logger.info(f"Folding sequences: {binder_seqs}")
    logger.info(f"Using parent sequences: {parent_binder_seqs}")
    
    # First, ensure we have MSAs for all sequences
    if parent_binder_seqs is None:
        logger.info("No parent sequences provided, generating new MSAs")
        a3m_paths = get_msa_for_binder.remote(binder_seqs, target_seq)
    else:
        # If parent sequences match binder sequences exactly, generate new MSAs
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
                all_files = list(output_dir.glob("*"))
                logger.info(f"All files in output directory: {[f.name for f in all_files]}")
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
            
            # Add epitope interaction analysis
            parser = PDB.PDBParser(QUIET=True)
            structure = parser.get_structure("protein", pdb_path)
            
            # Add sequence property metrics for the binder
            binder_charged_fraction = calc_percentage_charged(binder_seq)
            binder_hydrophobic_fraction = calc_percentage_hydrophobic(binder_seq)
            
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
                'mutations': parent_info.get('mutations', None)
            })
    
    return results

@app.function(
    image=image,
    timeout=9600,
    volumes={MODAL_VOLUME_PATH: volume},
)
def update_metrics_for_all_folded():
    """Update metrics CSV with results from all folded structures, only processing new hashes."""
    folded_dir = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["folded"]
    metrics_file = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["metrics_csv"]
    
    # Create metrics directory if it doesn't exist
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing metrics if available
    existing_df = pd.DataFrame()
    existing_hashes = set()
    if metrics_file.exists():
        existing_df = pd.read_csv(metrics_file)
        existing_hashes = set(existing_df['seq_hash'].unique())
        logger.info(f"Found existing metrics for {len(existing_hashes)} sequence hashes")
    
    # Get all unique sequence hashes from folded files
    all_files = list(folded_dir.glob("*_scores*.json"))
    all_hashes = set(f.name.split('_')[0] for f in all_files)
    
    # Identify new hashes that need processing
    new_hashes = all_hashes - existing_hashes
    logger.info(f"Found {len(new_hashes)} new sequence hashes to process")
    
    if not new_hashes:
        logger.info("No new sequences to process")
        return existing_df  # Return existing DataFrame instead of file path
    
    # Process new hashes
    new_results = []
    for seq_hash in new_hashes:
        results = get_metrics_from_hash(seq_hash)
        if results:
            new_results.extend(results)
    
    if new_results:
        # Convert new results to DataFrame
        new_df = pd.DataFrame(new_results)
        
        # Combine with existing results
        if not existing_df.empty:
            df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            df = new_df
        
        # Save updated metrics
        df.to_csv(metrics_file, index=False)
        logger.info(f"Added metrics for {len(new_results)} new models, total {len(df)} entries")
    else:
        logger.info("No new valid results to add")
        df = existing_df  # Use existing DataFrame if no new results
    
    return df


# @app.function(
#     image=image,
#     timeout=9600,
#     volumes={MODAL_VOLUME_PATH: volume},
# )
# def rebuild_substituted_a3ms() -> List[Path]:
#     """Rebuild a3m files that were created by substitution, using the same parent templates
#     but redoing the substitution process.
    
#     Returns:
#         List[Path]: Paths to rebuilt a3m files
#     """
#     msa_dir = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["msa_results"]
#     rebuilt_paths = []
    
#     # Find all lineage files
#     lineage_files = list(msa_dir.glob("*.lineage.json"))
#     logger.info(f"Found {len(lineage_files)} sequences with lineage information")
    
#     for lineage_file in lineage_files:
#         try:
#             with open(lineage_file, 'r') as f:
#                 lineage_data = json.load(f)
                
#             binder_seq = lineage_data.get('binder_seq')
#             parent_seq = lineage_data.get('parent_seq')
            
#             if not (binder_seq and parent_seq):
#                 logger.warning(f"Missing sequence information in {lineage_file}")
#                 continue
            
#             # Get paths
#             parent_hash = hash_seq(f"{parent_seq}:{EGFR}")
#             template_a3m_path = msa_dir / f"{parent_hash}.a3m"
#             output_a3m_path = msa_dir / f"{lineage_file.stem.replace('.lineage', '')}.a3m"
            
#             if not template_a3m_path.exists():
#                 logger.warning(f"Parent template not found: {template_a3m_path}")
#                 continue
                
#             # Redo the substitution
#             logger.info(f"Rebuilding {output_a3m_path}")
#             rebuilt_path = swap_binder_seq_into_a3m(
#                 binder_seq=binder_seq,
#                 template_a3m_path=template_a3m_path,
#                 output_path=output_a3m_path
#             )
#             rebuilt_paths.append(rebuilt_path)
            
#         except Exception as e:
#             logger.error(f"Failed to process {lineage_file}: {str(e)}")
#             continue
    
#     logger.info(f"Successfully rebuilt {len(rebuilt_paths)} a3m files")
#     return rebuilt_paths

# @app.local_entrypoint()
# def do_rebuild():
#     rebuild_substituted_a3ms.remote()

@app.local_entrypoint()
def benchmark_fold():
    import time
    logger.info("Starting benchmark fold")
    start_time = time.time()

    parent_binder_seq = 'PSFSACPSNYDGYCMNGGVCHYFESLTSITCQCIIGYIGDRCQTFDLRYTELRR'
    binder_seqs = ['PSFSACPSNYDGYCMNGGVCHYFESLTSITCQCIIGYIGDRCQTDDLRYTELRR']

    logger.info("Calling fold_binder")
    result = fold_binder.remote(binder_seqs=binder_seqs, parent_binder_seqs=[parent_binder_seq])
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Benchmark fold completed in {duration:.2f} seconds")

    return result