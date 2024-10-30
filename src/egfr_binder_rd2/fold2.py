import os
import subprocess
import logging
import json
from datetime import datetime
from pathlib import Path
import tempfile
from typing import List
from modal import Image, App, method, enter, Dict, Volume, Mount, interact
from egfr_binder_rd2 import (
    EGFS, 
    EGFR, 
    COLABFOLD_GPU_CONCURRENCY_LIMIT, 
    MODAL_VOLUME_NAME,
    MSA_QUERY_HOST_URL,
    OUTPUT_DIRS,
    LOGGING_CONFIG
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

# Define constants for the modal volume path
MODAL_VOLUME_PATH = "/colabfold_data"

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


@app.cls(
    image=image,
    gpu='a100',
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
    def run_msa_generation(self, sequence: str) -> Path:
        """Run complete MSA generation pipeline."""
        fasta_path = self.save_sequences_as_fasta.remote([sequence])
        logger.info(f"Created FASTA file at: {fasta_path}")
        
        # Use a flat directory structure in msa_results
        out_dir = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["msa_results"]
        
        out_dir.mkdir(parents=True, exist_ok=True)
        result_dir = self.query_msa_server.remote(fasta_path, out_dir)
        return result_dir


@app.function(
    image=image,
    timeout=9600,
    volumes={MODAL_VOLUME_PATH: volume},
)
def get_msa_for_binder(binder_seq: str, target_seq: str=EGFR) -> Path:
    """Get the MSA results for a given binder sequence."""
    msa = MSAQuery()
    return msa.run_msa_generation.remote(f'{binder_seq}:{target_seq}')



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
        f'{EGFS}:{EGFR}'
    ]

    msa = MSAQuery()
    result_dir = msa.run_msa_generation.remote(test_sequences)
    print(f"MSA results saved to: {result_dir}")


@app.function(
    image=image,
    timeout=9600,
    volumes={MODAL_VOLUME_PATH: volume},
)
def a3m_from_template(binder_seq: str, parent_binder_seq: str, target_seq: str=EGFR) -> Path:
    """Get the a3m file path for a given binder sequence."""
    assert len(binder_seq) == len(parent_binder_seq), "Binder sequences must be the same length"
    
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
        return a3m_path
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
        
        return swap_binder_seq_into_a3m(binder_seq=binder_seq, template_a3m_path=template_a3m_path, output_path=a3m_path)


def get_a3m_path(binder_seq: str, target_seq: str) -> Path:
    seq_hash = hash_seq(f'{binder_seq}:{target_seq}')
    return Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["msa_results"] / f"{seq_hash}.a3m"


@app.function(
    image=image,
    timeout=9600,
    volumes={MODAL_VOLUME_PATH: volume},
)
def fold_mutated_binder(binder_seq: str, parent_binder_seq: str, target_seq: str=EGFR) -> Path:
    """Fold a mutated binder sequence."""
    a3m_path = a3m_from_template.remote(binder_seq=binder_seq, parent_binder_seq=parent_binder_seq, target_seq=target_seq)
    return fold.remote(a3m_path)

def get_metrics_from_hash(seq_hash: str) -> dict:
    """Extract metrics from folding results for a given sequence hash."""
    folded_dir = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["folded"]
    
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
            
            results.append({
                'seq_hash': seq_hash,
                'binder_sequence': binder_seq,
                'target_sequence': sequences.get('target', ''),
                'model_number': model_number,
                'binder_plddt': binder_plddt,
                'binder_pae': binder_pae,
                'pae_interaction': pae_interaction,
                'ptm': data.get('ptm', 0),
                'i_ptm': data.get('iptm', 0),
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
        return metrics_file
    
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
    
    return df
