import os
import subprocess
import logging
import json
from datetime import datetime
from pathlib import Path
import tempfile

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
    get_a3m_path,
    get_fasta_path,
    get_folded_dir,
    initialize_metrics_db,
    update_metrics_db,
    get_metrics
)

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

# Initialize Metrics Database
initialize_metrics_db()

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
        
        logger.info(f'input directory contents: {os.listdir(input_path)}')
        logger.info(f"Output directory contents: {os.listdir(out_dir)}")
        logger.info(f'current directory contents: {os.listdir(".")}')

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
)
class MSAQuery:

    @enter()
    def setup(self):
        os.environ["PATH"] = "/localcolabfold/colabfold-conda/bin:" + os.environ["PATH"]

    @method()
    def save_sequences_as_fasta(self, sequences: list[str]) -> Path:
        """Save sequences to a FASTA file, using sequence hashes as headers."""
        fasta_path = get_fasta_path(sequences)
        # Write sequences to FASTA
        with open(fasta_path, "w") as f:
            for seq in sequences:
                seq_hash = hash_seq(seq)
                f.write(f">{seq_hash}\n{seq}\n")
        
        logger.info(f"Saved {len(sequences)} sequences to {fasta_path}")
        return fasta_path

    @method()
    def query_msa_server(self, input_path: str | Path, out_dir: str | Path, host_url=MSA_QUERY_HOST_URL, retries=5):
        """Query the MSA server to generate alignments."""
        # Convert paths to Path objects
        input_path = Path(input_path)
        out_dir = Path(out_dir)
        
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
            return out_dir
        except subprocess.CalledProcessError as e:
            logger.error(f"MSA generation failed with error: {e}")
            logger.error(f"Error output: {e.stderr}")
            raise

    @method()
    def run_msa_generation(self, sequences: list[str]) -> Path:
        """Run complete MSA generation pipeline."""
        fasta_path = self.save_sequences_as_fasta.remote(sequences)
        logger.info(f"Created FASTA file at: {fasta_path}")
        out_dir = OUTPUT_DIRS["msa_results"] / fasta_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        result_dir = self.query_msa_server.remote(fasta_path, out_dir)
        return result_dir

@app.local_entrypoint()
def test_msa_server_query():
    """Test the MSA server query functionality."""
    test_sequences = [
        f'{EGFS}:{EGFR}'
    ]

    msa = MSAQuery()
    result_dir = msa.run_msa_generation.remote(test_sequences)
    print(f"MSA results saved to: {result_dir}")