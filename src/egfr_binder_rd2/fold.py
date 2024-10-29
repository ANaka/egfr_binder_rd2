import os
import argparse
import tempfile
import shutil
import subprocess
import logging
from egfr_binder_rd2 import TEMPLATE_A3M_PATH, EGFS, EGFR, COLABFOLD_GPU_CONCURRENCY_LIMIT, FOLD_RESULTS_DIR
from modal import Image, App, method, enter, Dict, Volume, Mount
from egfr_binder_rd2.utils import get_mutation_diff, hash_seq
import io
import zipfile
import re
import json
from datetime import datetime
# from Bio import PDB  # Add this import at the module level
# import numpy as np
# import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = App("colabfold")

# Initialize the Volume
volume = Volume.from_name("fold-results-volume", create_if_missing=True)

# Create a mount for the template file
template_mount = Mount.from_local_dir(
    local_path=os.path.dirname(TEMPLATE_A3M_PATH),
    remote_path="/root/templates"
)

# Update the template path constant to use the mounted path
MOUNTED_TEMPLATE_PATH = os.path.join(
    "/root/templates", 
    os.path.basename(TEMPLATE_A3M_PATH)
)

image = (
    Image
    .debian_slim(python_version="3.11")
    .micromamba(python_version="3.11")
    .apt_install("wget", "git", "curl")
    .run_commands('wget https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/install_colabbatch_linux.sh')
    .run_commands('bash install_colabbatch_linux.sh', gpu="a100",)
    # Update the pip installations to use biopython instead of Bio
    .pip_install(
        'biopython==1.81',  # Specify version for consistency
        'pandas',
        'numpy'
    )
)

with image.imports():
    from Bio import PDB
    import numpy as np
    import pandas as pd

def generate_a3m_files(
    binder_sequences, 
    output_folder="/data/input_a3m", 
    template_a3m_path=MOUNTED_TEMPLATE_PATH,  # Updated default path
    target_a3m_path=None,  # New optional parameter
    target_sequence=None
):
    os.makedirs(output_folder, exist_ok=True)
    
    # Load template A3M
    with open(template_a3m_path, 'r') as template_file:
        template_lines = template_file.readlines()
    
    # Load target A3M if provided
    if target_a3m_path:
        with open(target_a3m_path, 'r') as target_file:
            target_lines = target_file.readlines()
        target_sequence = ''.join([line.strip() for line in target_lines if not line.startswith('#') and not line.startswith('>')])
    else:
        # Extract the target sequence from the template if target_sequence is None
        if target_sequence is None:
            template_sequence = template_lines[2].strip()
            binder_length = int(template_lines[0].split(',')[0].strip('#'))
            target_sequence = template_sequence[binder_length:]
    
    for name, binder_seq in binder_sequences.items():
        output_path = os.path.join(output_folder, f"{name}.a3m")
        
        with open(output_path, 'w') as output_file:
            # Write the header lines from template
            output_file.writelines(template_lines[:2])
            
            # Write the binder sequence (if provided) and target sequence
            if binder_seq:
                output_file.write(f"{binder_seq}{target_sequence}\n")
            else:
                output_file.write(f"{target_sequence}\n")
            
            # Write the rest of the template file
            output_file.writelines(template_lines[3:])
    
    return output_folder

@app.cls(
    image=image,
    gpu='a100',
    timeout=9600,
    concurrency_limit=COLABFOLD_GPU_CONCURRENCY_LIMIT,
    volumes={"/data": volume},
    mounts=[template_mount]
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
    def fold(self, binder_sequences, template_a3m_path, target_a3m_path=None, target_sequence=None,  **kwargs):
        input_path = generate_a3m_files(
            binder_sequences=binder_sequences,
            output_folder="/data/input_a3m",  # Using Volume path
            template_a3m_path=template_a3m_path,
            target_a3m_path=target_a3m_path,  # Pass the target A3M path
            target_sequence=target_sequence
        )
        logger.info(f"Generated A3M files in: {input_path}")

        out_dir = "/data/output"  # Using Volume path
        os.makedirs(out_dir, exist_ok=True)
        logger.info(f"Created output directory: {out_dir}")

        cmd = ["colabfold_batch", input_path, out_dir]
        
        # Handle arguments
        for key, value in kwargs.items():
            key = key.replace('_', '-')
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            elif value is not None:
                cmd.extend([f"--{key}", str(value)])
                
        # Removed the zip flag
        # cmd.append('--zip')
        
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
        
        # Directly process the output directory
        all_results = self.extract_metrics_and_pdbs(out_dir)
        logger.info(f"Extracted {len(all_results)} results")
        
        return all_results

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
    
    @staticmethod
    def extract_metrics_and_pdbs(output_dir):
        logger.info(f"Processing metrics and PDBs in {output_dir}")
        all_results = []
        pdbs = {}
        
        for filename in os.listdir(output_dir):
            if filename.endswith('.pdb'):
                pdb_path = os.path.join(output_dir, filename)
                
                # Derive the JSON filename by replacing '_unrelaxed_' with '_scores_' and changing the extension
                json_filename = filename.replace('_unrelaxed_', '_scores_').replace('.pdb', '.json')
                json_path = os.path.join(output_dir, json_filename)
                
                if not os.path.exists(json_path):
                    logger.warning(f"Missing JSON file for PDB: {json_filename}")
                    continue
                
                # Read and process the PDB file
                with open(pdb_path, 'r') as pdb_file:
                    pdb_content = pdb_file.read()
                    sequences = LocalColabFold.extract_sequence_from_pdb(pdb_content)
                
                # Read and process the JSON scores file
                with open(json_path, 'r') as json_file:
                    data = json.load(json_file)
                    
                    plddt_array = np.array(data.get('plddt', []))
                    pae_array = np.array(data.get('pae', []))
                    
                    # Extract model number safely
                    rank_match = re.search(r'rank_(\d+)', filename)
                    model_number = int(rank_match.group(1)) if rank_match else 0
                    
                    binder_seq = sequences.get('binder', '')
                    binder_length = len(binder_seq)  # **Dynamic binder_length per result**
                    target_length = len(sequences.get('target', ''))
                    
                    pae_interaction = 0
                    if pae_array.size:
                        # Ensure that the pae array dimensions accommodate dynamic binder lengths
                        if pae_array.shape[0] >= binder_length and pae_array.shape[1] >= binder_length:
                            pae_interaction = (pae_array[binder_length:, :binder_length].mean() + pae_array[:binder_length, binder_length:].mean()) / 2
                        else:
                            logger.warning(f"PAE array shape {pae_array.shape} does not match binder_length {binder_length}")
                    
                    binder_plddt = plddt_array[:binder_length].mean() if plddt_array.size else 0
                    binder_pae = pae_array[:binder_length, :binder_length].mean() if pae_array.size else 0
                    
                    result = {
                        'seq_name': filename.split('_unrelaxed_rank')[0],
                        'binder_sequence': binder_seq,
                        'target_sequence': sequences.get('target', ''),
                        'binder_length': binder_length,
                        'target_length': target_length,
                        'model_number': model_number,  # Use the safely extracted model number
                        'binder_plddt': float(binder_plddt),
                        'binder_pae': float(binder_pae),
                        'pae_interaction': float(pae_interaction),
                        'ptm': data.get('ptm', 0),
                        'i_ptm': data.get('iptm', 0),
                        'seq_id': hash_seq(sequences.get('binder', '')),
                        'pdb_content': pdb_content
                    }
                    all_results.append(result)
                    pdbs[f"{filename}_model_{result['model_number']}"] = pdb_content

                    logger.info(f"Processed PDB and scores for {filename}_model_{result['model_number']}")
        
        logger.info(f"Extracted {len(all_results)} total results and {len(pdbs)} PDB files")
        return all_results

@app.function(
    timeout=4800,
    volumes={"/data": volume},
    mounts=[template_mount]  # Mount remains the same
)
def fold_and_extract(
    binder_sequences: dict, 
    template_a3m_path: str = MOUNTED_TEMPLATE_PATH,  # Use the mounted path
    target_a3m_path: str = None,  # New optional parameter
    target_sequence: str = None, 
    **kwargs
):
    lcf = LocalColabFold()
    result = lcf.fold.remote(
        binder_sequences=binder_sequences,
        template_a3m_path=template_a3m_path,
        target_a3m_path=target_a3m_path,  # Pass the target A3M path
        target_sequence=target_sequence,
        **kwargs
    )
    return result  # Now returns a list of results

@app.function(
    timeout=12800,
    volumes={"/data": volume},
    mounts=[template_mount]  # Mount remains the same
)
def parallel_fold_and_extract(
    binder_sequences: dict, 
    template_a3m_path: str = None, 
    target_a3m_path: str = None,  # New optional parameter
    target_sequence: str = None,  # New optional parameter
    batch_size: int = 10,
    **kwargs
):
    if template_a3m_path is None:
        template_a3m_path = MOUNTED_TEMPLATE_PATH  # Use the mounted path
    
    lcf = LocalColabFold()
    all_results = []
    
    # Prepare batches
    batches = []
    for i in range(0, len(binder_sequences), batch_size):
        batch = dict(list(binder_sequences.items())[i:i+batch_size])
        batches.append((batch, template_a3m_path, target_a3m_path, target_sequence))
    
    all_results = []
    for result in lcf.fold.starmap(batches, kwargs=kwargs):
        all_results.extend(result)
    
    return all_results

@app.local_entrypoint()
def test():
    # Use the mounted template path
    template_a3m_path = MOUNTED_TEMPLATE_PATH
    target_a3m_path = "/root/templates/custom_target.a3m"  # Example target A3M path
    binder_sequences = {
        "binder1": EGFS,
    }
    target_sequence = EGFR
    results_a3m_with_target = fold_and_extract.remote(
        template_a3m_path=template_a3m_path,
        target_a3m_path=target_a3m_path,  # Pass the target A3M path
        binder_sequences=binder_sequences,
        target_sequence=target_sequence,
        num_recycle=1,
        num_models=1
    )

    # Test a3m-based folding without provided target sequence
    results_a3m_without_target = fold_and_extract.remote(
        template_a3m_path=template_a3m_path,
        binder_sequences=binder_sequences,
        num_recycle=1,
        num_models=1
    )

@app.local_entrypoint()
def manual_parallel_fold():
    def mutate_seq(seq, position, new_aa):
        return seq[:position-1] + new_aa + seq[position:]

    def apply_mutation(seq, mutation_str):
        new_aa = mutation_str[-1]
        position = int(mutation_str[:-1])
        return mutate_seq(seq, position, new_aa)

    egfs = 'NSYPGCPSSYDGYCLNGGVCMHIESLDSYTCNCVIGYSGDRCQTRDLRWW'
    seed = egfs
    seed = 'PSYSGCPSSYDGYCGNGGVCMHIESLDSYTCQCVIGYSGDRVQTRDLRWT'
    seed = 'ISYSACPLSYDGVCGNGGVCKHALSLDSYTCQCVWGYSGDRVQTRDLRYT'

    mutations = []
    for pos in range(7, 50):
        mutations.append(f'{pos}A')
    seqs = {}
    for mutation in mutations:
        seq = apply_mutation(seed, mutation)
        seqs[hash_seq(seq)] = seq
        
    # Use the mounted template path
    template_a3m_path = MOUNTED_TEMPLATE_PATH
    results = list(parallel_fold_and_extract.remote(
        binder_sequences=seqs, 
        batch_size=3, 
        num_recycle=1, 
        num_models=1
    ))
    fold_df = pd.DataFrame(results)
    fold_df['mut_str'] = fold_df['binder_sequence'].apply(get_mutation_diff, seq2=EGFS)
    fold_df['seq_name'] = fold_df['binder_sequence'].apply(hash_seq)
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = FOLD_RESULTS_DIR / f'fold_results_{now}.csv'
    # save the fold_df
    fold_df.to_csv(fp, index=False)
    
@app.local_entrypoint()
def manual_parallel_fold_validate():
    seed = 'ISYSACPLSYDGVCGNGGVCKHALSLDSYTCQCVWGYSGDRVQTRDLRYT'

    seqs = {}
    seqs[hash_seq(seed)] = seed
        
    # Use the mounted template path
    template_a3m_path = MOUNTED_TEMPLATE_PATH
    results = list(parallel_fold_and_extract.remote(
        binder_sequences=seqs, 
        batch_size=3, 
        num_recycle=3, 
        num_models=5
    ))
    fold_df = pd.DataFrame(results)
    fold_df['mut_str'] = fold_df['binder_sequence'].apply(get_mutation_diff, seq2=EGFS)
    fold_df['seq_name'] = fold_df['binder_sequence'].apply(hash_seq)
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = FOLD_RESULTS_DIR / f'fold_results_{now}.csv'
    # save the fold_df
    fold_df.to_csv(fp, index=False)

@app.local_entrypoint()
def quick_test():
    # Use the mounted template path
    template_a3m_path = MOUNTED_TEMPLATE_PATH
    binder_sequences = {
        "test_binder": "NSYPGCPSSYDGYCLNGGVCMHIESLDSYTCNCVIGYSGDRCQTRDLRWW",
    }
    
    results = fold_and_extract.remote(
        binder_sequences=binder_sequences,
        template_a3m_path=template_a3m_path,  # Use the mounted path
        num_recycle=1,
        num_models=1
    )
    
    print("Test Results:", results)




