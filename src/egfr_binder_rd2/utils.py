import hashlib
from egfr_binder_rd2 import PEPMLM_RESULTS_DIR, FOLD_RESULTS_DIR
import re
import pandas as pd

def hash_seq(sequence):
    """
    Generate a hash for a given protein sequence.
    
    Args:
    sequence (str): The protein sequence to hash
    
    Returns:
    str: A hexadecimal string representation of the hash
    """
    # Remove any whitespace and convert to uppercase
    cleaned_sequence = ''.join(sequence.split()).upper()
    
    # Create a SHA256 hash object
    hasher = hashlib.sha256()
    
    # Update the hasher with the cleaned sequence encoded as UTF-8
    hasher.update(cleaned_sequence.encode('utf-8'))
    
    # Return the hexadecimal representation of the hash
    return hasher.hexdigest()[:6]


def get_mutation_diff(seq1, seq2):

    """
    Compare two sequences and return a string of mutations.
    
    Args:
    seq1 (str): The original sequence
    seq2 (str): The mutated sequence
    
    Returns:
    str: A comma-separated string of mutations in the format {original_aa}{position}{new_aa}
    """
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of equal length")
    
    mutations = []
    for i, (aa1, aa2) in enumerate(zip(seq2, seq1)):
        if aa1 != aa2:
            mutations.append(f"{aa1}{i+1}{aa2}")
    
    return ",".join(mutations)


def get_mlm_results():
    csvs = list(PEPMLM_RESULTS_DIR.glob('*.csv'))
    dfs = []
    for csv in csvs:
        _df = pd.read_csv(csv)
        match = re.search(r'gen(\d+)', str(csv))
        if match:
            gen_num = int(match.group(1))
        else:
            gen_num = None
        _df['generation'] = gen_num
        dfs.append(_df)
    pepmlm_df = pd.concat(dfs).sort_values('unmasked_ppl').reset_index(drop=True)
    return pepmlm_df

def get_mlm_ids():
    mlm_df = get_mlm_results()
    return mlm_df['seq_id'].unique().tolist()

