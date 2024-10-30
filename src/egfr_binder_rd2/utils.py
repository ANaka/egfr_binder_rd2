import hashlib
import json
from pathlib import Path
from egfr_binder_rd2 import  OUTPUT_DIRS

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
    for i, (aa1, aa2) in enumerate(zip(seq1, seq2)):
        if aa1 != aa2:
            mutations.append(f"{aa1}{i+1}{aa2}")
    
    return ",".join(mutations)

def get_a3m_path(binder_seq: str, target_seq: str) -> Path:
    """
    Generate the file path for the A3M file based on the binder and target sequences.
    
    Args:
        binder_seq (str): The binder protein sequence.
        target_seq (str): The target protein sequence.
    
    Returns:
        Path: The path to the A3M file.
    """
    binder_hash = hash_seq(binder_seq)
    target_hash = hash_seq(target_seq)
    filename = f"{binder_hash}_{target_hash}.a3m"
    return OUTPUT_DIRS["a3m"] / filename

def get_fasta_path(sequences: list[str]) -> Path:
    """
    Generate the file path for the FASTA file based on the sequences.
    
    Args:
        sequences (list[str]): A list of protein sequences.
    
    Returns:
        Path: The path to the FASTA file.
    """
    all_seqs_hash = hash_seq("".join(sequences))
    return OUTPUT_DIRS["fastas"] / f"{all_seqs_hash}.fasta"

def get_folded_dir(binder_seq: str, target_seq: str) -> Path:
    """
    Generate the directory path for storing folded structures based on the binder and target sequences.
    
    Args:
        binder_seq (str): The binder protein sequence.
        target_seq (str): The target protein sequence.
    
    Returns:
        Path: The directory path for folded structures.
    """
    binder_hash = hash_seq(binder_seq)
    target_hash = hash_seq(target_seq)
    return OUTPUT_DIRS["folded"] / f"{binder_hash}_{target_hash}"

def initialize_metrics_db():
    """
    Initialize the metrics database by creating the necessary directories and the JSON file if they don't exist.
    """
    metrics_db_path = OUTPUT_DIRS["metrics_db"]
    metrics_db_path.parent.mkdir(parents=True, exist_ok=True)
    if not metrics_db_path.exists():
        with open(metrics_db_path, "w") as f:
            json.dump({}, f)

def update_metrics_db(entry: dict):
    """
    Update the metrics database with a new entry.
    
    Args:
        entry (dict): The metrics entry to add.
    """
    metrics_db_path = OUTPUT_DIRS["metrics_db"]
    if not metrics_db_path.exists():
        raise FileNotFoundError(f"Metrics database not found at {metrics_db_path}")
    
    with open(metrics_db_path, "r") as f:
        metrics_db = json.load(f)
    
    metrics_db[entry["id"]] = entry
    
    with open(metrics_db_path, "w") as f:
        json.dump(metrics_db, f, indent=4)

def get_metrics(entry_id: str) -> dict:
    """
    Retrieve metrics for a specific entry ID.
    
    Args:
        entry_id (str): The unique identifier for the metrics entry.
    
    Returns:
        dict: The metrics entry if found, else an empty dictionary.
    """
    metrics_db_path = OUTPUT_DIRS["metrics_db"]
    if not metrics_db_path.exists():
        raise FileNotFoundError(f"Metrics database not found at {metrics_db_path}")
    
    with open(metrics_db_path, "r") as f:
        metrics_db = json.load(f)
    
    return metrics_db.get(entry_id, {})

