import hashlib
import json
from pathlib import Path
from egfr_binder_rd2 import  OUTPUT_DIRS, MODAL_VOLUME_PATH
from egfr_binder_rd2 import EvolutionMetadata

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



def parse_fasta_lines(lines):
    """
    Parse FASTA-like lines into a dictionary of header:sequence pairs.
    
    Args:
        lines (list): List of strings containing FASTA format lines
        
    Returns:
        dict: Dictionary with headers as keys and sequences as values
    """
    sequences = {}
    current_header = None
    current_sequence = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('>'):
            # If we were building a sequence, save it
            if current_header is not None:
                sequences[current_header] = ''.join(current_sequence)
                current_sequence = []
            
            # Start new sequence, store header without '>'
            current_header = line[1:]
        else:
            # Add sequence line
            current_sequence.append(line)
    
    # Don't forget to save the last sequence
    if current_header is not None:
        sequences[current_header] = ''.join(current_sequence)
    
    return sequences

def swap_binder_seq_into_a3m(
    binder_seq, 
    template_a3m_path,  # Updated default path
    output_path=None,  # New optional parameter
):
    
    # Load template A3M
    with open(template_a3m_path, 'r') as template_file:
        template_lines = template_file.readlines()
    

    template_sequence = template_lines[2].strip()
    binder_length = int(template_lines[0].split(',')[0].strip('#'))
    target_sequence = template_sequence[binder_length:]
    
    seq_dict = parse_fasta_lines(template_lines[3:])
    seq_dict['101'] = binder_seq + ''.join(['-'] * len(target_sequence))

    with open(output_path, 'w') as output_file:
        # Write the header line from template
        output_file.writelines(template_lines[0])
        
        # Write the binder sequence (if provided) and target sequence
        output_file.write('>101\t102\n')
        output_file.write(f"{binder_seq}{target_sequence}\n")
        
        for header, seq in seq_dict.items():
            output_file.write(f">{header}\n")
            output_file.write(f"{seq}\n")
    
    return output_path

def load_evolution_metadata(metadata_file: Path) -> EvolutionMetadata:
    """Load evolution metadata from a JSON file"""
    with open(metadata_file, "r") as f:
        data = json.load(f)
    return EvolutionMetadata(**data)

def get_expression_model_path() -> Path:
    """Get path to saved expression model"""
    return Path(MODAL_VOLUME_PATH) / "expression_model.pt"


