import modal
from typing import List
from egfr_binder_rd2 import (
    EGFS, 
    EGFR, 
    COLABFOLD_GPU_CONCURRENCY_LIMIT, 
    MODAL_VOLUME_NAME,
    OUTPUT_DIRS,
    LOGGING_CONFIG,
    MODAL_VOLUME_PATH
)
import json
from pathlib import Path
from egfr_binder_rd2.utils import hash_seq

import logging

# Set up logging using configuration
logging.basicConfig(level=LOGGING_CONFIG["level"], format=LOGGING_CONFIG["format"])
logger = logging.getLogger(__name__)

# Define the container image with required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "pandas",
    )
)

app = modal.App("esm2-inference")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)


with image.imports():
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    import torch
    import pandas as pd
    import numpy as np

def get_esm2_pll(model, tokenizer, sequences: List[str], batch_size: int = 32, alpha: float = 0.1, beta: float = 0.1):
    """
    Run ESM2 predictions on a batch of sequences using the provided model and tokenizer.
    
    Args:
        model: The ESM2 model
        tokenizer: The ESM2 tokenizer
        sequences: List of sequences to process
        batch_size: Size of batches to process
        alpha: Probability adjustment parameter
        beta: Probability adjustment parameter
    
    Returns:
        List of prediction results for each sequence
    """
    all_results = []
    
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i+batch_size]
        batch_inputs = tokenizer(batch_sequences, return_tensors="pt", padding=True).to("cuda")

        with torch.no_grad():
            logits = model(**batch_inputs).logits

        token_probs = torch.softmax(logits, dim=-1)
        actual_token_probs = torch.gather(token_probs, 2, batch_inputs["input_ids"].unsqueeze(-1)).squeeze(-1)

        # Apply mask-consistent probability adjustment
        mask_consistent_probs = ((alpha + beta) / alpha) * actual_token_probs - (beta / alpha)
        mask_consistent_probs = torch.clamp(mask_consistent_probs, min=1e-16)

        pll = torch.log(mask_consistent_probs)

        # Handle padding
        mask = (batch_inputs["input_ids"] != tokenizer.pad_token_id).float()
        sequence_lengths = mask.sum(dim=1)
        sequence_log_pll = (pll * mask).sum(dim=1)
        normalized_log_pll = sequence_log_pll / sequence_lengths

        # Convert to CPU and numpy for each sequence in batch
        for seq_idx in range(len(batch_sequences)):
            seq_mask = mask[seq_idx]
            seq_length = int(sequence_lengths[seq_idx].item())
            
            result = {
                "sequence": batch_sequences[seq_idx],
                "token_probabilities": actual_token_probs[seq_idx, :seq_length].cpu().numpy(),
                "mask_consistent_probabilities": mask_consistent_probs[seq_idx, :seq_length].cpu().numpy(),
                "token_log_plls": pll[seq_idx, :seq_length].cpu().numpy(),
                "sequence_log_pll": sequence_log_pll[seq_idx].item(),
                "normalized_log_pll": normalized_log_pll[seq_idx].item(),
                "sequence_length": seq_length
            }
            all_results.append(result)

    return all_results

@app.cls(
    gpu="A100",
    image=image,
    container_idle_timeout=1200,
    timeout=9600,
    volumes={MODAL_VOLUME_PATH: volume},
)
class ESM2Model:
    @modal.enter()
    def setup(self):
        # Load model and tokenizer from Hugging Face
        self.model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D").cuda()
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.model.eval()

    @modal.method()
    def predict_batch(self, sequences: List[str], batch_size: int = 32, alpha: float = 0.1, beta: float = 0.1):
        return get_esm2_pll(self.model, self.tokenizer, sequences, batch_size, alpha, beta)

@app.function(image=image,
    timeout=9600,
    volumes={MODAL_VOLUME_PATH: volume},
    )
def process_sequences(sequences: List[str]=None, save_results: bool=True):
    # Create output directory if it doesn't exist
    results_dir = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS['esm2_pll_results']
    results_dir.mkdir(parents=True, exist_ok=True)

    if sequences is None:
        logger.info("No sequences provided, using all folded sequences")
        folded_metrics = pd.read_csv(Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS['metrics_csv'])
        sequences = folded_metrics['binder_sequence'].unique()
    
    logger.info(f"Processing {len(sequences)} sequences")
    logger.info(f"Results directory: {results_dir}")
    
    # Process only sequences that haven't been computed yet
    sequences_to_process = []
    cached_results = []
    
    for seq in sequences:
        # Add 'seq_' prefix to distinguish from binder-target pair hashes
        seq_hash = f"bdr_{hash_seq(seq)}"
        result_path = results_dir / f"{seq_hash}.json"
        
        if result_path.exists():
            # Load cached result
            with open(result_path, 'r') as f:
                cached_results.append(json.load(f))
        else:
            sequences_to_process.append(seq)

    logger.info(f"Sequences to process: {sequences_to_process}")
    
    # Process new sequences if any
    if sequences_to_process:
        logger.info(f"Processing {len(sequences_to_process)} new sequences")
        model = ESM2Model()
        new_results = model.predict_batch.remote(sequences_to_process)
        logger.info(f"Received predictions for {len(new_results)} sequences")
        
        if save_results:
            # Add hash to results and save individually
            for seq, result in zip(sequences_to_process, new_results):
                seq_hash = f"bdr_{hash_seq(seq)}"
                result['sequence_hash'] = seq_hash
                
                # Convert NumPy arrays to lists before saving
                result['token_probabilities'] = result['token_probabilities'].tolist()
                result['mask_consistent_probabilities'] = result['mask_consistent_probabilities'].tolist()
                result['token_log_plls'] = result['token_log_plls'].tolist()
                
                # Save individual result
                result_path = results_dir / f"{seq_hash}.json"
                logger.info(f"Saving result to {result_path}")
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2)
        
        # Combine with cached results
        all_results = cached_results + new_results
        logger.info(f"Total results: {len(all_results)}")
    else:
        logger.info("No new sequences to process")
        all_results = cached_results
    
    return all_results

@app.function(
    image=image,
    timeout=9600,
    volumes={MODAL_VOLUME_PATH: volume},
)
def update_pll_metrics():
    """Update metrics CSV with results from all ESM2 PLL calculations, only processing new sequences."""
    results_dir = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS['esm2_pll_results']
    metrics_file = Path(MODAL_VOLUME_PATH) / 'esm2_pll_metrics.csv'
    
    # Create metrics directory if it doesn't exist
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing metrics if available
    existing_df = pd.DataFrame()
    existing_hashes = set()
    if metrics_file.exists():
        existing_df = pd.read_csv(metrics_file)
        existing_hashes = set(existing_df['sequence_hash'].unique())
        logger.info(f"Found existing metrics for {len(existing_hashes)} sequence hashes")
    
    # Get all JSON files in results directory
    all_files = list(results_dir.glob("*.json"))
    all_hashes = set(f.stem for f in all_files)
    
    # Identify new hashes that need processing
    new_hashes = all_hashes - existing_hashes
    logger.info(f"Found {len(new_hashes)} new sequence hashes to process")
    
    if not new_hashes:
        logger.info("No new sequences to process")
        return existing_df
    
    # Process new hashes
    new_results = []
    for seq_hash in new_hashes:
        result_path = results_dir / f"{seq_hash}.json"
        if result_path.exists():
            with open(result_path, 'r') as f:
                result = json.load(f)
                
            
                
                # Extract relevant metrics
                metrics = {
                    'sequence_hash': seq_hash,
                    'sequence': result['sequence'],
                    'sequence_length': result['sequence_length'],
                    'normalized_log_pll': result['normalized_log_pll'],
                    'sequence_log_pll': result['sequence_log_pll'],
                }
                new_results.append(metrics)
    
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
        logger.info(f"Added metrics for {len(new_results)} new sequences, total {len(df)} entries")
    else:
        logger.info("No new valid results to add")
        df = existing_df
    
    return df



@app.local_entrypoint()
def test_run():
    binder_seqs = [
    'WVQLQESGGGLVQPGGSLRLSCAASGRTFSSYAMGWFRQAPGKQREFVAAIRWSGGYTYYTDSVKGRFTISRDNAKTTVYLQMNSLKPEDTAVYYCAATYLSSDYSRYALPQRPLDYDYWGQGTQVTVSSLE:LEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCKLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLRGCPTNGHHHHHH',
    'QVQLQESGGGLVQPGGSLRLSCAASGRTFSSHAMGWFRQAPGKQREFVAAIRWSGGYTYYTDSVKGRFTISRDNAKTTVYLQMNSLKPEDTAVYYCAATYLSSDYSRYALPQRPLDYDYWGQGTQVTVSSLE:LEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCKLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLRGCPTNGHHHHHH',
    'SVDEECPASYEGFCQNDGTCLYLEKLDRYACRCREGYIGERCEFRDLDYWLEQ',
    'DSECPLSHDGYCLHDGVCMYIEALDKYACNCVVGYIGERCQYRDLKWWENLEERLKEHRAKRLALLGPGPPGVVEKEKYKVSITEKVNPGGPATMPMTLTDSNGNKTTLTITVTPEGLEAIRKRRAGEKVKYTMTSTDTGDKFVLVDLDGYCLHDGVCMYIEALDKYACNCVVGYIGERCQYRDLK',
    'DSECPLSHDGYCLHDGVCMYIEALDKYACNCVVGYIGERCQYRDLKWWENLEERLKEHRAKRLALL',
        'NSDSECPLSHDGYCLHDGVCMYIEALDKYACNCVVGYIGERCQYRDLKWWELR',
    ]
    process_sequences.remote(binder_seqs)