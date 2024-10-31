import modal
from typing import List
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

# Define the container image with required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
    )
)

app = modal.App("esm2-inference")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

# Add these constants near the top with other constants
ESM2_WEIGHTS_URL = "https://huggingface.co/mhcelik/esm-efficient/resolve/main/650M.safetensors?download=true"
ESM2_WEIGHTS_PATH = f"{MODAL_VOLUME_PATH}/650M.safetensors"

with image.imports():
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    import torch

@app.cls(
    gpu="A10G",
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
    def predict_batch(self, sequences: List[str], batch_size: int = 4, alpha: float = 0.1, beta: float = 0.1):
        all_results = []
        
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            batch_inputs = self.tokenizer(batch_sequences, return_tensors="pt", padding=True).to("cuda")

            with torch.no_grad():
                logits = self.model(**batch_inputs).logits

            token_probs = torch.softmax(logits, dim=-1)
            actual_token_probs = torch.gather(token_probs, 2, batch_inputs["input_ids"].unsqueeze(-1)).squeeze(-1)

            # Apply mask-consistent probability adjustment
            mask_consistent_probs = ((alpha + beta) / alpha) * actual_token_probs - (beta / alpha)
            mask_consistent_probs = torch.clamp(mask_consistent_probs, min=1e-16)

            pll = torch.log(mask_consistent_probs)

            # Handle padding
            mask = (batch_inputs["input_ids"] != self.tokenizer.pad_token_id).float()
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

@app.function(image=image,
    timeout=9600,
    volumes={MODAL_VOLUME_PATH: volume},
    )
def process_sequences(sequences: List[str]):
    model = ESM2Model()
    return model.predict_batch.remote(sequences)


@app.local_entrypoint()
def main():
    binder_seqs = [
        'WVQLQESGGGLVQPGGSLRLSCAASGRTFSSYAMGWFRQAPGKQREFVAAIRWSGGYTYYTDSVKGRFTISRDNAKTTVYLQMNSLKPEDTAVYYCAATYLSSDYSRYALPQRPLDYDYWGQGTQVTVSSLE:LEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCKLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLRGCPTNGHHHHHH',
        'QVQLQESGGGLVQPGGSLRLSCAASGRTFSSHAMGWFRQAPGKQREFVAAIRWSGGYTYYTDSVKGRFTISRDNAKTTVYLQMNSLKPEDTAVYYCAATYLSSDYSRYALPQRPLDYDYWGQGTQVTVSSLE:LEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCKLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLRGCPTNGHHHHHH',
        'SVDEECPASYEGFCQNDGTCLYLEKLDRYACRCREGYIGERCEFRDLDYWLEQ',
        'DSECPLSHDGYCLHDGVCMYIEALDKYACNCVVGYIGERCQYRDLKWWENLEERLKEHRAKRLALLGPGPPGVVEKEKYKVSITEKVNPGGPATMPMTLTDSNGNKTTLTITVTPEGLEAIRKRRAGEKVKYTMTSTDTGDKFVLVDLDGYCLHDGVCMYIEALDKYACNCVVGYIGERCQYRDLK',
        'DSECPLSHDGYCLHDGVCMYIEALDKYACNCVVGYIGERCQYRDLKWWENLEERLKEHRAKRLALL',
        'NSDSECPLSHDGYCLHDGVCMYIEALDKYACNCVVGYIGERCQYRDLKWWELR',
    ]
    print(process_sequences.remote(binder_seqs))
