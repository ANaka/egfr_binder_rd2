from pathlib import Path
import enum
from dataclasses import dataclass
from typing import List
from datetime import datetime
import json

# Configuration Constants
MSA_QUERY_HOST_URL = "https://api.colabfold.com"

OUTPUT_DIRS = {
    "a3m": Path("a3m/"),
    "folded": Path("folded/"),
    "fastas": Path("fastas/"),
    "metrics_csv": Path("metrics.csv"),
    "msa_results": Path("msa_results/"),
    "lineage_csv": Path("lineage.csv"),
    "esm2_pll_results": Path("esm2_pll_results/"),
    "bt_models": Path("bt_models/"),
    "rd1_fold_df": Path("rd1_fold_df.csv"),
    "evolution_trajectories": Path("evolution_trajectories/"),
}

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(levelname)s - %(message)s"
}

# Directory Paths
DATA_DIR = Path(__file__).parent.parent / "data"
FOLD_RESULTS_DIR = DATA_DIR / "fold_results"
PEPMLM_RESULTS_DIR = DATA_DIR / "pepmlm_results"
EVO_PROT_GRAD_RESULTS_DIR = DATA_DIR / "evo_prot_grad_results"
MODAL_VOLUME_PATH = "/colabfold_data"
# Example Sequences
EGFS = 'NSYPGCPSSYDGYCLNGGVCMHIESLDSYTCNCVIGYSGDRCQTRDLRWWELR'
EGFR = ('LEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVA'
        'GYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRF'
        'SNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIIC'
        'AQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVN'
        'PEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEF'
        'KDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAW'
        'PENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTI'
        'NWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCKL'
        'LEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTL'
        'VWKYADAGHVCHLCHPNCTYGCTGPGLRGCPTNGHHHHHH')

# COLABFOLD_GPU_CONCURRENCY_LIMIT = 80
COLABFOLD_GPU_CONCURRENCY_LIMIT = 10

# Volume Name
MODAL_VOLUME_NAME = "egfr_binders"


class ExpertType(enum.Enum):
    ESM = "esm"
    iPAE = "pae_interaction"
    iPTM = "i_ptm"
    
    @classmethod
    def from_str(cls, label: str) -> 'ExpertType':
        return cls(label.lower())

@dataclass
class ExpertConfig:
    type: ExpertType
    weight: float = 1.0
    temperature: float = 1.0
    make_negative: bool = False
    transform_type: str = "rank"
    model_name: str = "facebook/esm2_t6_8M_UR50D"

@dataclass
class EvolutionMetadata:
    """Metadata for tracking evolution progress"""
    start_time: str
    config: dict
    parent_sequences: List[str]
    generation_metrics: List[dict] = None

    @classmethod
    def create(cls, config: dict, parent_sequences: List[str]):
        return cls(
            start_time=datetime.now().isoformat(),
            config=config,
            parent_sequences=parent_sequences,
            generation_metrics=[]
        )
    
    def add_generation(self, generation: int, metrics: dict):
        if self.generation_metrics is None:
            self.generation_metrics = []
        self.generation_metrics.append({
            "generation": generation,
            **metrics
        })
    
    def save(self, output_dir: Path):
        metadata_file = output_dir / f"evolution_{self.start_time}.json"
        with open(metadata_file, "w") as f:
            json.dump(self.__dict__, f, indent=2)