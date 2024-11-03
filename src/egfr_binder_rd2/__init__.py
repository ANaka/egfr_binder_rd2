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
    "esm2_pll_results_exact": Path("esm2_pll_results_exact/"),
    "bt_models": Path("bt_models/"),
    "rd1_fold_df": Path("rd1_fold_df.csv"),
    "evolution_trajectories": Path("evolution_trajectories/"),
    "inference_results": Path("inference_results/"),
    "folded_high_quality": Path("folded_high_quality/"),
    "esm2_exact_pll_metrics": Path("esm2_exact_pll_metrics.csv"),
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
OFFICIAL_EGFR = 'LEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPS'

# COLABFOLD_GPU_CONCURRENCY_LIMIT = 80
COLABFOLD_GPU_CONCURRENCY_LIMIT = 55

# Volume Name
MODAL_VOLUME_NAME = "egfr_binders"


class ExpertType(enum.Enum):
    ESM = "esm"
    iPAE = "pae_interaction"
    iPTM = "i_ptm"
    pLDDT = "binder_plddt"
    HYDROPATHY = "binder_hydropathy"
    PLL = "sequence_log_pll"
    
    @classmethod
    def from_str(cls, label: str) -> 'ExpertType':
        return cls(label.lower())

@dataclass
class ExpertConfig:
    type: ExpertType
    temperature: float = 1.0
    make_negative: bool = False
    transform_type: str = "standardize"
    model_name: str = "facebook/esm2_t6_8M_UR50D"

# Add new config class for ensemble models
@dataclass
class PartialEnsembleExpertConfig(ExpertConfig):
    num_heads: int = 10
    dropout: float = 0.1
    explore_weight: float = 0.2

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

# Add new default expert configurations
DEFAULT_EXPERT_CONFIGS = [
    # ExpertConfig(
    #     type=ExpertType.ESM, 
    #     temperature=2.0,
    #     model_name="facebook/esm2_t6_8M_UR50D",
    #     # model_name='facebook/esm2_t33_650M_UR50D'
    # ),
    PartialEnsembleExpertConfig(
        type=ExpertType.iPAE,
        temperature=1.0,
        make_negative=True,
        transform_type="standardize",
        num_heads=10,
        dropout=0.15,
        explore_weight=0.2,
    ),
    PartialEnsembleExpertConfig(
        type=ExpertType.iPTM,
        temperature=1.0,
        make_negative=False,
        transform_type="standardize",
        num_heads=10,
        dropout=0.15,
        explore_weight=0.2,
    ),
    PartialEnsembleExpertConfig(
        type=ExpertType.pLDDT,
        temperature=1.0,
        make_negative=False,
        transform_type="standardize",
        num_heads=10,
        dropout=0.15,
        explore_weight=0.2,
    ),
    PartialEnsembleExpertConfig(
        type=ExpertType.HYDROPATHY,
        temperature=1.0,
        make_negative=True,
        transform_type="standardize",
        num_heads=3,
        dropout=0.1,
        explore_weight=0.,
    ),
    PartialEnsembleExpertConfig(
        type=ExpertType.PLL,
        temperature=1.0,
        make_negative=False,
        transform_type="standardize",
        num_heads=10,
        dropout=0.15,
        explore_weight=0.2,
    ),
]
