# Solubility-Weighted Index,
# Bhandari, B.K., Gardner, P.P. and Lim, C.S.,(2020),
# doi: 10.1093/bioinformatics/btaa578

import argparse
import re
import os
import sys
import numpy as np
weights = {'A': 0.8356471476582918,
           'C': 0.5208088354857734,
           'E': 0.9876987431418378,
           'D': 0.9079044671339564,
           'G': 0.7997168496420723,
           'F': 0.5849790194237692,
           'I': 0.6784124413866582,
           'H': 0.8947913996466419,
           'K': 0.9267104557513497,
           'M': 0.6296623675420369,
           'L': 0.6554221515081433,
           'N': 0.8597433107431216,
           'Q': 0.789434648348208,
           'P': 0.8235328714705341,
           'S': 0.7440908318492778,
           'R': 0.7712466317693457,
           'T': 0.8096922697856334,
           'W': 0.6374678690957594,
           'V': 0.7357837119163659,
           'Y': 0.6112801822947587}

# Constants from logistic fitting
# prob = 1 / (1 + exp(-(a * x + b)));

A = 81.0581
B = -62.7775


def calculate_swi(sequence):
    """Calculate solubility probability for a given protein sequence.
    
    Args:
        sequence (str): Amino acid sequence using single letter codes
        
    Returns:
        tuple: (swi_score, solubility_probability)
    """
    # Validate sequence
    valid = re.compile('^[ACEDGFIHKMLNQPSRTWVY]+$')
    sequence = sequence.upper().replace('U', 'C')
    
    if not valid.search(sequence):
        raise ValueError("Invalid amino acid sequence")
        
    # Calculate SWI
    swi = np.mean([weights[i] for i in sequence])
    
    
    
    return swi

def calculate_solubility(sequence):
    # Calculate probability
    swi = calculate_swi(sequence)
    prob = 1/(1 + np.exp(-(A*swi + B)))
    
    return prob
