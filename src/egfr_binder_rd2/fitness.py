import modal
from egfr_binder_rd2.fold import calculate_solubility

get_msa = modal.Function.lookup("simplefold", 'get_msa_for_binder')
# a3m_from_template = modal.Function.lookup("simplefold", 'a3m_from_template')
fold_binder = modal.Function.lookup("simplefold", 'fold_binder')
update_metrics = modal.Function.lookup("simplefold", 'update_metrics_for_all_folded')
esm2_pll = modal.Function.lookup("esm2-inference", 'process_sequences')
esm2_pll_exact = modal.Function.lookup("esm2-inference", 'process_sequences_exact')
update_pll_metrics = modal.Function.lookup("esm2-inference", 'update_pll_metrics')
get_exact_plls = modal.Function.lookup("esm2-inference", 'update_exact_pll_metrics')
sample_sequences = modal.Function.lookup("bt-training", 'sample_sequences')
update_high_quality_metrics = modal.Function.lookup("simplefold", 'update_high_quality_metrics')


def get_fitness():

    
    fdf = update_metrics.remote()
    df = update_pll_metrics.remote()
    fdf = fdf.merge(df, left_on='binder_sequence', right_on='sequence', how='left')
    fdf['pae_interaction_rank'] = 1 - fdf['pae_interaction'].rank(pct=True)
    fdf['i_ptm_rank'] = fdf['i_ptm'].rank(pct=True)
    fdf['sequence_log_pll_rank'] = fdf['sequence_log_pll'].rank(pct=True)
    fdf['p_soluble_rank'] = fdf['p_soluble'].rank(pct=True)
    # fdf['fitness'] = (fdf['pae_interaction_rank'] + fdf['i_ptm_rank'] + fdf['sequence_log_pll_rank']) / 3
    fdf['fitness'] = (fdf['pae_interaction_rank'] + fdf['i_ptm_rank'] + fdf['sequence_log_pll_rank'] + fdf['p_soluble_rank']) / 4
    fdf = fdf.sort_values('fitness', ascending=False).reset_index(drop=True)

    exact_plls = get_exact_plls.remote()
    edf = fdf.merge(exact_plls.add_prefix('exact_'), left_on='binder_sequence', right_on='exact_sequence', how='left')
    edf['exact_sequence_log_pll_rank'] = edf['exact_sequence_log_pll'].rank(ascending=True, pct=True)
    edf['exact_fitness'] = (edf['i_ptm_rank'] + edf['exact_sequence_log_pll_rank'] + edf['pae_interaction_rank']) / 3


    # cols = [
    #     'seq_hash', 'binder_length', 'fitness', 'pae_interaction', 'i_ptm',  'sequence_log_pll', 'p_soluble',
    #     'exact_fitness', 'exact_sequence_log_pll', 'exact_sequence_log_pll_rank',
    #     'pae_interaction_rank', 'i_ptm_rank',
    #     'sequence_log_pll_rank', 'p_soluble_rank',
    #     'binder_plddt', 'binder_hydropathy','binder_pae',
    #         'ptm', 'binder_charged_fraction',
    #     'binder_hydrophobic_fraction',
    #     'binder_sequence',
    # ]
    # results = edf[cols].round(3)
    return edf

def get_exact_fitness():
    hq_results, top_ranked = update_high_quality_metrics.remote()
    exact_plls = get_exact_plls.remote()
    exact = top_ranked.merge(exact_plls, left_on='binder_sequence', right_on='sequence', how='left')
    exact['p_soluble'] = exact['binder_sequence'].apply(calculate_solubility)
    exact['p_soluble_rank'] = exact['p_soluble'].rank(ascending=True, pct=True)
    exact['fitness'] = exact['sequence_log_pll'] + exact['pae_interaction']
    exact['i_ptm_rank'] = exact['i_ptm'].rank(ascending=True, pct=True)
    exact['sequence_log_pll_rank'] = exact['sequence_log_pll'].rank(ascending=True, pct=True)
    exact['pae_interaction_rank'] = exact['pae_interaction'].rank(ascending=True, pct=True)
    exact['fitness'] = (exact['i_ptm_rank'] + exact['sequence_log_pll_rank'] + exact['pae_interaction_rank']) / 3
    return exact.sort_values('fitness', ascending=False).reset_index(drop=True)