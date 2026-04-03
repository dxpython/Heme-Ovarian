from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from heme_pipeline.config import PipelineSettings, load_settings, resolve_project_paths
from heme_pipeline.consensus_agent.plots import plot_cindex_comparison
from heme_pipeline.consensus_agent.selector import results_to_comparison_table, select_best_model
from heme_pipeline.data_ingestion import (
    align_clinical_columns,
    detect_survival_columns,
    normalize_expression_matrix,
    read_gene_list,
    read_table,
)
from heme_pipeline.deg import filter_deg, plot_deg_heatmap, plot_volcano, run_deg_welch_ttest
from heme_pipeline.enrichment.go_kegg import plot_enrichment_bar, run_enrichr_multilib
from heme_pipeline.expression_validation.boxplots import plot_four_genes_by_binary_group
from heme_pipeline.gsea_gsva.gsea import plot_gsea_leading_edge, run_gsea_prerank
from heme_pipeline.gsea_gsva.gsva import run_gsva_rscript
from heme_pipeline.hmrg_intersection import intersect_deg_hmrg, plot_venn2
from heme_pipeline.immune_ssgsea.ssgsea import compare_high_low, plot_immune_heatmap, run_ssgsea_immune
from heme_pipeline.km_roc.km_curves import plot_km_two_group
from heme_pipeline.km_roc.td_roc import plot_td_roc
from heme_pipeline.logging_utils import get_logger, setup_logging
from heme_pipeline.mutation_analysis.maf import load_maf, mutation_matrix_from_maf, plot_mutation_landscape
from heme_pipeline.preprocess.transforms import impute_median, subset_common_genes
from heme_pipeline.reporting.export import write_table
from heme_pipeline.reporting.tables import table1_baseline, table2_multivariate_signature
from heme_pipeline.risk_model.plots import (
    plot_risk_distribution,
    plot_risk_survival_scatter,
    plot_signature_heatmap,
)
from heme_pipeline.risk_model.score import compute_cutoff, cox_risk_from_training, stratify_risk
from heme_pipeline.survival_models.pipelines import (
    train_coxnet_variant,
    train_rsf,
    train_stepwise_cox,
    train_svm_rfe_cox,
    train_xgboost_survival,
)
from heme_pipeline.survival_models.univariate_cox import univariate_cox_screen
from heme_pipeline.tide_analysis.tide import load_tide_table

LOG = get_logger("runner")


def _cohort_sample_column(settings: PipelineSettings, expression_key: str) -> str:
    for c in (settings.cohorts.training, settings.cohorts.validation, settings.cohorts.deg_reference):
        if c.expression_key == expression_key:
            return c.sample_id_column_clinical
    return "sample_id"


def _load_expr_matrix(
    settings: PipelineSettings,
    project_root: Path,
    dataset_key: str,
    gene_column: str | None,
) -> pd.DataFrame:
    ds = getattr(settings.datasets, dataset_key)
    path = project_root / ds.expression if ds.expression else None
    if not path or not path.is_file():
        raise FileNotFoundError(f"expression path for {dataset_key}: {path}")
    df = read_table(path, file_format=ds.format)
    return normalize_expression_matrix(df, gene_column=gene_column, samples_as="columns")


def _load_clinical(
    settings: PipelineSettings,
    project_root: Path,
    dataset_key: str,
) -> pd.DataFrame:
    ds = getattr(settings.datasets, dataset_key)
    if not ds.clinical:
        raise FileNotFoundError(f"clinical path missing for {dataset_key}")
    path = project_root / ds.clinical
    df = read_table(path, file_format=ds.format)
    aliases = settings.column_map.global_.sample_aliases
    sample_col = _cohort_sample_column(settings, dataset_key)
    return align_clinical_columns(df, aliases, sample_col)


def _prepare_survival(
    clinical: pd.DataFrame,
    settings: PipelineSettings,
) -> tuple[pd.DataFrame, str, str]:
    tcol, ecol = detect_survival_columns(
        clinical,
        settings.column_map.global_.survival_time_aliases,
        settings.column_map.global_.survival_event_aliases,
        settings.survival.time_col,
        settings.survival.event_col,
    )
    out = clinical.copy()
    out[settings.survival.time_col] = pd.to_numeric(out[tcol], errors="coerce")
    ev = pd.to_numeric(out[ecol], errors="coerce")
    out[settings.survival.event_col] = (ev > 0).astype(int)
    return out, settings.survival.time_col, settings.survival.event_col


def run_deg_module(settings: PipelineSettings, project_root: Path, results_dir: Path, fig_dir: Path) -> None:
    expr = _load_expr_matrix(settings, project_root, settings.deg.cohort, gene_column=None)
    clin = _load_clinical(settings, project_root, settings.deg.cohort)
    clin = align_clinical_columns(
        clin,
        settings.column_map.global_.sample_aliases,
        settings.cohorts.deg_reference.sample_id_column_clinical,
    )
    groups = clin.set_index("sample_id")[settings.deg.group_column]
    groups = groups.reindex(expr.columns)
    deg = run_deg_welch_ttest(expr, groups, settings.deg.tumor_label, settings.deg.normal_label)
    deg_path = results_dir / "deg" / "GSE18520_tumor_vs_normal_deg_full.tsv"
    write_table(deg, deg_path)
    deg_f = filter_deg(deg, settings.deg.padj_threshold, settings.deg.log2fc_threshold)
    write_table(deg_f, results_dir / "deg" / "GSE18520_tumor_vs_normal_deg_filtered.tsv")
    plot_volcano(
        deg,
        settings.deg.padj_threshold,
        settings.deg.log2fc_threshold,
        fig_dir / "Figure1" / "Figure1A_volcano.pdf",
        fig_dir / "Figure1" / "Figure1A_volcano.png",
        settings.reporting.dpi,
        settings.reporting.export_pdf,
        settings.reporting.export_png,
    )
    sig_genes = deg_f["gene"].astype(str).tolist()[: min(50, len(deg_f))]
    plot_deg_heatmap(
        expr,
        sig_genes,
        groups,
        fig_dir / "Figure1" / "Figure1B_deg_heatmap.pdf",
        fig_dir / "Figure1" / "Figure1B_deg_heatmap.png",
        settings.reporting.dpi,
        settings.reporting.export_pdf,
        settings.reporting.export_png,
    )


def run_hmrg_enrichment(settings: PipelineSettings, project_root: Path, results_dir: Path, fig_dir: Path) -> None:
    hmrg_path = project_root / settings.hmrg.gene_list_path
    hmrg = read_gene_list(hmrg_path)
    deg_f = pd.read_csv(results_dir / "deg" / "GSE18520_tumor_vs_normal_deg_filtered.tsv", sep="\t")
    cand = intersect_deg_hmrg(deg_f["gene"].tolist(), hmrg)
    write_table(cand, results_dir / "hmrg" / "candidate_genes_deg_intersection_hmrg.tsv")
    plot_venn2(
        set(deg_f["gene"].astype(str).str.upper()),
        set(h.upper() for h in hmrg),
        "DEG",
        "HMRG",
        fig_dir / "Figure2" / "Figure2A_venn.pdf",
        fig_dir / "Figure2" / "Figure2A_venn.png",
        settings.reporting.dpi,
        settings.reporting.export_pdf,
        settings.reporting.export_png,
    )
    libs = settings.enrichment.enrichr_libraries
    enr = run_enrichr_multilib(
        cand["gene"].tolist(),
        libs,
        settings.enrichment.enrichr_cutoff,
        settings.enrichment.use_enrichr_api,
    )
    for lib, df in enr.items():
        safe = lib.replace(" ", "_")
        write_table(df, results_dir / "enrichment" / f"enrichr_{safe}.tsv")
        plot_enrichment_bar(
            df,
            lib,
            15,
            fig_dir / "Figure2" / f"Figure2B_enrichment_{safe}.pdf",
            fig_dir / "Figure2" / f"Figure2B_enrichment_{safe}.png",
            settings.reporting.dpi,
            settings.reporting.export_pdf,
            settings.reporting.export_png,
        )


def run_survival_consensus(settings: PipelineSettings, project_root: Path, results_dir: Path, fig_dir: Path) -> None:
    expr_tr = _load_expr_matrix(settings, project_root, settings.cohorts.training.expression_key, None)
    expr_va = _load_expr_matrix(settings, project_root, settings.cohorts.validation.expression_key, None)
    clin_tr = _load_clinical(settings, project_root, settings.cohorts.training.expression_key)
    clin_va = _load_clinical(settings, project_root, settings.cohorts.validation.expression_key)
    clin_tr, tc, ec = _prepare_survival(clin_tr, settings)
    clin_va, _, _ = _prepare_survival(clin_va, settings)
    clin_tr = clin_tr.set_index("sample_id")
    clin_va = clin_va.set_index("sample_id")
    expr_tr = impute_median(expr_tr)
    expr_va = impute_median(expr_va)
    expr_tr, expr_va = subset_common_genes(expr_tr, expr_va)
    common_samples_tr = expr_tr.columns.intersection(clin_tr.index)
    common_samples_va = expr_va.columns.intersection(clin_va.index)
    expr_tr = expr_tr[common_samples_tr]
    expr_va = expr_va[common_samples_va]
    time_tr = clin_tr.loc[expr_tr.columns, tc].values.astype(float)
    event_tr = clin_tr.loc[expr_tr.columns, ec].values.astype(int)
    time_va = clin_va.loc[expr_va.columns, tc].values.astype(float)
    event_va = clin_va.loc[expr_va.columns, ec].values.astype(int)
    cand = pd.read_csv(results_dir / "hmrg" / "candidate_genes_deg_intersection_hmrg.tsv", sep="\t")
    genes_cand = cand["gene"].astype(str).tolist()
    uni = univariate_cox_screen(
        expr_tr,
        time_tr,
        event_tr,
        genes_cand,
        settings.survival.univariate_cox_p_threshold,
        settings.survival.max_univariate_genes,
    )
    write_table(uni, results_dir / "survival" / "univariate_cox_screen.tsv")
    uni_genes = uni["gene"].astype(str).tolist() if not uni.empty else []
    if not uni_genes:
        uni_genes = genes_cand[: min(50, len(genes_cand))]
    if settings.signature.prefer_fixed_for_tables:
        sig_genes = [g for g in settings.signature.fixed_genes if g in expr_tr.index]
        if len(sig_genes) < 2:
            sig_genes = uni_genes[: min(30, len(uni_genes))]
    else:
        sig_genes = uni_genes[: min(30, len(uni_genes))]
    X_tr = expr_tr.loc[sig_genes].T.astype(float)
    X_va = expr_va.loc[sig_genes].T.astype(float)
    rs = settings.consensus_models
    results = []
    if "rsf" in rs.enabled:
        results.append(
            train_rsf(
                X_tr,
                time_tr,
                event_tr,
                X_va,
                time_va,
                event_va,
                rs.rsf.n_estimators,
                rs.rsf.min_samples_leaf,
                rs.rsf.max_depth,
                settings.project.random_seed,
            )
        )
    if "ridge_cox" in rs.enabled:
        results.append(
            train_coxnet_variant(
                X_tr,
                time_tr,
                event_tr,
                X_va,
                time_va,
                event_va,
                "ridge_cox",
                0.0,
                rs.ridge_cox.alphas,
                settings.project.random_seed,
            )
        )
    if "elastic_net_cox" in rs.enabled:
        results.append(
            train_coxnet_variant(
                X_tr,
                time_tr,
                event_tr,
                X_va,
                time_va,
                event_va,
                "elastic_net_cox",
                rs.elastic_net_cox.l1_ratio,
                rs.elastic_net_cox.alphas,
                settings.project.random_seed,
            )
        )
    if "lasso_cox" in rs.enabled:
        results.append(
            train_coxnet_variant(
                X_tr,
                time_tr,
                event_tr,
                X_va,
                time_va,
                event_va,
                "lasso_cox",
                1.0,
                rs.lasso_cox.alphas,
                settings.project.random_seed,
            )
        )
    if "stepwise_cox" in rs.enabled:
        results.append(
            train_stepwise_cox(
                X_tr,
                time_tr,
                event_tr,
                X_va,
                time_va,
                event_va,
                rs.stepwise_cox.max_features,
                settings.project.random_seed,
            )
        )
    if "svm_rfe_cox" in rs.enabled:
        results.append(
            train_svm_rfe_cox(
                X_tr,
                time_tr,
                event_tr,
                X_va,
                time_va,
                event_va,
                rs.svm_rfe_cox.n_features_to_select,
                settings.project.random_seed,
            )
        )
    if "xgboost_survival" in rs.enabled:
        results.append(
            train_xgboost_survival(
                X_tr,
                time_tr,
                event_tr,
                X_va,
                time_va,
                event_va,
                rs.xgboost_survival.n_estimators,
                rs.xgboost_survival.max_depth,
                rs.xgboost_survival.learning_rate,
                settings.project.random_seed,
            )
        )
    if not results:
        raise RuntimeError("no survival models completed; check consensus_models.enabled and data")
    comp = results_to_comparison_table(results)
    write_table(comp, results_dir / "survival" / "model_comparison_cindex.tsv")
    plot_cindex_comparison(
        comp,
        fig_dir / "Figure3" / "Figure3A_model_cindex_comparison.pdf",
        fig_dir / "Figure3" / "Figure3A_model_cindex_comparison.png",
        settings.reporting.dpi,
        settings.reporting.export_pdf,
        settings.reporting.export_png,
    )
    best = select_best_model(results)
    pd.DataFrame([{"best_model": best.name, "train_cindex": best.train_cindex, "val_cindex": best.val_cindex}]).to_csv(
        results_dir / "survival" / "consensus_best_model.tsv",
        sep="\t",
        index=False,
    )
    fixed_genes = [g for g in settings.signature.fixed_genes if g in expr_tr.index]
    risk_tr, cph = cox_risk_from_training(expr_tr, time_tr, event_tr, fixed_genes)
    risk_va = pd.Series(
        cph.predict_partial_hazard(expr_va.loc[fixed_genes].T).values.flatten(),
        index=expr_va.columns,
        name="risk_score",
    )
    write_table(table2_multivariate_signature(expr_tr, time_tr, event_tr, fixed_genes), results_dir / "tables" / "Table2_multivariate_cox_signature.tsv")
    cutoff_tr = compute_cutoff(risk_tr, settings.risk_score.cutoff_method, settings.risk_score.cutoff_quantile)
    cutoff_va = compute_cutoff(risk_va, settings.risk_score.cutoff_method, settings.risk_score.cutoff_quantile)
    if not settings.risk_score.separate_cutoff_per_cohort:
        cutoff_va = cutoff_tr
    g_tr = stratify_risk(risk_tr, cutoff_tr)
    g_va = stratify_risk(risk_va, cutoff_va)
    plot_risk_distribution(
        risk_tr,
        risk_va,
        cutoff_tr,
        cutoff_va,
        fig_dir / "Figure4" / "Figure4A_risk_distribution.pdf",
        fig_dir / "Figure4" / "Figure4A_risk_distribution.png",
        settings.reporting.dpi,
        settings.reporting.export_pdf,
        settings.reporting.export_png,
    )
    plot_risk_survival_scatter(
        risk_tr,
        pd.Series(time_tr, index=risk_tr.index),
        pd.Series(event_tr, index=risk_tr.index),
        fig_dir / "Figure4" / "Figure4B_risk_vs_survival_train.pdf",
        fig_dir / "Figure4" / "Figure4B_risk_vs_survival_train.png",
        settings.reporting.dpi,
        settings.reporting.export_pdf,
        settings.reporting.export_png,
        "Training",
    )
    order = risk_tr.sort_values(ascending=False).index
    plot_signature_heatmap(
        expr_tr,
        fixed_genes,
        pd.Series(0, index=order),
        fig_dir / "Figure4" / "Figure4C_signature_heatmap_train.pdf",
        fig_dir / "Figure4" / "Figure4C_signature_heatmap_train.png",
        settings.reporting.dpi,
        settings.reporting.export_pdf,
        settings.reporting.export_png,
    )
    plot_km_two_group(
        pd.Series(time_tr, index=expr_tr.columns),
        pd.Series(event_tr, index=expr_tr.columns),
        g_tr,
        {"high": "high", "low": "low"},
        fig_dir / "Figure3" / "Figure3_KM_train.pdf",
        fig_dir / "Figure3" / "Figure3_KM_train.png",
        settings.reporting.dpi,
        settings.reporting.export_pdf,
        settings.reporting.export_png,
        "Kaplan-Meier training",
    )
    plot_km_two_group(
        pd.Series(time_va, index=expr_va.columns),
        pd.Series(event_va, index=expr_va.columns),
        g_va,
        {"high": "high", "low": "low"},
        fig_dir / "Figure3" / "Figure3_KM_validation.pdf",
        fig_dir / "Figure3" / "Figure3_KM_validation.png",
        settings.reporting.dpi,
        settings.reporting.export_pdf,
        settings.reporting.export_png,
        "Kaplan-Meier validation",
    )
    horizons = [12.0, 36.0, 60.0]
    plot_td_roc(
        time_tr,
        event_tr,
        risk_tr.values,
        horizons,
        fig_dir / "Figure4" / "Figure4D_time_dependent_ROC_train.pdf",
        fig_dir / "Figure4" / "Figure4D_time_dependent_ROC_train.png",
        settings.reporting.dpi,
        settings.reporting.export_pdf,
        settings.reporting.export_png,
    )
    plot_td_roc(
        time_va,
        event_va,
        risk_va.values,
        horizons,
        fig_dir / "Figure4" / "Figure4E_time_dependent_ROC_val.pdf",
        fig_dir / "Figure4" / "Figure4E_time_dependent_ROC_val.png",
        settings.reporting.dpi,
        settings.reporting.export_pdf,
        settings.reporting.export_png,
    )
    clin_tr_reset = clin_tr.reset_index()
    t1 = table1_baseline(clin_tr_reset, g_tr, None, None)
    write_table(t1, results_dir / "tables" / "Table1_baseline_summary.tsv")
    risk_tr.to_csv(results_dir / "survival" / "risk_scores_training.tsv", sep="\t")
    risk_va.to_csv(results_dir / "survival" / "risk_scores_validation.tsv", sep="\t")
    g_tr.to_csv(results_dir / "survival" / "risk_groups_training.tsv", sep="\t")
    g_va.to_csv(results_dir / "survival" / "risk_groups_validation.tsv", sep="\t")


def run_gsea_gsva_module(settings: PipelineSettings, project_root: Path, results_dir: Path, fig_dir: Path) -> None:
    gmt = settings.gsea_gsva.msigdb_c2_gmt
    if not gmt:
        LOG.warning("msigdb_c2_gmt not set; GSEA/GSVA skipped")
        return
    gmt_path = project_root / gmt
    expr = _load_expr_matrix(settings, project_root, settings.cohorts.training.expression_key, None)
    risk = pd.read_csv(results_dir / "survival" / "risk_scores_training.tsv", sep="\t", index_col=0).iloc[:, 0]
    g = stratify_risk(risk, compute_cutoff(risk, settings.risk_score.cutoff_method, settings.risk_score.cutoff_quantile))
    hi = g[g == "high"].index
    lo = g[g == "low"].index
    mean_hi = expr[hi].mean(axis=1)
    mean_lo = expr[lo].mean(axis=1)
    ranked = (mean_hi - mean_lo).sort_values(ascending=False)
    gsea_out = results_dir / "gsea" / "gsea_prerank"
    gsea_df = run_gsea_prerank(ranked, gmt_path, gsea_out, settings.gsea_gsva.gsea_permutations)
    write_table(gsea_df, results_dir / "gsea" / "gsea_results.tsv")
    plot_gsea_leading_edge(
        gsea_df,
        fig_dir / "Figure5" / "Figure5A_GSEA_NES.pdf",
        fig_dir / "Figure5" / "Figure5A_GSEA_NES.png",
        settings.reporting.dpi,
        settings.reporting.export_pdf,
        settings.reporting.export_png,
    )
    if settings.gsea_gsva.run_gsva_via_r:
        expr_path = results_dir / "gsea" / "expr_for_gsva.csv"
        expr.to_csv(expr_path)
        gsva_out = results_dir / "gsea" / "gsva_scores.csv"
        run_gsva_rscript(
            expr_path,
            gmt_path,
            gsva_out,
            project_root / settings.gsea_gsva.r_gsva_script,
            settings.gsea_gsva.gsva_kernel,
        )


def run_immune_module(settings: PipelineSettings, project_root: Path, results_dir: Path, fig_dir: Path) -> None:
    gmt = project_root / settings.immune.ssgsea_gene_sets_gmt
    expr = _load_expr_matrix(settings, project_root, settings.cohorts.training.expression_key, None)
    scores = run_ssgsea_immune(expr, gmt, results_dir / "immune" / "ssgsea")
    write_table(scores, results_dir / "immune" / "ssgsea_scores.tsv")
    risk_groups = pd.read_csv(results_dir / "survival" / "risk_groups_training.tsv", sep="\t", index_col=0).iloc[:, 0]
    cmp_df = compare_high_low(scores, risk_groups)
    write_table(cmp_df, results_dir / "immune" / "immune_high_vs_low.tsv")
    plot_immune_heatmap(
        scores,
        risk_groups,
        fig_dir / "Figure6" / "Figure6A_immune_ssgsea_heatmap.pdf",
        fig_dir / "Figure6" / "Figure6A_immune_ssgsea_heatmap.png",
        settings.reporting.dpi,
        settings.reporting.export_pdf,
        settings.reporting.export_png,
    )


def run_correlation_module(settings: PipelineSettings, project_root: Path, results_dir: Path, fig_dir: Path) -> None:
    from heme_pipeline.correlation_analysis.immune_gene import merge_immune_and_genes, plot_corr_heatmap, spearman_matrix

    scores = pd.read_csv(results_dir / "immune" / "ssgsea_scores.tsv", sep="\t", index_col=0)
    expr = _load_expr_matrix(settings, project_root, settings.cohorts.training.expression_key, None)
    merged = merge_immune_and_genes(scores, expr, settings.signature.fixed_genes)
    rho, pval = spearman_matrix(merged)
    rho.to_csv(results_dir / "immune" / "spearman_rho_matrix.tsv", sep="\t")
    pval.to_csv(results_dir / "immune" / "spearman_p_matrix.tsv", sep="\t")
    plot_corr_heatmap(
        rho,
        pval,
        fig_dir / "Figure6" / "Figure6B_immune_gene_correlation.pdf",
        fig_dir / "Figure6" / "Figure6B_immune_gene_correlation.png",
        settings.reporting.dpi,
        settings.reporting.export_pdf,
        settings.reporting.export_png,
    )


def run_mutation_module(settings: PipelineSettings, project_root: Path, results_dir: Path, fig_dir: Path) -> None:
    if not settings.mutation.maf_path:
        LOG.warning("mutation maf_path empty; mutation module skipped")
        return
    maf = load_maf(project_root / settings.mutation.maf_path)
    mat = mutation_matrix_from_maf(
        maf,
        settings.mutation.gene_column,
        settings.mutation.sample_column,
        settings.mutation.top_n_genes,
    )
    risk_groups = pd.read_csv(results_dir / "survival" / "risk_groups_training.tsv", sep="\t", index_col=0).iloc[:, 0]
    plot_mutation_landscape(
        mat,
        risk_groups,
        fig_dir / "Figure7" / "Figure7A_mutation_landscape.pdf",
        fig_dir / "Figure7" / "Figure7A_mutation_landscape.png",
        settings.reporting.dpi,
        settings.reporting.export_pdf,
        settings.reporting.export_png,
    )


def run_drug_module(settings: PipelineSettings, project_root: Path, results_dir: Path, fig_dir: Path) -> None:
    from heme_pipeline.drug_sensitivity.prrhothetic import compare_groups_ic50, estimate_ic50_prr, load_gdsc_pair

    if not settings.drug_sensitivity.gdsc_expression_reference:
        LOG.warning("GDSC paths empty; drug module skipped")
        return
    expr_ref, ic50 = load_gdsc_pair(
        project_root / settings.drug_sensitivity.gdsc_expression_reference,
        project_root / settings.drug_sensitivity.gdsc_ic50_matrix,
    )
    expr = _load_expr_matrix(settings, project_root, settings.cohorts.training.expression_key, None)
    drugs: list[str] = []
    if settings.drug_sensitivity.drug_list_path:
        drugs = Path(project_root / settings.drug_sensitivity.drug_list_path).read_text().splitlines()
        drugs = [d.strip() for d in drugs if d.strip()]
    else:
        drugs = ic50.columns.astype(str).tolist()[:50]
    pred = estimate_ic50_prr(expr, expr_ref, ic50, drugs)
    write_table(pred, results_dir / "drug" / "predicted_ic50.tsv")
    risk_groups = pd.read_csv(results_dir / "survival" / "risk_groups_training.tsv", sep="\t", index_col=0).iloc[:, 0]
    cmpd = compare_groups_ic50(pred, risk_groups)
    write_table(cmpd, results_dir / "drug" / "ic50_high_vs_low.tsv")


def run_tide_module(settings: PipelineSettings, project_root: Path, results_dir: Path, fig_dir: Path) -> None:
    if not settings.tide.tide_results_path:
        LOG.warning("TIDE path empty; tide module skipped")
        return
    tide = load_tide_table(project_root / settings.tide.tide_results_path)
    risk_groups = pd.read_csv(results_dir / "survival" / "risk_groups_training.tsv", sep="\t", index_col=0).iloc[:, 0]
    from heme_pipeline.tide_analysis.tide import align_tide_risk, plot_tide_scores

    aligned = align_tide_risk(tide, settings.tide.sample_id_column, risk_groups)
    cols = [c for c in aligned.columns if c not in {"risk_group"} and pd.api.types.is_numeric_dtype(aligned[c])]
    plot_tide_scores(
        aligned,
        cols[:5],
        fig_dir / "Figure7" / "Figure7B_TIDE_scores.pdf",
        fig_dir / "Figure7" / "Figure7B_TIDE_scores.png",
        settings.reporting.dpi,
        settings.reporting.export_pdf,
        settings.reporting.export_png,
    )


def run_expression_validation(settings: PipelineSettings, project_root: Path, results_dir: Path, fig_dir: Path) -> None:
    expr185 = _load_expr_matrix(settings, project_root, "gse18520", None)
    clin185 = _load_clinical(settings, project_root, "gse18520")
    clin185 = align_clinical_columns(
        clin185,
        settings.column_map.global_.sample_aliases,
        settings.cohorts.deg_reference.sample_id_column_clinical,
    )
    groups = clin185.set_index("sample_id")[settings.deg.group_column]
    plot_deg_heatmap(
        expr185,
        settings.expression_validation.gse18520_boxplot_genes,
        groups,
        fig_dir / "Figure8" / "Figure8A_GSE18520_expression_heatmap.pdf",
        fig_dir / "Figure8" / "Figure8A_GSE18520_expression_heatmap.png",
        settings.reporting.dpi,
        settings.reporting.export_pdf,
        settings.reporting.export_png,
    )
    plot_four_genes_by_binary_group(
        expr185,
        groups,
        settings.expression_validation.gse18520_boxplot_genes,
        (settings.deg.normal_label, settings.deg.tumor_label),
        fig_dir / "Figure8" / "Figure8A2_GSE18520_four_genes_boxplot.pdf",
        fig_dir / "Figure8" / "Figure8A2_GSE18520_four_genes_boxplot.png",
        settings.reporting.dpi,
        settings.reporting.export_pdf,
        settings.reporting.export_png,
        "GSE18520 four-gene expression",
    ).to_csv(results_dir / "expression_validation" / "GSE18520_four_genes_tests.tsv", sep="\t", index=False)
    expr_uc = _load_expr_matrix(settings, project_root, "ucsc_oc", None)
    genes_uc = [g for g in settings.expression_validation.ucsc_oc_boxplot_genes if g in expr_uc.index]
    if genes_uc:
        proxy = expr_uc.loc[genes_uc].astype(float).mean(axis=0)
        g_uc = stratify_risk(proxy, compute_cutoff(proxy, settings.risk_score.cutoff_method, settings.risk_score.cutoff_quantile))
        plot_four_genes_by_binary_group(
            expr_uc,
            g_uc,
            settings.expression_validation.ucsc_oc_boxplot_genes,
            ("low", "high"),
            fig_dir / "Figure8" / "Figure8B2_UCSC_OC_four_genes_boxplot.pdf",
            fig_dir / "Figure8" / "Figure8B2_UCSC_OC_four_genes_boxplot.png",
            settings.reporting.dpi,
            settings.reporting.export_pdf,
            settings.reporting.export_png,
            "UCSC-OC four-gene expression by median signature level",
        ).to_csv(results_dir / "expression_validation" / "UCSC_OC_four_genes_tests.tsv", sep="\t", index=False)
    plot_signature_heatmap(
        expr_uc,
        settings.expression_validation.ucsc_oc_boxplot_genes,
        pd.Series(0, index=expr_uc.columns),
        fig_dir / "Figure8" / "Figure8B_UCSC_OC_signature_heatmap.pdf",
        fig_dir / "Figure8" / "Figure8B_UCSC_OC_signature_heatmap.png",
        settings.reporting.dpi,
        settings.reporting.export_pdf,
        settings.reporting.export_png,
    )


def run_qpcr(settings: PipelineSettings, project_root: Path, results_dir: Path, fig_dir: Path) -> None:
    if not settings.qpcr.clinical_qpcr_path:
        LOG.warning("qPCR path empty; qPCR skipped")
        return
    from heme_pipeline.qpcr_plot.plots import load_qpcr, plot_qpcr_four_genes

    df = load_qpcr(project_root / settings.qpcr.clinical_qpcr_path, settings.qpcr.sheet_name)
    plot_qpcr_four_genes(
        df,
        settings.qpcr.group_column,
        settings.qpcr.gene_columns,
        fig_dir / "Figure8" / "Figure8C_RTqPCR_four_genes.pdf",
        fig_dir / "Figure8" / "Figure8C_RTqPCR_four_genes.png",
        settings.reporting.dpi,
        settings.reporting.export_pdf,
        settings.reporting.export_png,
        "RT-qPCR validation",
    )


def ensure_dirs(project_root: Path) -> None:
    for sub in [
        "deg",
        "hmrg",
        "enrichment",
        "survival",
        "gsea",
        "immune",
        "drug",
        "tables",
        "expression_validation",
    ]:
        (project_root / "results" / sub).mkdir(parents=True, exist_ok=True)
    for f in range(1, 9):
        (project_root / "figures" / f"Figure{f}").mkdir(parents=True, exist_ok=True)


MODULES = {
    "deg": run_deg_module,
    "hmrg_enrichment": run_hmrg_enrichment,
    "survival": run_survival_consensus,
    "gsea_gsva": run_gsea_gsva_module,
    "immune": run_immune_module,
    "correlation": run_correlation_module,
    "mutation": run_mutation_module,
    "drug": run_drug_module,
    "tide": run_tide_module,
    "expression": run_expression_validation,
    "qpcr": run_qpcr,
}


def run_pipeline(config_path: str | Path, only: list[str] | None = None) -> None:
    settings = load_settings(config_path)
    project_root = resolve_project_paths(settings, config_path)
    setup_logging(project_root / settings.paths.logs_dir, settings.logging.level, settings.logging.file_name)
    ensure_dirs(project_root)
    results_dir = project_root / settings.paths.results_dir
    fig_dir = project_root / settings.paths.figures_dir
    mods = only if only else list(MODULES.keys())
    for name in mods:
        if name not in MODULES:
            raise KeyError(name)
        LOG.info("running module %s", name)
        MODULES[name](settings, project_root, results_dir, fig_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="heme-pipeline")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        choices=sorted(MODULES.keys()),
    )
    return p
