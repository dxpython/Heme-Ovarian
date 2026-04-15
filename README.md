# Heme–Ovarian Prognostic and Immune Multi-Omics Analysis Pipeline

This project reproduces and extends the computational workflow from the paper *A Multi-Machine Learning Consensus Agent Identifies Heme Metabolism-Related Prognostic Biomarkers in Ovarian Cancer with Experimental Validation*. Based on public cohorts (UCSC-OC, GSE18520, and GSE26712) and a heme metabolism-related gene (HMRG) list, it performs differential expression, enrichment, multi-model consensus survival modeling, risk scoring, KM/ROC analysis, clinical association, GSEA/GSVA, ssGSEA immune infiltration and correlation, mutation profiling, drug sensitivity prediction (GDSC/pRRophetic style), TIDE integration, expression validation, and RT-qPCR visualization.

## Project Structure

```
.
├── pyproject.toml
├── README.md
├── .gitignore
├── configs/
│   ├── default.yaml
│   └── figure_mapping.yaml
├── data/                 # Raw and intermediate expression/clinical data (user-provided, not committed)
├── metadata/             # HMRG list, immune gene-set GMT files, etc.
├── results/              # Tables and intermediate statistical outputs
├── figures/              # Figure1–Figure8, PDF/PNG
├── logs/                 # Run logs
├── r_scripts/
│   └── gsva.R            # GSVA (requires R + Bioconductor GSVA/GSEABase)
├── scripts/
│   └── run_pipeline.py   # Entry script
└── src/
    └── heme_pipeline/    # Python package
        ├── cli.py
        ├── runner.py     # Main orchestrator
        ├── config.py
        ├── data_ingestion/
        ├── preprocess/
        ├── deg/
        ├── hmrg_intersection/
        ├── enrichment/
        ├── survival_models/
        ├── consensus_agent/
        ├── risk_model/
        ├── km_roc/
        ├── clinical_assoc/
        ├── gsea_gsva/
        ├── immune_ssgsea/
        ├── correlation_analysis/
        ├── mutation_analysis/
        ├── drug_sensitivity/
        ├── tide_analysis/
        ├── qpcr_plot/
        └── reporting/
```

## Dependency Management (`uv`)

Use [uv](https://github.com/astral-sh/uv) in the project root to install and sync dependencies (shown for documentation; not auto-executed by this repository):

```bash
uv sync
```

Optionally install the `rpy2` extras group:

```bash
uv sync --extra rpy2
```

## Data Preparation

1. Organize expression matrices and clinical tables for UCSC-OC, GSE18520, and GSE26712 as CSV/TSV/TXT/Excel files; fill in each cohort's `expression`, `clinical`, and `format` fields under `datasets` in `configs/default.yaml`.
2. Write the symbols of **283 HMRGs** to `metadata/hmrg_genes.txt`, one symbol per line (the file is currently empty and must be filled with the real list).
3. MSigDB C2 GMT: download and set its path in `gsea_gsva.msigdb_c2_gmt`.
4. Immune ssGSEA: `metadata/immune_cell_28.gmt` is used by default; you can replace it with a custom GMT based on literature.
5. Mutation data: set the MAF path in `mutation.maf_path`; ensure column names match `gene_column` and `sample_column`.
6. Drug sensitivity: prepare GDSC-style expression reference and IC50 matrix paths; see the `drug_sensitivity` section.
7. TIDE: set the output table path from the TIDE web tool or script in `tide.tide_results_path`, and configure the sample ID column name.
8. RT-qPCR: set the clinical quantification result path in `qpcr.clinical_qpcr_path`, and map the four-gene column names in `qpcr.gene_columns`.

## Configuration

- Global configuration: `configs/default.yaml` (training/validation cohort keys, `deg` thresholds, consensus model list, four-gene signature, risk cutoff, output figure DPI, etc.).
- Figure mapping: `configs/figure_mapping.yaml` (naming alignment with Figure1–8 and Table1–2).

## Running the Pipeline

After dependency setup, run the entry command (from project root after `uv sync`):

```bash
uv run heme-pipeline --config configs/default.yaml
```

Run only selected modules:

```bash
uv run heme-pipeline --config configs/default.yaml --only deg survival immune
```

Available module names: `deg`, `hmrg_enrichment`, `survival`, `gsea_gsva`, `immune`, `correlation`, `mutation`, `drug`, `tide`, `expression`, `qpcr`.

Or use the script directly:

```bash
uv run python scripts/run_pipeline.py --config configs/default.yaml
```

## Main Outputs (Mapped to Figures/Tables)

| Output | Description |
|------|------|
| `results/deg/` | Full and filtered DEG tables |
| `figures/Figure1/` | Volcano plot and DEG heatmap |
| `results/hmrg/` | Candidate genes (DEG ∩ HMRG) |
| `figures/Figure2/` | Venn diagram, GO/KEGG Enrichr bar plots |
| `results/survival/` | Univariate Cox, model C-index comparison, best consensus model, training/validation risks and groups |
| `figures/Figure3/` | Training/validation KM curves |
| `figures/Figure4/` | Risk distribution, risk-survival scatter, signature heatmap, time-dependent ROC |
| `results/tables/Table1`, `Table2` | Baseline summary and multivariate Cox (four genes) |
| `results/gsea/` | GSEA tables and GSVA scores (when R execution succeeds) |
| `figures/Figure5/` | GSEA NES bar plots |
| `results/immune/` | ssGSEA scores, between-group comparisons, Spearman matrix |
| `figures/Figure6/` | Immune heatmap and correlation heatmap |
| `figures/Figure7/` | Mutation landscape and TIDE boxplots |
| `results/drug/` | Predicted IC50 and between-group comparisons |
| `figures/Figure8/` | GSE18520/UCSC expression heatmaps and RT-qPCR four-gene plots |
| `logs/pipeline.log` | Full pipeline log |

## R Dependencies (GSVA)

If `gsea_gsva.run_gsva_via_r` is enabled, R and Bioconductor packages `GSVA` and `GSEABase` must be installed locally, and `Rscript` must be available in `PATH`. `r_scripts/gsva.R` is invoked by a Python subprocess.

## Notes

- Differential expression: GSE18520 tumor vs normal, Welch's t-test + BH correction, with thresholds `padj < 0.05` and `|log2FC| > 0.5`.
- Consensus modeling: fit on training data and evaluate C-index on validation data; select the best model by weighted composite score. The final four-gene signature phenotype in the paper uses multivariate Cox on the training set to estimate coefficients and HRs (`Table2`).
- The `xgboost_survival` name is retained in config; implementation uses `GradientBoostingSurvivalAnalysis` from `scikit-survival` (conceptually aligned with gradient-boosted survival, while avoiding XGBoost Cox API version differences).

## License

For research usage, follow the licenses of all data sources and tools (GEO, UCSC, MSigDB, Enrichr, GDSC, etc.).
