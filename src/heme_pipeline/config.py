from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProjectConfig(BaseModel):
    name: str = "heme_ovarian_pipeline"
    random_seed: int = 42


class PathsConfig(BaseModel):
    base_dir: str = "."
    data_dir: str = "data"
    metadata_dir: str = "metadata"
    results_dir: str = "results"
    figures_dir: str = "figures"
    logs_dir: str = "logs"


class DatasetEntry(BaseModel):
    expression: str = ""
    clinical: str = ""
    format: str = "tsv"


class DatasetsConfig(BaseModel):
    ucsc_oc: DatasetEntry = Field(default_factory=DatasetEntry)
    gse18520: DatasetEntry = Field(default_factory=DatasetEntry)
    gse26712: DatasetEntry = Field(default_factory=DatasetEntry)


class CohortRef(BaseModel):
    name: str
    expression_key: str
    sample_id_column_clinical: str = "sample_id"


class CohortsConfig(BaseModel):
    training: CohortRef
    validation: CohortRef
    deg_reference: CohortRef


class ColumnMapGlobal(BaseModel):
    sample_aliases: list[str] = Field(default_factory=list)
    gene_aliases: list[str] = Field(default_factory=list)
    survival_time_aliases: list[str] = Field(default_factory=list)
    survival_event_aliases: list[str] = Field(default_factory=list)


class ColumnMapConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    global_: ColumnMapGlobal = Field(alias="global", default_factory=ColumnMapGlobal)


class DegConfig(BaseModel):
    cohort: str = "gse18520"
    group_column: str = "group"
    tumor_label: str = "tumor"
    normal_label: str = "normal"
    padj_threshold: float = 0.05
    log2fc_threshold: float = 0.5
    method: str = "welch_ttest_bh"


class HmrgConfig(BaseModel):
    gene_list_path: str = "metadata/hmrg_genes.txt"
    expected_count: int = 283


class EnrichmentConfig(BaseModel):
    organism: str = "human"
    go_bp: bool = True
    go_cc: bool = True
    go_mf: bool = True
    kegg: bool = True
    enrichr_cutoff: float = 0.05
    use_enrichr_api: bool = True
    enrichr_libraries: list[str] = Field(default_factory=list)


class SurvivalConfig(BaseModel):
    time_col: str = "OS_time"
    event_col: str = "OS_event"
    duration_unit: str = "months"
    univariate_cox_p_threshold: float = 0.05
    max_univariate_genes: int = 200


class ModelParams(BaseModel):
    n_estimators: int = 500
    min_samples_leaf: int = 5
    max_depth: int | None = None


class RidgeCoxParams(BaseModel):
    alphas: list[float] = Field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])


class ElasticNetCoxParams(BaseModel):
    l1_ratio: float = 0.5
    alphas: list[float] = Field(default_factory=lambda: [0.01, 0.1, 1.0])


class LassoCoxParams(BaseModel):
    alphas: list[float] = Field(default_factory=lambda: [0.001, 0.01, 0.1, 1.0])


class StepwiseCoxParams(BaseModel):
    max_features: int = 30
    direction: str = "forward"


class SvmRfeCoxParams(BaseModel):
    n_features_to_select: int = 15


class XgbSurvParams(BaseModel):
    n_estimators: int = 200
    max_depth: int = 3
    learning_rate: float = 0.05


class ConsensusModelsConfig(BaseModel):
    enabled: list[str] = Field(default_factory=list)
    rsf: ModelParams = Field(default_factory=ModelParams)
    ridge_cox: RidgeCoxParams = Field(default_factory=RidgeCoxParams)
    elastic_net_cox: ElasticNetCoxParams = Field(default_factory=ElasticNetCoxParams)
    lasso_cox: LassoCoxParams = Field(default_factory=LassoCoxParams)
    stepwise_cox: StepwiseCoxParams = Field(default_factory=StepwiseCoxParams)
    svm_rfe_cox: SvmRfeCoxParams = Field(default_factory=SvmRfeCoxParams)
    xgboost_survival: XgbSurvParams = Field(default_factory=XgbSurvParams)


class SignatureConfig(BaseModel):
    fixed_genes: list[str] = Field(default_factory=list)
    prefer_fixed_for_tables: bool = True


class RiskScoreConfig(BaseModel):
    cutoff_method: str = "median"
    cutoff_quantile: float | None = None
    separate_cutoff_per_cohort: bool = True


class GseaGsvaConfig(BaseModel):
    msigdb_c2_gmt: str = ""
    gene_sets: str = "C2"
    gsea_permutations: int = 1000
    gsva_kernel: str = "gaussian"
    run_gsva_via_r: bool = True
    r_gsva_script: str = "r_scripts/gsva.R"


class ImmuneConfig(BaseModel):
    ssgsea_gene_sets_gmt: str = "metadata/immune_cell_28.gmt"
    immune_cell_labels: list[str] = Field(default_factory=list)


class MutationConfig(BaseModel):
    maf_path: str = ""
    matrix_path: str = ""
    gene_column: str = "Hugo_Symbol"
    sample_column: str = "Tumor_Sample_Barcode"
    top_n_genes: int = 20


class DrugSensitivityConfig(BaseModel):
    gdsc_expression_reference: str = ""
    gdsc_ic50_matrix: str = ""
    drug_list_path: str = ""
    method: str = "ridge"


class TideConfig(BaseModel):
    tide_results_path: str = ""
    sample_id_column: str = "sample"


class QpcrConfig(BaseModel):
    clinical_qpcr_path: str = ""
    sheet_name: str | int = 0
    group_column: str = "group"
    gene_columns: dict[str, str] = Field(default_factory=dict)


class ExpressionValidationConfig(BaseModel):
    gse18520_boxplot_genes: list[str] = Field(default_factory=list)
    ucsc_oc_boxplot_genes: list[str] = Field(default_factory=list)


class ReportingConfig(BaseModel):
    figure_prefix: str = "Figure"
    table_prefix: str = "Table"
    export_pdf: bool = True
    export_png: bool = True
    dpi: int = 300
    figure_size_inches: list[float] = Field(default_factory=lambda: [7.0, 5.5])


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file_name: str = "pipeline.log"


class PipelineSettings(BaseSettings):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    datasets: DatasetsConfig = Field(default_factory=DatasetsConfig)
    cohorts: CohortsConfig
    column_map: ColumnMapConfig
    deg: DegConfig = Field(default_factory=DegConfig)
    hmrg: HmrgConfig = Field(default_factory=HmrgConfig)
    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)
    survival: SurvivalConfig = Field(default_factory=SurvivalConfig)
    consensus_models: ConsensusModelsConfig = Field(default_factory=ConsensusModelsConfig)
    signature: SignatureConfig = Field(default_factory=SignatureConfig)
    risk_score: RiskScoreConfig = Field(default_factory=RiskScoreConfig)
    gsea_gsva: GseaGsvaConfig = Field(default_factory=GseaGsvaConfig)
    immune: ImmuneConfig = Field(default_factory=ImmuneConfig)
    mutation: MutationConfig = Field(default_factory=MutationConfig)
    drug_sensitivity: DrugSensitivityConfig = Field(default_factory=DrugSensitivityConfig)
    tide: TideConfig = Field(default_factory=TideConfig)
    qpcr: QpcrConfig = Field(default_factory=QpcrConfig)
    expression_validation: ExpressionValidationConfig = Field(
        default_factory=ExpressionValidationConfig
    )
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = SettingsConfigDict(extra="ignore", env_nested_delimiter="__")


def load_settings(config_path: str | Path) -> PipelineSettings:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "column_map" in raw and "global" in raw["column_map"]:
        cm = raw["column_map"]
        raw["column_map"] = {"global": cm["global"]}
    return PipelineSettings.model_validate(raw)


def resolve_project_paths(settings: PipelineSettings, config_path: str | Path) -> Path:
    base = Path(settings.paths.base_dir).expanduser()
    if not base.is_absolute():
        base = Path(config_path).resolve().parent.parent / base
    return base.resolve()
