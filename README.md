# Heme–Ovarian 预后与免疫多组学分析流程

本项目用于复现与扩展论文《A Multi-Machine Learning Consensus Agent Identifies Heme Metabolism-Related Prognostic Biomarkers in Ovarian Cancer with Experimental Validation》中的计算流程：基于公开队列 UCSC-OC、GSE18520、GSE26712 与血红素代谢相关基因（HMRGs）列表，完成差异表达、富集、多模型共识生存建模、风险评分、KM/ROC、临床关联、GSEA/GSVA、ssGSEA 免疫浸润与相关性、突变谱、药物敏感性（GDSC/pRRophetic 风格）、TIDE 整合、表达验证与 RT-qPCR 可视化等。

## 目录结构

```
.
├── pyproject.toml
├── README.md
├── .gitignore
├── configs/
│   ├── default.yaml
│   └── figure_mapping.yaml
├── data/                 # 原始与中间表达/临床等（由用户放置，不入库）
├── metadata/             # HMRG 列表、免疫基因集 GMT 等
├── results/              # 表格与中间统计结果
├── figures/              # Figure1–Figure8、PDF/PNG
├── logs/                 # 运行日志
├── r_scripts/
│   └── gsva.R            # GSVA（需 R + Bioconductor GSVA/GSEABase）
├── scripts/
│   └── run_pipeline.py   # 入口脚本
└── src/
    └── heme_pipeline/    # Python 包
        ├── cli.py
        ├── runner.py     # 总控编排
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

## 依赖管理（uv）

在项目根目录使用 [uv](https://github.com/astral-sh/uv) 安装与同步依赖（以下为说明，本仓库不自动执行）：

```bash
uv sync
```

可选安装 `rpy2` 组：

```bash
uv sync --extra rpy2
```

## 数据准备

1. 将 UCSC-OC、GSE18520、GSE26712 的表达矩阵与临床表整理为 CSV/TSV/TXT/Excel；在 `configs/default.yaml` 的 `datasets` 中填写各队列 `expression`、`clinical` 与 `format`。
2. 将 **283 个 HMRG** 基因符号写入 `metadata/hmrg_genes.txt`，每行一个基因符号（文件当前为空，需自行填入真实列表）。
3. MSigDB C2 GMT：下载后把路径填入 `gsea_gsva.msigdb_c2_gmt`。
4. 免疫 ssGSEA：默认使用 `metadata/immune_cell_28.gmt`，可按文献替换为自定义 GMT。
5. 突变：MAF 路径填入 `mutation.maf_path`；列名与 `gene_column`、`sample_column` 一致。
6. 药物敏感性：准备 GDSC 风格表达参考矩阵与 IC50 矩阵路径，见 `drug_sensitivity` 段。
7. TIDE：将 TIDE 网页或脚本输出表路径填入 `tide.tide_results_path`，并设置样本 ID 列名。
8. RT-qPCR：临床定量结果路径填入 `qpcr.clinical_qpcr_path`，并在 `qpcr.gene_columns` 中映射四基因列名。

## 配置

- 全局配置：`configs/default.yaml`（训练/验证队列键、`deg` 阈值、共识模型列表、四基因签名、风险 cutoff、输出图形 DPI 等）。
- 图号映射：`configs/figure_mapping.yaml`（与 Figure1–8、Table1–2 命名对应）。

## 运行方式

安装包后使用入口命令（需在项目根且已 `uv sync`）：

```bash
uv run heme-pipeline --config configs/default.yaml
```

仅运行部分模块：

```bash
uv run heme-pipeline --config configs/default.yaml --only deg survival immune
```

可用模块名：`deg`、`hmrg_enrichment`、`survival`、`gsea_gsva`、`immune`、`correlation`、`mutation`、`drug`、`tide`、`expression`、`qpcr`。

或使用脚本：

```bash
uv run python scripts/run_pipeline.py --config configs/default.yaml
```

## 主要输出（与图表对应）

| 输出 | 说明 |
|------|------|
| `results/deg/` | 全量与筛选 DEG 表 |
| `figures/Figure1/` | 火山图、DEG 热图 |
| `results/hmrg/` | 候选基因（DEG∩HMRG） |
| `figures/Figure2/` | Venn、GO/KEGG Enrichr 条形图 |
| `results/survival/` | 单因素 Cox、模型 C-index 对比、共识最优模型、训练/验证风险与分组 |
| `figures/Figure3/` | 训练/验证 KM |
| `figures/Figure4/` | 风险分布、风险-生存散点、签名热图、时间依赖 ROC |
| `results/tables/Table1`、`Table2` | 基线汇总、多因素 Cox（四基因） |
| `results/gsea/` | GSEA 表、GSVA 分数（R 成功时） |
| `figures/Figure5/` | GSEA NES 条形图 |
| `results/immune/` | ssGSEA 分数、组间比较、Spearman 矩阵 |
| `figures/Figure6/` | 免疫热图、相关性热图 |
| `figures/Figure7/` | 突变 landscape、TIDE 箱线图 |
| `results/drug/` | 预测 IC50、组间比较 |
| `figures/Figure8/` | GSE18520/UCSC 表达热图、RT-qPCR 四基因图 |
| `logs/pipeline.log` | 全流程日志 |

## R 依赖（GSVA）

若启用 `gsea_gsva.run_gsva_via_r`，需本机安装 R 与 Bioconductor 包 `GSVA`、`GSEABase`，并保证 `Rscript` 在 `PATH` 中。`r_scripts/gsva.R` 由 Python 子进程调用。

## 说明

- 差异表达：GSE18520 肿瘤 vs 正常，Welch t 检验 + BH，阈值 `padj < 0.05` 且 `|log2FC| > 0.5`。
- 共识模型：在训练集拟合、验证集评估 C-index，加权综合得分选最优；最终论文四基因签名表型以多因素 Cox 在训练集上估计系数与 HR（`Table2`）。
- `xgboost_survival` 名称在配置中保留；实现采用 `scikit-survival` 的 `GradientBoostingSurvivalAnalysis`（与梯度提升生存一致，避免 XGBoost Cox API 版本差异）。

## 许可证

研究用途请遵循各数据来源与工具许可（GEO、UCSC、MSigDB、Enrichr、GDSC 等）。
