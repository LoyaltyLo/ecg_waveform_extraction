# Legacy / 已归档脚本

本目录存放被主线取代的历史脚本，仅作存档，不再维护。
归档于 2026-08-25，均无存活代码引用。

| 文件 | 原作用 | 被什么取代 |
|---|---|---|
| `parse_aecg.py` | 最早的 ElementTree 版 aECG 解析器原型（含未完成逻辑） | `utils/aecg_parser.py` |
| `qrs_polarity.py` | Lead I/II 逐拍 QRS 极性 v1（自写阈值规则） | `extraction/qrs_refiner.compute_qrs_polarity_v2`，由 `limb_lead_processor` 内部调用 |
| `batch_qrs_polarity_all.py` | 2 导联手动搭 HSMM 流水线统计极性（硬编码路径） | `limb_lead_processor.LimbLeadProcessor` + `batch_limb_leads.py`（6 导联） |
| `detect_polarity_v2.py` | 记录级 5 方法加权投票 RA-LA 极性检测（读 7 月 output/rala_full 产物） | `limb_lead_reversal.LimbLeadReversalDetector` |
| `export_polarity_v2_xlsx.py` | 上者的 9-sheet Excel 报表 | `export_reversal_xlsx.py` |
| `check_integration.py` | 合成数据 9 步流水线冒烟自检 | `tests/test_pipeline.py` |
| `check_stage2.py` | Stage 2（P 波提取/分析）集成自检 | `tests/test_pipeline.py` |
| `check_real_ecg.py` | MIT-BIH 100 号单记录端到端验证 | `batch_process.py`（批量版） |
| `save_p_wave_test_json.py` | P 波优化验收：50 条记录指标存 JSON（对应 docs/P_WAVE_OPTIMIZATION_PLAN.md） | 优化已落地 `extraction/p_wave_extractor.py`，验收完成 |
| `generate_optimized_plots.py` | P 波优化验证图（仅 Lead II） | 同上，验收完成 |
| `check_optimized_p_wave.py` | P 波优化指标 + 与机测标注 MAE 对比 | 同上，验收完成 |
| `detect_polarity.py` *(更早归档)* | 旧版 4 方法 RA-LA 检测 | `limb_lead_reversal.py` |
| `download_mitbih.py` *(更早归档)* | 下载 MIT-BIH 全库 | `download_all.py`（注意本文件有 `Path` 未导入 bug，运行会 NameError） |
| `export_polarity_xlsx.py` *(更早归档)* | 旧版极性报表（7 sheet） | `export_reversal_xlsx.py` |

注意：本目录不是 Python 包（无 `__init__.py`），脚本不能通过 `python -m ecg_waveform_extraction.legacy.xxx` 运行；
且部分脚本含机器特定绝对路径或已知 bug，仅作参考。

当前主线见仓库根：`limb_lead_processor.py` / `limb_lead_reversal.py` / `chest_lead_analyzer.py` +
`batch_limb_leads.py` / `batch_ra_la_simple.py` / `batch_reversal_detect.py` / `export_reversal_xlsx.py`。
