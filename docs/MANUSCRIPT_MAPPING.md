# Correspondencia manuscrito--código

| Elemento | Generador | Entrada principal |
|---|---|---|
| Distribución del dataset | `scripts/build_master_manifest.py` | manifiesto auditado |
| Rendimiento predictivo | `scripts/aggregate_results.py` | resultados confirmatorios |
| Matrices de confusión | `scripts/generate_gate12_confusion_matrices.py` | predicciones |
| Fidelidad XAI | `scripts/run_xai_faithfulness.py` | registros de atribución |
| Profiling computacional | `scripts/profile_model.py` | checkpoints |
| Radar multicriterio | `scripts/generate_gate12_radar_multicriteria.py` | tablas agregadas |

Los artefactos seleccionados se encuentran en `artifacts/`. Las versiones PNG y
PDF se mantienen juntas para facilitar el uso en manuscritos y Overleaf.

Los nombres de salida deben coincidir con `main.tex`. Cada cambio de resultados
requiere un nuevo tag del repositorio.
