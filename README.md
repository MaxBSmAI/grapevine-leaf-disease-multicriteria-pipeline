# Grapevine disease reproducible ML pipeline

Repositorio reproducible para evaluar ResNet-50, ViT-B/16, Swin-Tiny y
YOLOv8n-CLS en clasificación de enfermedades foliares de vid.

## Inicio rápido

1. Instale Python 3.12 y clone el repositorio.
2. Instale `environment.yml` o `requirements-lock.txt`.
3. Copie `configs/pipeline.example.yaml` a `configs/pipeline.local.yaml` y
   complete las rutas locales del dataset.
4. Ejecute `python scripts/run_pipeline.py --config configs/pipeline.local.yaml --output-dir results --stage status --dry-run`.
5. Consulte `docs/RUNBOOK.md` y `protocol/` antes de ejecutar etapas.

El dataset, checkpoints y resultados completos no se redistribuyen. Las
semillas, gates y bloqueos están definidos en `protocol/`. Consulte
`docs/MANUSCRIPT_MAPPING.md` para relacionar cada figura y tabla con su script.
