# Guía de ejecución

Ejecute los comandos desde la raíz del repositorio utilizando `.\.venv\Scripts\python.exe`.
Ningún comando aprueba automáticamente una puerta de control científico (Gate).

## 1. Entorno y estado

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_environment.ps1 -Backend cuda128
.\.venv\Scripts\python.exe scripts/run_pipeline.py --config configs/pipeline.local.yaml --output-dir results --stage status --dry-run
```

## 2. Completar Gate 2 y aprobar Gate 3

Complete todas las filas de `data/similarity/review_decisions_template.csv` y, a continuación, ejecute:

```powershell
.\.venv\Scripts\python.exe scripts/consolidate_confirmed_groups.py --config configs/pipeline.local.yaml --output-dir data/similarity
```

Después de la revisión humana, registre Gate 3 en `protocol/GATE_APPROVALS.json` y cambie
`configs/data/split.json` a `approval_status: approved_gate_3`.

## 3. Partición, normalización exclusiva del entrenamiento y verificaciones de implementación

```powershell
.\.venv\Scripts\python.exe scripts/create_canonical_split.py --config configs/pipeline.local.yaml --output-dir data/splits
# Registre Gate 4 después de revisar todas las auditorías de la partición.
.\.venv\Scripts\python.exe scripts/calculate_train_normalisation.py --config configs/pipeline.local.yaml --output-dir data/metadata
.\.venv\Scripts\python.exe scripts/run_smoke_test.py --config configs/pipeline.local.yaml --output-dir results/smoke
.\.venv\Scripts\python.exe scripts/run_batch16_preflight.py --config configs/pipeline.local.yaml --output-dir results/preflight
```

Revise los informes y, posteriormente, registre explícitamente Gate 5 y Gate 6. El preflight
realiza un paso AdamW FP32 con el batch físico configurado para cada arquitectura, sin leer el
dataset ni acceder a validación o prueba.

## 4. Selección utilizando exclusivamente validación

```powershell
.\.venv\Scripts\python.exe scripts/run_pipeline.py --config configs/pipeline.local.yaml --output-dir results --stage tuning
```

El tuning es reanudable: cada corrida guarda un checkpoint de progreso por época; Optuna utiliza
una base SQLite persistente por arquitectura; y las corridas y candidatos focal completados se
reutilizan después de una interrupción. Si Git no está disponible, cada artefacto registra un hash
SHA-256 determinista del árbol de código, scripts, configuración y protocolo.

Revise todos los ensayos, registre Gate 7, establezca la semilla aprobada para la muestra XAI,
confirme la versión de código registrada y congele el experimento:

```powershell
.\.venv\Scripts\python.exe scripts/freeze_experiment.py --config configs/pipeline.local.yaml --output-dir protocol/frozen
.\.venv\Scripts\python.exe scripts/run_pipeline.py --config configs/pipeline.local.yaml --output-dir results --stage functional
```

La matriz funcional está expresamente excluida de la publicación y no utiliza datos de prueba.
Registre Gate 8 únicamente si las 12 combinaciones se ejecutan correctamente.

## 5. Autorización explícita del conjunto de prueba y ejecución confirmatoria

Después de registrar Gate 9, desbloquee el acceso al conjunto de prueba mediante un comando atribuible:

```powershell
.\.venv\Scripts\python.exe scripts/authorise_test_access.py --config configs/pipeline.local.yaml --output-dir protocol --authorised-by "NAME" --authorisation-date "YYYY-MM-DD" --confirm-test-access
.\.venv\Scripts\python.exe scripts/run_pipeline.py --config configs/pipeline.local.yaml --output-dir results --stage confirmatory
```

El comando confirmatorio puede reanudarse: no se sobrescriben los directorios de ejecuciones
completadas ni los archivos existentes de predicciones sobre el conjunto de prueba.

## 6. XAI, perfilado, estadística y publicación

Cree la muestra XAI congelada e independiente del modelo. Después, ejecute los scripts de XAI y
perfilado una vez por cada checkpoint seleccionado. Aplique las pruebas estadísticas pareadas a
los resultados confirmatorios completos. Ejecute `generate_publication_artifacts.py` únicamente
después de revisar y aprobar Gate 10. Las opciones exactas están disponibles mediante `--help`;
el inventario de comandos se encuentra en `scripts/README.md`.
