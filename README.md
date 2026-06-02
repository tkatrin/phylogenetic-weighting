# PLM Phylogenetic Weighting

Проект посвящён исследованию того, можно ли использовать protein language models для приближённого восстановления филогенетических весов белковых последовательностей.

Основная идея:

```text
protein sequence -> PLM embedding -> phylogenetic weight
```

В работе используются:
- phylogenetic weighting methods;
- protein language models;
- unsupervised density methods;
- supervised regression;
- pairwise ranking;
- feature ablation;
- анализ зависимости качества от размера семейства.

## Основные задачи проекта

Проект исследует несколько вопросов:

1. Содержат ли PLM embeddings филогенетический сигнал.
2. Насколько embedding distance связан с tree distance.
3. Можно ли восстановить phylogenetic weights по embeddings.
4. Какие локальные признаки embedding space наиболее информативны.
5. Как качество зависит от размера семейства.
6. Насколько pairwise ranking лучше regression.

## Используемые модели

### Protein language models

- ESM2 8M
- ESM2 35M

### Phylogenetic targets

- branch sharing
- mean tree distance
- terminal branch

## Структура репозитория

```text
configs/
    конфигурации экспериментов

scripts/
    точки запуска отдельных этапов pipeline

src/plm_phylo_weighting/

    data/
        загрузка и парсинг Pandit
        выбор семейств

    weights/
        phylogenetic weighting
        classical weighting
        PLM density weighting

    plm/
        загрузка ESM2
        embedding cache

    models/
        feature engineering
        regression models
        ranking models

    experiments/
        отдельные этапы экспериментов

    evaluation/
        метрики и корреляции

    summaries/
        итоговые таблицы

    plots/
        построение графиков

data/
    локальные данные

results/
    результаты экспериментов

models/
    сохранённые модели
```

## Новые блоки анализа

В текущей версии проекта добавлены:

- embedding distance vs tree distance;
- local PLM features vs phylogenetic weights;
- feature ablation;
- size analysis;
- сравнение разных phylogenetic targets;
- сравнение нескольких ESM2 моделей.

## Конфигурация по умолчанию

```yaml
n_families: 300
target_mean_ano: 500
```

Используется разделение train/test по семействам, а не по отдельным последовательностям.

## Установка

```bash
pip install -r requirements.txt
pip install -e .
```

## Полный запуск

```bash
python scripts/run_full_pipeline.py --config configs/default.yaml
```

## Запуск отдельных этапов

```bash
python scripts/run_prepare_data.py --config configs/default.yaml

python scripts/run_compute_embeddings.py --config configs/default.yaml

python scripts/run_unsupervised.py --config configs/default.yaml

python scripts/run_phylo_signal.py --config configs/default.yaml

python scripts/run_supervised.py --config configs/default.yaml

python scripts/run_local_feature_signal.py --config configs/default.yaml

python scripts/run_feature_ablation.py --config configs/default.yaml

python scripts/run_rankers.py --config configs/default.yaml

python scripts/run_size_analysis.py --config configs/default.yaml

python scripts/run_make_summary.py --config configs/default.yaml
```

## Основные выходные файлы

```text
selected_families.csv
family_checks.csv
phylo_weights.csv

embedding_tree_distance_correlation_summary.csv

density_features_vs_phylo_weights_summary.csv

unsupervised_summary.csv
supervised_summary.csv
pairwise_summary.csv
ranker_summary.csv

feature_ablation_summary.csv

size_analysis_summary.csv
size_correlation_summary.csv

final_summary.csv
```

## Идея работы

В отличие от большинства работ по теме PLM и филогении, проект исследует не восстановление дерева напрямую, а задачу восстановления phylogenetic weighting.

Особый акцент сделан на:
- локальной структуре embedding space;
- density-based признаках;
- pairwise ranking;
- переносимости между семействами.