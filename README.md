# PLM Phylogenetic Weighting

Проект посвящён исследованию того, можно ли использовать protein language models для приближённого восстановления филогенетических весов белковых последовательностей.

Основная идея работы:

```text
protein sequence -> PLM embedding -> phylogenetic weight
```

В проекте исследуется, насколько филогенетическая информация сохраняется в embedding space современных protein language models и можно ли использовать эту информацию для восстановления различных вариантов phylogenetic weighting без явного построения дерева.

## Что исследуется

Работа фокусируется на нескольких вопросах:

- содержат ли PLM embeddings филогенетический сигнал;
- насколько embedding distance связан с tree distance;
- можно ли восстанавливать phylogenetic weights по embeddings;
- какие локальные свойства embedding space наиболее информативны;
- как качество зависит от размера семейства;
- насколько ranking-based подходы лучше regression.

## Используемые модели

### Protein language models

- ESM2 8M
- ESM2 35M

### Phylogenetic weighting targets

- branch sharing
- mean tree distance
- terminal branch

## Используемые подходы

### Unsupervised methods

Используются density-based методы на embedding space:
- mean distance;
- kNN distance;
- kernel density;
- graph density.

### Supervised methods

Используются:
- regression;
- pairwise ranking;
- gradient boosting rankers;
- feature ablation.

## Данные

В качестве источника белковых семейств используется база Pandit.

Конфигурация по умолчанию:
- 300 семейств;
- средний размер семейства около 500 последовательностей;
- разделение train/test выполняется по семействам, а не по отдельным последовательностям.

## Структура репозитория

```text
configs/
    конфигурации экспериментов

scripts/
    запуск отдельных этапов pipeline

src/plm_phylo_weighting/

    data/
        загрузка и парсинг Pandit

    weights/
        методы взвешивания

    plm/
        работа с ESM2 и embeddings

    models/
        regression и ranking модели

    experiments/
        экспериментальные блоки

    evaluation/
        метрики и корреляции

    plots/
        построение графиков

    summaries/
        итоговые таблицы

data/
    локальные данные

results/
    результаты экспериментов

models/
    сохранённые модели
```

## Основные экспериментальные блоки

### Phylogenetic signal

Исследуется связь:
- embedding distance vs tree distance;
- local PLM features vs phylogenetic weights.

### Supervised prediction

Исследуется восстановление phylogenetic weights с помощью:
- regression;
- pairwise ranking;
- boosting rankers.

### Feature ablation

Проверяется вклад:
- PCA features;
- density features;
- rank features.

### Size analysis

Исследуется влияние размера семейства на качество модели.

## Установка

```bash
pip install -r requirements.txt
pip install -e .
```

## Полный запуск

```bash
python scripts/run_full_pipeline.py --config configs/default.yaml
```

## Основные выходные файлы

```text
selected_families.csv
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

Большинство существующих работ по теме PLM и филогении исследуют:
- reconstruction of phylogenetic trees;
- evolutionary distances;
- homology detection.

В отличие от них, данный проект исследует задачу восстановления phylogenetic weighting по embedding space protein language models.

Основной акцент делается на:
- локальной геометрии embedding space;
- density-based признаках;
- pairwise ranking;
- переносимости между различными белковыми семействами.
