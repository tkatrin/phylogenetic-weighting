# Структура проекта

Репозиторий организован как Python package, а не как набор ноутбуков.

## Основные директории

### configs/

Конфигурации экспериментов:
- число семейств;
- используемые PLM;
- параметры density methods;
- train/test split.

### scripts/

CLI-скрипты для запуска этапов pipeline.

### src/plm_phylo_weighting/

Основной код проекта.

#### data/

Работа с Pandit:
- загрузка;
- парсинг;
- выбор семейств;
- проверки данных.

#### weights/

Методы взвешивания:
- phylogenetic weighting;
- clustering weighting;
- Henikoff weighting;
- PLM density weighting.

#### plm/

Работа с protein language models:
- загрузка ESM2;
- вычисление embeddings;
- caching.

#### models/

ML-модели:
- regression;
- pairwise ranking;
- rankers;
- feature engineering.

#### experiments/

Отдельные экспериментальные блоки.

Основные модули:
- prepare_data.py
- unsupervised.py
- supervised.py
- rankers.py
- phylo_signal.py
- local_feature_signal.py
- feature_ablation.py
- size_analysis.py

#### evaluation/

Корреляции и метрики качества.

#### plots/

Построение графиков и визуализаций.

#### summaries/

Итоговые таблицы и агрегация результатов.

## Основные новые эксперименты

### phylo_signal.py

Проверяет:
- embedding distance vs tree distance;
- наличие филогенетического сигнала в embeddings.

### local_feature_signal.py

Проверяет:
- связь локальных density features с phylogenetic weights.

### feature_ablation.py

Проверяет:
- какие признаки действительно важны.

### size_analysis.py

Проверяет:
- как качество зависит от размера семейства.

## Рекомендуемый порядок запуска

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

## Что не нужно загружать в git

Не рекомендуется загружать:
- raw data;
- embeddings;
- model checkpoints;
- большие csv результатов;
- кэшированные модели.