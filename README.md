# LAIM Trace-to-Basket Converter

Преобразование AEF-телеметрии (формат ТЗ "Сбор данных для системы распознавания
аномалий в МАС", v1.6) в формат тестового датасета АвтоАсессора
(см. `Формат_тестового_датасета.xlsx`).

## Состав

```
laim_converter/
├── __init__.py            # public API
├── converter.py           # основной конвертер (диспетчер по aef_kind, агрегация трейсов)
├── self_checks.py         # модуль самопроверки сгенерированной корзины
└── run_conversion.py      # CLI / точка входа с встроенными self-check'ами
```

## Поток данных

```
spans (parquet, формат ТЗ)
    │
    │  (extractor.py)  диспетчер по aef_kind:
    │                  - input_request → user_query
    │                  - root start_agent → output_answer
    │                  - все start_agent → [module_name]_*
    ▼
TraceRecord(query_id=span_id корневого input_request,
            input_query, output_answer, session_id, modules)
    │
    │  (mode='queries' или 'dialogue')
    ▼
basket (pd.DataFrame в формате xlsx)
    │
    │  (LEFT JOIN на (session_id, query_id) или (session_id))
    ▼
basket + scenario + reference_answer + *_metric из validation_data
    │
    │  (self_checks.py)  16 проверок:
    │    - наличие обязательных полей
    │    - типы, NULL'ы, уникальность query_id
    │    - dialogue list[(str,str,str)] structure
    │    - наличие *_metric колонок
    │    - парность [module_name]_* полей
    │    - покрытие validation_data ↔ basket
    │    - покрытие traces ↔ basket
    │    - перенос метрик и reference_answer
    ▼
CheckReport (PASS/FAIL/WARN/SKIP по каждой проверке)
```

## Использование

### Из Python

```python
import pandas as pd
from laim_converter import convert_and_verify

traces = pd.read_parquet('spans.parquet')              # формат ТЗ
validation_data = pd.read_parquet('validation.parquet') # с *_metric

# Полный пайплайн: конвертация + self-check'и + вывод отчёта
result = convert_and_verify(
    traces=traces,
    validation_data=validation_data,
    mode='queries',          # или 'dialogue'
    raise_on_fail=False,     # True - падать при FAIL'ах
    verbose=True,            # печатать отчёт
)

basket = result['auto_assessor_df']  # тестовая корзина в целевом формате
report = result['report']            # CheckReport со всеми проверками
print(report.has_failures)           # True/False
print(report.summary)                # {'PASS': 16, 'WARN': 0, 'FAIL': 0, 'SKIP': 0}
```

### Только конвертация без self-check'ов

```python
from laim_converter import convert

result = convert(traces, validation_data, mode='queries', strict=False)
basket = result['auto_assessor_df']
```

### Только self-check существующей корзины

```python
from laim_converter import self_check

report = self_check(
    basket=existing_basket_df,
    mode='queries',
    validation_data=validation_data,    # для проверок покрытия (опционально)
    original_traces=traces,             # для проверки полноты (опционально)
    raise_on_fail=False,
)
print(report.render())
```

### Обратная совместимость

Сохранена сигнатура исходного `main(validation_data, traces) -> dict`:

```python
from laim_converter import main
out = main(validation_data, traces)
basket = out['auto_assessor_df']
```

Режим автоопределяется по наличию колонки `dialogue` в `validation_data`.

### CLI

```bash
# Demo на example_spans.parquet
python -m laim_converter.run_conversion

# Реальный запуск
python -m laim_converter.run_conversion path/to/spans.parquet \
    --validation path/to/validation.parquet \
    --mode queries \
    --output path/to/basket.parquet \
    --strict
```

## Поддерживаемые режимы

### `mode='queries'` (по умолчанию)
Одна строка на пользовательский запрос. Колонки:
- **обязательные:** `query_id`, `input_query`, `output_answer`
- **опциональные:** `session_id`
- **модульные (per start_agent):** `[module_name]_input_query`, `[module_name]_output_answer`
- **из validation_data:** `scenario`, `reference_answer`, `*_metric`, `[module_name]_[metric]_metric`

### `mode='dialogue'`
Одна строка на сессию. Колонки:
- **обязательные:** `dialogue: list[(query_id, input_query, output_answer)]`
- **опциональные:** `session_id`
- **из validation_data:** `scenario`, `*_metric`

Записи группируются по `session_id`, упорядочиваются по `start_time_ns` корневого
`input_request`-спана.

## Алгоритм извлечения

1. **Группировка по `trace_id`.**
2. **Поиск корневого `input_request`-спана** — `parent_span_id == 'root'` AND
   `aef_kind == 'input_request'`. Это HTTP-эндпоинт пользовательского запроса.
3. **Извлечение `input_query`** — парсинг `input_text` как JSON, извлечение
   ключа `message` (или fallback: `query`, `input`, `text`, `prompt`, `question`).
4. **Поиск корневого `start_agent`** — `parent_span_id == span_id_корневого_request`.
   Это верхнеуровневый агент трейса.
5. **Извлечение `output_answer`** — парсинг `output_text` корневого `start_agent`,
   извлечение `summary`/`answer`/`result`/etc. Fallback: `output_text` самого
   `input_request` (если не `processing`); затем последний `output_request`.
6. **Модульная декомпозиция** — все `start_agent`-спаны трейса трактуются как
   отдельные модули; имя — нормализованное `span_name` (CamelCase → snake_case).
7. **`query_id`** — `span_id` корневого `input_request` (base64, уникален в трейсе).
8. **`session_id`** — берётся из `session_id`-колонки спана.

## Реестр self-check'ов

| Проверка | Когда выполняется | Что валидирует |
|---|---|---|
| `basket_not_empty` | всегда | Непустая корзина |
| `required_columns` | всегда | `query_id`/`input_query`/`output_answer` (или `dialogue`) |
| `no_nulls_in_required` | всегда | Отсутствие NULL в обязательных полях |
| `required_types` | всегда | Корректные типы (str / list[tuple]) |
| `query_id_uniqueness` | mode=queries | Уникальность `query_id` |
| `nonempty_input_query` | mode=queries | Все `input_query` непусты (FAIL) |
| `nonempty_output_answer` | mode=queries | Все `output_answer` непусты (WARN) |
| `dialogue_structure` | mode=dialogue | Каждый элемент — `(str, str, str)` |
| `dialogue_nonempty` | mode=dialogue | Списки `dialogue` непустые |
| `session_id_uniqueness` | mode=dialogue | Уникальность session_id |
| `metric_columns_present` | всегда | Наличие ≥1 `*_metric` колонки |
| `module_field_pairing` | всегда | Парность `[module]_input_query` ↔ `_output_answer` |
| `no_internal_columns` | всегда | Отсутствие технических колонок (`traceid`, `history`...) |
| `no_unknown_object_types` | всегда | Только примитивы / list в значениях |
| `validation_coverage` | если есть validation_data | Все строки разметки → строки корзины |
| `metrics_carried_over` | если есть validation_data | `*_metric` перенесены в корзину |
| `reference_answers_carried_over` | если есть validation_data | `reference_answer*` перенесены |
| `trace_coverage` | если есть original_traces, mode=queries | Все трейсы с `input_request` → корзина |
| `session_id_consistency` | если есть original_traces | session_id корзины ⊆ session_id трейсов |

Возможные статусы: `PASS` / `FAIL` (блокирующее) / `WARN` (некритичное) / `SKIP`
(нечего проверять).

## Зависимости

- pandas
- pyarrow (для чтения/записи parquet)
- (опционально) polars - для прогона строгой валидации схемы по `validation_code.py`
  из ТЗ. Конвертер не требует polars сам по себе.

## Совместимость с форматом ТЗ

Конвертер работает с плоской таблицей spans согласно spec.parquet:
- 43 колонки фиксированной схемы
- Все идентификаторы — base64-строки
- `parent_span_id`, `agent_id` — стражевые `"root"`/`"outside"`
- `start_time_ns`, `end_time_ns` — int64 наносекунды
- `aef_kind` — enum из 11 значений
- `input_text`, `output_text` — JSON-строки, семантика которых зависит от `aef_kind`

Не требует:
- предварительной декомпрессии (zlib/base64) — отсутствует в ТЗ
- нормализации имён колонок — имена уже соответствуют целевым
- вложенных структур `spans`/`attributes` — формат уже плоский
