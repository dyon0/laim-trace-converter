"""
LAIM Trace-to-Basket Converter
==============================

Преобразование AEF-телеметрии (формат ТЗ "Сбор данных для системы распознавания
аномалий в МАС", v1.6) в формат тестового датасета АвтоАсессора
(см. Формат_тестового_датасета.xlsx).

Поддерживает два режима:

    - "queries"  - одна строка на пользовательский запрос (root input_request).
                   Поля: query_id, input_query, output_answer,
                         session_id, [module_name]_input_query,
                         [module_name]_output_answer.
    - "dialogue" - одна строка на сессию.
                   Поля: session_id, dialogue: list[(query_id, input_query, output_answer)].

Целевые поля типа [metric_name]_metric и reference_answer не извлекаются
из трейсов - они приходят из validation_data (размеченной корзины) через
JOIN по (session_id, query_id).

Public API:

    convert(traces, validation_data=None, mode='queries', strict=False) -> dict
    main(validation_data, traces) -> dict   # обратно-совместимый интерфейс
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd


# ============================================================================
# Константы из ТЗ
# ============================================================================

AEF_KINDS: Tuple[str, ...] = (
    'llm', 'start_agent', 'chain', 'tool', 'retriever',
    'input_request', 'output_request',
    'kafka_produce', 'kafka_consume', 'other', 'guard',
)

# Минимально необходимый набор колонок входа (по spec.parquet)
REQUIRED_TRACE_COLUMNS: Tuple[str, ...] = (
    'trace_id', 'span_id', 'parent_span_id', 'agent_id',
    'start_time_ns', 'end_time_ns',
    'aef_kind', 'span_name',
    'input_text', 'output_text',
    'session_id',
)

# Стражевые значения (sentinels) - не NULL, но семантика "отсутствует/неприменимо"
S_EMPTY = ''
S_ROOT = 'root'
S_OUTSIDE = 'outside'
S_INT_NA = -1
S_FLOAT_NA = -1.0

# Эвристический список ключей, в которых ожидается человекочитаемый
# текст пользовательского запроса (для разбора JSON-полезной нагрузки)
_USER_QUERY_KEYS: Tuple[str, ...] = (
    'message', 'query', 'input', 'text', 'prompt', 'question', 'content', 'goal',
)

# То же для текста ответа агента
_AGENT_ANSWER_KEYS: Tuple[str, ...] = (
    'answer', 'result', 'response', 'output', 'summary',
    'content', 'text', 'message', 'confirmation',
)

Mode = Literal['queries', 'dialogue']


# ============================================================================
# JSON-парсинг с учётом стражевых значений
# ============================================================================

def _parse_json(text: Any) -> Optional[Any]:
    """Распарсить JSON-строку, считая стражевые значения отсутствующими.

    Возвращает None для: NaN, None, пустого стража "", невалидного JSON.
    """
    if text is None:
        return None
    if isinstance(text, float) and pd.isna(text):
        return None
    if not isinstance(text, str) or text == S_EMPTY:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _stringify(value: Any) -> str:
    """Привести значение к строковому виду, годному для записи в корзину."""
    if value is None:
        return ''
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _extract_text_from_payload(
    payload: Any,
    candidate_keys: Tuple[str, ...],
) -> Optional[str]:
    """Из распарсенного JSON-объекта извлечь человекочитаемый текст.

    Сначала перебираются `candidate_keys` (предпочтительные строковые поля);
    если не найдены - сериализуется весь объект.
    """
    if payload is None:
        return None
    if isinstance(payload, str):
        return payload.strip() or None
    if isinstance(payload, dict):
        for key in candidate_keys:
            if key in payload:
                value = payload[key]
                if isinstance(value, str) and value.strip():
                    return value.strip()
                if isinstance(value, (dict, list)):
                    return json.dumps(value, ensure_ascii=False)
        # Не нашли явного ключа - сериализуем словарь целиком
        return json.dumps(payload, ensure_ascii=False)
    if isinstance(payload, list):
        return json.dumps(payload, ensure_ascii=False)
    return _stringify(payload)


# ============================================================================
# Извлечение по типам спанов (диспетчер по aef_kind)
# ============================================================================

def extract_user_query(span: pd.Series) -> Optional[str]:
    """Извлечь текст пользовательского запроса из input_request-спана.

    По ТЗ input_text для input_request - HTTP body, обычно
    `{"message": "<текст>"}`.
    """
    payload = _parse_json(span['input_text'])
    return _extract_text_from_payload(payload, _USER_QUERY_KEYS)


def extract_agent_answer(span: pd.Series) -> Optional[str]:
    """Извлечь финальный ответ агента из output_text спана.

    Для start_agent: output_text = `{"summary": "..."}` или подобное.
    Для output_request: HTTP response body.
    """
    payload = _parse_json(span['output_text'])
    return _extract_text_from_payload(payload, _AGENT_ANSWER_KEYS)


def extract_llm_messages(span: pd.Series) -> List[Dict[str, str]]:
    """Извлечь массив сообщений из LLM-спана.

    По ТЗ для aef_kind='llm' input_text - JSON-массив объектов
    `[{"role": ..., "content": ...}, ...]`.
    """
    payload = _parse_json(span['input_text'])
    if not isinstance(payload, list):
        return []
    messages = []
    for item in payload:
        if isinstance(item, dict):
            role = item.get('role', '')
            content = item.get('content', '')
            if isinstance(content, str) and content.strip():
                messages.append({'role': str(role), 'content': content.strip()})
    return messages


# ============================================================================
# Нормализация имён модулей
# ============================================================================

def _normalize_module_name(span_name: str) -> str:
    """Преобразовать имя спана (напр. 'TravelPlannerAgent') в безопасный
    суффикс для колонки корзины ('travel_planner_agent').
    """
    if not span_name or not isinstance(span_name, str):
        return 'module'
    # CamelCase -> snake_case
    s = re.sub(r'(?<!^)(?=[A-Z])', '_', span_name).lower()
    s = re.sub(r'[^a-z0-9_]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s or 'module'


# ============================================================================
# Извлечение записи из одного трейса
# ============================================================================

@dataclass
class TraceRecord:
    """Извлечённая запись по одному трейсу: один пользовательский запрос
    и финальный ответ системы, плюс модульная декомпозиция.
    """
    trace_id: str
    query_id: str               # = span_id корневого input_request-спана
    session_id: Optional[str]
    input_query: str
    output_answer: str
    start_time_ns: int
    modules: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def is_complete(self) -> bool:
        return bool(self.input_query) and bool(self.output_answer)


def _extract_modules(trace_spans: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """По всем start_agent-спанам собрать модульную декомпозицию.

    Каждый start_agent трактуется как отдельный модуль; имя - из span_name.
    При повторе имени в трейсе суффиксы _2, _3 ...
    """
    modules: Dict[str, Dict[str, str]] = {}
    agents = trace_spans[trace_spans['aef_kind'] == 'start_agent']
    if len(agents) == 0:
        return modules

    agents = agents.sort_values('start_time_ns')
    name_counts: Dict[str, int] = {}

    for _, span in agents.iterrows():
        base = _normalize_module_name(span.get('span_name', ''))
        name_counts[base] = name_counts.get(base, 0) + 1
        unique = base if name_counts[base] == 1 else f'{base}_{name_counts[base]}'

        in_payload = _parse_json(span['input_text'])
        out_payload = _parse_json(span['output_text'])

        modules[unique] = {
            'input_query': _extract_text_from_payload(in_payload, _USER_QUERY_KEYS) or '',
            'output_answer': _extract_text_from_payload(out_payload, _AGENT_ANSWER_KEYS) or '',
        }
    return modules


def extract_trace_record(trace_spans: pd.DataFrame) -> Optional[TraceRecord]:
    """Извлечь TraceRecord из спанов одного трейса.

    Алгоритм:
      1. Найти корневой input_request (parent_span_id == 'root', aef_kind == 'input_request').
      2. Из его input_text взять текст пользовательского запроса.
      3. Найти "корневой" start_agent (непосредственный потомок input_request).
         Из его output_text взять финальный ответ агента.
      4. Если корневого агента нет - fallback на output_text самого input_request,
         затем на последний output_request.
      5. Собрать модульную декомпозицию по всем start_agent-спанам.
    """
    if len(trace_spans) == 0:
        return None

    trace_id = trace_spans.iloc[0]['trace_id']

    root_request_mask = (
        (trace_spans['aef_kind'] == 'input_request') &
        (trace_spans['parent_span_id'] == S_ROOT)
    )
    root_requests = trace_spans[root_request_mask]
    if len(root_requests) == 0:
        return None

    root_request = root_requests.sort_values('start_time_ns').iloc[0]

    user_query = extract_user_query(root_request)
    if not user_query:
        return None

    query_id = str(root_request['span_id'])
    sess_raw = root_request.get('session_id', S_EMPTY)
    session_id = str(sess_raw) if sess_raw not in (S_EMPTY, None) and not (
        isinstance(sess_raw, float) and pd.isna(sess_raw)
    ) else None

    # Корневой агент - непосредственный потомок input_request с aef_kind='start_agent'
    root_agent_mask = (
        (trace_spans['aef_kind'] == 'start_agent') &
        (trace_spans['parent_span_id'] == root_request['span_id'])
    )
    root_agents = trace_spans[root_agent_mask]

    output_answer: Optional[str] = None
    if len(root_agents) > 0:
        root_agent = root_agents.sort_values('start_time_ns').iloc[0]
        output_answer = extract_agent_answer(root_agent)

    # Fallback 1: output_text самого input_request (если он непустой и не "processing")
    if not output_answer:
        candidate = extract_agent_answer(root_request)
        if candidate and candidate.lower() not in (
            '{"status": "processing"}', 'processing', '{"status":"processing"}'
        ):
            output_answer = candidate

    # Fallback 2: последний output_request
    if not output_answer:
        outreqs = trace_spans[trace_spans['aef_kind'] == 'output_request']
        if len(outreqs) > 0:
            last = outreqs.sort_values('start_time_ns').iloc[-1]
            output_answer = extract_agent_answer(last)

    modules = _extract_modules(trace_spans)

    return TraceRecord(
        trace_id=str(trace_id),
        query_id=query_id,
        session_id=session_id,
        input_query=user_query,
        output_answer=output_answer or '',
        start_time_ns=int(root_request['start_time_ns']),
        modules=modules,
    )


# ============================================================================
# Валидация входной схемы
# ============================================================================

def validate_input_schema(traces: pd.DataFrame) -> List[str]:
    """Проверить базовое соответствие входа спецификации ТЗ.

    Возвращает список строк с ошибками; пустой список = всё валидно.
    """
    errors: List[str] = []

    if len(traces) == 0:
        errors.append('Traces DataFrame пуст')
        return errors

    missing = [c for c in REQUIRED_TRACE_COLUMNS if c not in traces.columns]
    if missing:
        errors.append(
            f'Отсутствуют обязательные колонки ТЗ: {missing}. '
            f'Имеющиеся: {list(traces.columns)[:10]}...'
        )

    if 'aef_kind' in traces.columns:
        invalid = set(traces['aef_kind'].astype(str).unique()) - set(AEF_KINDS)
        if invalid:
            errors.append(f'Недопустимые значения aef_kind: {invalid}')

    if 'start_time_ns' in traces.columns:
        if traces['start_time_ns'].isna().any():
            errors.append('start_time_ns содержит NULL (нарушение страж-семантики ТЗ)')

    if 'parent_span_id' in traces.columns:
        if traces['parent_span_id'].isna().any():
            errors.append('parent_span_id содержит NULL (должен быть base64 или "root")')

    return errors


# ============================================================================
# Сборка выходной таблицы по режимам
# ============================================================================

def _build_queries_dataframe(records: List[TraceRecord]) -> pd.DataFrame:
    """Режим 'queries': одна строка на трейс."""
    if not records:
        return pd.DataFrame(
            columns=['query_id', 'session_id', 'input_query', 'output_answer']
        )

    rows = []
    for r in records:
        row = {
            'query_id': r.query_id,
            'session_id': r.session_id or '',
            'input_query': r.input_query,
            'output_answer': r.output_answer,
        }
        for module_name, payload in r.modules.items():
            row[f'{module_name}_input_query'] = payload['input_query']
            row[f'{module_name}_output_answer'] = payload['output_answer']
        rows.append(row)

    df = pd.DataFrame(rows)
    # Заполняем NaN в модульных колонках пустыми строками (страж-семантика)
    for col in df.columns:
        if col.endswith('_input_query') or col.endswith('_output_answer'):
            df[col] = df[col].fillna('')
    return df


def _build_dialogue_dataframe(records: List[TraceRecord]) -> pd.DataFrame:
    """Режим 'dialogue': одна строка на сессию."""
    if not records:
        return pd.DataFrame(columns=['session_id', 'dialogue'])

    sessions: Dict[str, List[TraceRecord]] = {}
    orphans: List[TraceRecord] = []
    for r in records:
        if r.session_id:
            sessions.setdefault(r.session_id, []).append(r)
        else:
            orphans.append(r)

    rows = []
    # Стабильный порядок сессий по времени первого запроса
    session_order = sorted(
        sessions.items(),
        key=lambda kv: min(rec.start_time_ns for rec in kv[1]),
    )
    for session_id, recs in session_order:
        recs_sorted = sorted(recs, key=lambda x: x.start_time_ns)
        dialogue = [
            (r.query_id, r.input_query, r.output_answer)
            for r in recs_sorted
        ]
        rows.append({'session_id': session_id, 'dialogue': dialogue})

    # Записи без session_id - однотурные диалоги
    for r in orphans:
        rows.append({
            'session_id': '',
            'dialogue': [(r.query_id, r.input_query, r.output_answer)],
        })

    return pd.DataFrame(rows)


# ============================================================================
# Слияние с validation_data
# ============================================================================

# Структурные колонки корзины - не должны затираться из validation_data
_STRUCTURAL_COLUMNS = frozenset({
    'query_id', 'input_query', 'output_answer', 'dialogue',
})


def _merge_validation(
    basket: pd.DataFrame,
    validation_data: pd.DataFrame,
    mode: Mode,
) -> pd.DataFrame:
    """LEFT JOIN корзины с разметкой по (session_id, query_id) или (session_id).

    Структурные колонки (input_query, output_answer, dialogue, query_id) не
    перезаписываются и не дублируются: если они есть в validation_data,
    они отбрасываются перед merge (приоритет у конвертера).
    """
    if validation_data is None or len(validation_data) == 0:
        return basket

    if mode == 'queries':
        keys: List[str] = []
        if 'query_id' in validation_data.columns and 'query_id' in basket.columns:
            keys.append('query_id')
        if 'session_id' in validation_data.columns and 'session_id' in basket.columns:
            keys.append('session_id')
        if not keys:
            return basket
        drop_cols = [
            c for c in validation_data.columns
            if c in _STRUCTURAL_COLUMNS and c not in keys
        ]
        vd = validation_data.drop(columns=drop_cols) if drop_cols else validation_data
        return basket.merge(vd, on=keys, how='left', suffixes=('', '_val'))

    if mode == 'dialogue':
        if 'session_id' not in validation_data.columns or 'session_id' not in basket.columns:
            return basket
        drop_cols = [
            c for c in validation_data.columns
            if c in _STRUCTURAL_COLUMNS and c != 'session_id'
        ]
        vd = validation_data.drop(columns=drop_cols) if drop_cols else validation_data
        return basket.merge(vd, on='session_id', how='left', suffixes=('', '_val'))

    return basket


# ============================================================================
# Public API
# ============================================================================

def convert(
    traces: pd.DataFrame,
    validation_data: Optional[pd.DataFrame] = None,
    mode: Mode = 'queries',
    strict: bool = False,
) -> Dict[str, Any]:
    """Преобразование TЗ-формата трейсов в формат тестовой корзины.

    Args:
        traces: DataFrame в формате spans по ТЗ (43 колонки).
        validation_data: размеченная корзина (опционально). Минимальные колонки:
            query_id (или session_id) + ≥1 *_metric. Может содержать scenario,
            reference_answer, [module_name]_*_metric.
        mode: 'queries' (одна строка на запрос) или 'dialogue' (одна на сессию).
        strict: если True, при ошибках валидации входа кидает ValueError.

    Returns:
        dict со следующими ключами:
            'auto_assessor_df' (pd.DataFrame): итоговая корзина
            'records' (list[TraceRecord]): извлечённые записи
            'errors' (list[str]): ошибки/предупреждения валидации входа
            'stats' (dict): статистика по конвертации
            'skipped_trace_ids' (list[str]): трейсы без корневого input_request
            'mode' (str): использованный режим
    """
    schema_errors = validate_input_schema(traces)
    if schema_errors and strict:
        raise ValueError(
            'Schema validation failed: ' + '; '.join(schema_errors)
        )

    records: List[TraceRecord] = []
    skipped: List[str] = []

    # Если базовых колонок нет - возвращаем пустой результат с ошибками
    if 'trace_id' not in traces.columns or 'aef_kind' not in traces.columns:
        return {
            'auto_assessor_df': pd.DataFrame(),
            'records': [],
            'errors': schema_errors,
            'stats': {'total_traces': 0, 'extracted_records': 0,
                      'skipped_traces': 0, 'output_rows': 0, 'mode': mode},
            'skipped_trace_ids': [],
            'mode': mode,
        }

    for trace_id, group in traces.groupby('trace_id'):
        record = extract_trace_record(group)
        if record is not None:
            records.append(record)
        else:
            skipped.append(str(trace_id))

    if mode == 'queries':
        basket = _build_queries_dataframe(records)
    elif mode == 'dialogue':
        basket = _build_dialogue_dataframe(records)
    else:
        raise ValueError(
            f"Unknown mode: '{mode}'. Допустимо: 'queries' или 'dialogue'."
        )

    if validation_data is not None:
        basket = _merge_validation(basket, validation_data, mode)

    stats = {
        'total_traces': int(traces['trace_id'].nunique()),
        'extracted_records': len(records),
        'skipped_traces': len(skipped),
        'output_rows': len(basket),
        'mode': mode,
    }

    return {
        'auto_assessor_df': basket,
        'records': records,
        'errors': schema_errors,
        'stats': stats,
        'skipped_trace_ids': skipped,
        'mode': mode,
    }


def main(validation_data: pd.DataFrame, traces: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Обратно-совместимый интерфейс под исходную сигнатуру main().

    Режим определяется автоматически: если у validation_data есть колонка
    'dialogue' - используется 'dialogue', иначе 'queries'.
    """
    is_dialogue = (
        validation_data is not None
        and 'dialogue' in getattr(validation_data, 'columns', [])
    )
    mode: Mode = 'dialogue' if is_dialogue else 'queries'
    result = convert(traces, validation_data, mode=mode, strict=False)
    return {'auto_assessor_df': result['auto_assessor_df']}
