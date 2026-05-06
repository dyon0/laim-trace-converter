"""
Synthetic AEF telemetry generator (LAIM МАС, ТЗ v1.6).

Назначение
----------
Сгенерировать таблицу spans, *строго* соответствующую спецификации ТЗ
(43 поля, страж-семантика, типы, NOT NULL), а вместе с ней - совместимую
размеченную корзину validation_data. Это позволяет end-to-end протестировать
весь конвейер: generator -> converter -> self_checks без живых данных.

Главные гарантии:
    * 43 колонки в фиксированном порядке;
    * все значения - не NULL, страж-семантика по ТЗ;
    * base64-идентификаторы для trace_id / span_id / parent_span_id / agent_id;
    * корректная иерархия (parent_span_id ссылается на существующий span_id);
    * agent_id = ближайший предок с aef_kind='start_agent', либо 'outside';
    * end_time_ns >= start_time_ns;
    * input_text/output_text - валидный JSON либо "";
    * llm_* заполнены только для aef_kind='llm', иначе стражи (-1, -1.0, false);
    * http_* заполнены только для input_request/output_request, иначе стражи;
    * kafka_* заполнены только для kafka_produce/kafka_consume.

Public API:
    generate_dataset(num_traces, ..., seed) -> {'spans': df, 'validation_data': df}
    main() — CLI: python -m laim_converter.synthetic_generator
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ============================================================================
# Константы спецификации ТЗ (раздел 3, Приложение A)
# ============================================================================

SPAN_COLUMNS: Tuple[str, ...] = (
    'trace_id', 'span_id', 'parent_span_id', 'agent_id',
    'start_time_ns', 'end_time_ns',
    'status_code', 'status_message',
    'aef_kind', 'otel_kind', 'span_name',
    'input_text', 'output_text',
    'llm_model', 'llm_prompt_tokens', 'llm_completion_tokens',
    'llm_total_tokens', 'llm_precached_prompt_tokens',
    'llm_temperature', 'llm_top_p', 'llm_max_tokens',
    'llm_repetition_penalty', 'llm_profanity_check', 'llm_stream',
    'http_method', 'http_path', 'http_status_code',
    'request_headers', 'response_headers',
    'kafka_topic', 'kafka_cluster', 'kafka_consumer_group',
    'kafka_bootstrap_servers',
    'meta_langgraph_step', 'meta_langgraph_node',
    'meta_langgraph_triggers', 'meta_langgraph_path',
    'meta_checkpoint_ns', 'meta_tags', 'meta_extra',
    'session_id', 'service_name', 'service_version',
)
assert len(SPAN_COLUMNS) == 43

STATUS_CODES = ('STATUS_CODE_UNSET', 'STATUS_CODE_OK', 'STATUS_CODE_ERROR')
AEF_KINDS = (
    'llm', 'start_agent', 'chain', 'tool', 'retriever',
    'input_request', 'output_request',
    'kafka_produce', 'kafka_consume', 'other', 'guard',
)
OTEL_KINDS = (
    'SPAN_KIND_INTERNAL', 'SPAN_KIND_SERVER', 'SPAN_KIND_CLIENT',
    'SPAN_KIND_PRODUCER', 'SPAN_KIND_CONSUMER', 'SPAN_KIND_UNSPECIFIED',
)
HTTP_METHODS = ('GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS', 'NONE')

_BASE64_RE = re.compile(r'^[A-Za-z0-9+/]+={0,2}$')


# ============================================================================
# Сценарии (профили выполнения МАС)
# ============================================================================

@dataclass(frozen=True)
class Scenario:
    """Профиль одного типа пользовательского запроса."""
    name: str                         # ключ scenario в validation_data
    user_query: str
    final_answer: str
    reference_answer: str             # эталон для асессора
    quality_metric: float             # 0.0..1.0
    relevance_metric: float           # 0.0..1.0
    root_agent: str                   # span_name корневого start_agent
    sub_agents: Tuple[str, ...] = ()  # span_name дочерних start_agent (опционально)
    tools: Tuple[str, ...] = ()       # span_name инструментов корневого агента
    use_retriever: bool = False
    use_kafka: bool = False
    has_guard: bool = False


DEFAULT_SCENARIOS: Tuple[Scenario, ...] = (
    Scenario(
        name='travel_booking',
        user_query='Забронируй поездку в Париж на 5 дней с 10 по 15 июня, отель 4* в центре, бизнес-класс.',
        final_answer='Поездка забронирована: отель Le Grand Paris (4*), бизнес-класс A380, чек № BK-98765.',
        reference_answer='Должна быть забронирована поездка в Париж 10-15 июня, отель 4* в центре, билеты бизнес-класса.',
        quality_metric=0.92,
        relevance_metric=0.95,
        root_agent='TravelPlannerAgent',
        sub_agents=('HotelBookingAgent', 'FlightBookingAgent'),
        tools=('search_hotels_api', 'search_flights_api'),
        use_retriever=True,
    ),
    Scenario(
        name='qa_simple',
        user_query='Сколько спутников у Юпитера?',
        final_answer='У Юпитера 95 подтверждённых спутников (по состоянию на 2024 год).',
        reference_answer='У Юпитера 95 подтверждённых спутников.',
        quality_metric=0.88,
        relevance_metric=0.97,
        root_agent='QnAAgent',
        tools=('knowledge_base_lookup',),
        use_retriever=True,
    ),
    Scenario(
        name='code_assist',
        user_query='Напиши на Python функцию, считающую число Фибоначчи через мемоизацию.',
        final_answer='```python\nfrom functools import lru_cache\n@lru_cache\ndef fib(n): return n if n<2 else fib(n-1)+fib(n-2)\n```',
        reference_answer='Корректная реализация fib через memoization (lru_cache или dict).',
        quality_metric=0.81,
        relevance_metric=0.9,
        root_agent='CodeAssistantAgent',
        tools=('python_executor',),
    ),
    Scenario(
        name='moderation',
        user_query='Сообщи прогноз погоды в Москве.',
        final_answer='В Москве сегодня +18°C, переменная облачность, без осадков.',
        reference_answer='Должен быть передан корректный прогноз погоды для Москвы.',
        quality_metric=0.9,
        relevance_metric=0.93,
        root_agent='WeatherAgent',
        tools=('weather_api',),
        use_kafka=True,
        has_guard=True,
    ),
)


# ============================================================================
# Утилиты идентификаторов и времени
# ============================================================================

def _b64_id(rng: random.Random, n_bytes: int) -> str:
    """Сгенерировать base64-строку из n_bytes случайных байт."""
    raw = bytes(rng.randint(0, 255) for _ in range(n_bytes))
    out = base64.b64encode(raw).decode('ascii')
    assert _BASE64_RE.match(out), out
    return out


# ============================================================================
# Построитель спана (страж-инициализация по ТЗ)
# ============================================================================

def _empty_span(trace_id: str, session_id: str, service_name: str,
                service_version: str) -> Dict[str, Any]:
    """Строка spans, в которой все 43 поля заполнены страж-значениями.

    Конкретные aef_kind-зависимые поля затем переопределяются билдером.
    """
    return {
        'trace_id': trace_id,
        'span_id': '',                # будет переопределено
        'parent_span_id': 'root',
        'agent_id': 'outside',
        'start_time_ns': 0,
        'end_time_ns': 0,
        'status_code': 'STATUS_CODE_UNSET',
        'status_message': '',
        'aef_kind': 'other',          # placeholder, будет переопределён
        'otel_kind': 'SPAN_KIND_INTERNAL',
        'span_name': '',
        'input_text': '',
        'output_text': '',
        # LLM-стражи
        'llm_model': '',
        'llm_prompt_tokens': -1,
        'llm_completion_tokens': -1,
        'llm_total_tokens': -1,
        'llm_precached_prompt_tokens': -1,
        'llm_temperature': -1.0,
        'llm_top_p': -1.0,
        'llm_max_tokens': -1,
        'llm_repetition_penalty': -1.0,
        'llm_profanity_check': False,
        'llm_stream': False,
        # HTTP-стражи
        'http_method': 'NONE',
        'http_path': '',
        'http_status_code': -1,
        'request_headers': '',
        'response_headers': '',
        # Kafka-стражи
        'kafka_topic': '',
        'kafka_cluster': '',
        'kafka_consumer_group': '',
        'kafka_bootstrap_servers': '',
        # Метаданные-стражи
        'meta_langgraph_step': -1,
        'meta_langgraph_node': '',
        'meta_langgraph_triggers': '',
        'meta_langgraph_path': '',
        'meta_checkpoint_ns': '',
        'meta_tags': '',
        'meta_extra': '',
        # Служебные
        'session_id': session_id,
        'service_name': service_name,
        'service_version': service_version,
    }


def _attach_langgraph(span: Dict[str, Any], step: int, node: str) -> None:
    """Заполнить блок meta_langgraph_* для спана внутри графа."""
    span['meta_langgraph_step'] = int(step)
    span['meta_langgraph_node'] = node
    span['meta_langgraph_triggers'] = json.dumps([f'branch:to:{node}'])
    span['meta_langgraph_path'] = json.dumps(['__pregel_pull', node])
    span['meta_checkpoint_ns'] = f'{node}:{step:04d}'
    span['meta_tags'] = json.dumps([f'graph:step:{step}'])


# ============================================================================
# Генератор одного трейса
# ============================================================================

@dataclass
class _TraceCtx:
    """Изменяемый контекст текущего трейса (счётчики, время, накопители)."""
    rng: random.Random
    trace_id: str
    session_id: str
    service_name: str
    service_version: str
    now_ns: int
    step: int = 0
    spans: List[Dict[str, Any]] = field(default_factory=list)
    # стек агентов: list[(span_id_агента, span_name_агента)]
    agent_stack: List[Tuple[str, str]] = field(default_factory=list)

    def advance(self, micros: int = 50_000) -> int:
        self.now_ns += micros * 1_000  # micros -> ns
        return self.now_ns

    def new_span_id(self) -> str:
        return _b64_id(self.rng, 8)

    def current_agent_id(self) -> str:
        return self.agent_stack[-1][0] if self.agent_stack else 'outside'

    def add(self, span: Dict[str, Any]) -> None:
        self.spans.append(span)

    def base_span(self, parent_id: str) -> Dict[str, Any]:
        s = _empty_span(self.trace_id, self.session_id,
                        self.service_name, self.service_version)
        s['span_id'] = self.new_span_id()
        s['parent_span_id'] = parent_id
        s['agent_id'] = self.current_agent_id()
        return s


def _make_input_request(ctx: _TraceCtx, scenario: Scenario) -> Dict[str, Any]:
    """Корневой input_request: HTTP-эндпоинт пользовательского запроса."""
    span = ctx.base_span(parent_id='root')
    span['agent_id'] = 'outside'  # снаружи любого агента
    span['aef_kind'] = 'input_request'
    span['otel_kind'] = 'SPAN_KIND_SERVER'
    span['span_name'] = 'user_http_request'
    span['status_code'] = 'STATUS_CODE_OK'
    span['start_time_ns'] = ctx.advance(0)
    # input_text - тело HTTP-запроса
    span['input_text'] = json.dumps(
        {'message': scenario.user_query}, ensure_ascii=False
    )
    span['output_text'] = json.dumps({'status': 'processing'})
    span['http_method'] = 'POST'
    span['http_path'] = '/chat'
    span['http_status_code'] = 201
    span['request_headers'] = json.dumps({
        'Content-Type': 'application/json',
        'X-Request-ID': str(ctx.rng.randint(1000, 9999)),
    })
    span['response_headers'] = json.dumps({'Content-Type': 'application/json'})
    span['end_time_ns'] = ctx.advance(40_000)
    return span


def _make_start_agent(
    ctx: _TraceCtx,
    parent_id: str,
    span_name: str,
    input_payload: Dict[str, Any],
    output_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """start_agent: создаёт новый агентский контекст (пушится в стек)."""
    span = ctx.base_span(parent_id=parent_id)
    span['aef_kind'] = 'start_agent'
    span['otel_kind'] = 'SPAN_KIND_INTERNAL'
    span['span_name'] = span_name
    span['status_code'] = 'STATUS_CODE_OK'
    span['start_time_ns'] = ctx.advance(20_000)
    span['input_text'] = json.dumps(input_payload, ensure_ascii=False)
    span['output_text'] = json.dumps(output_payload, ensure_ascii=False)
    # agent_id такого спана = его собственный span_id
    # (т.к. это сам корень агентского поддерева)
    span['agent_id'] = span['span_id']
    # Временная отметка конца ставится на «сейчас + ~50 мкс»; реально она
    # переписывается _close_agent_span() после того, как все дочерние спаны
    # этого агента уже сгенерированы (start_agent оборачивает все детей).
    span['end_time_ns'] = span['start_time_ns'] + 50_000
    ctx.step += 1
    _attach_langgraph(span, ctx.step,
                      re.sub(r'(?<!^)(?=[A-Z])', '_', span_name).lower())
    return span


def _close_agent_span(ctx: _TraceCtx, agent_span: Dict[str, Any]) -> None:
    """Зафиксировать end_time_ns корневого/дочернего start_agent после
    завершения всего его поддерева (текущее ctx.now_ns)."""
    agent_span['end_time_ns'] = ctx.advance(20_000)


def _make_llm(ctx: _TraceCtx, parent_id: str, system: str,
              user: str, response: str) -> Dict[str, Any]:
    """llm-спан: вызов модели."""
    span = ctx.base_span(parent_id=parent_id)
    span['aef_kind'] = 'llm'
    span['span_name'] = 'llm_reasoning'
    span['status_code'] = 'STATUS_CODE_OK'
    span['start_time_ns'] = ctx.advance(30_000)
    messages = [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': user},
    ]
    span['input_text'] = json.dumps(messages, ensure_ascii=False)
    span['output_text'] = response
    span['llm_model'] = 'GigaChat:1.0.26.20'
    p_tok = ctx.rng.randint(40, 200)
    c_tok = ctx.rng.randint(15, 120)
    span['llm_prompt_tokens'] = p_tok
    span['llm_completion_tokens'] = c_tok
    span['llm_total_tokens'] = p_tok + c_tok
    span['llm_precached_prompt_tokens'] = ctx.rng.randint(0, p_tok // 2)
    span['llm_temperature'] = round(ctx.rng.uniform(0.2, 0.9), 2)
    span['llm_top_p'] = round(ctx.rng.uniform(0.7, 1.0), 2)
    span['llm_max_tokens'] = 1024
    span['llm_repetition_penalty'] = round(ctx.rng.uniform(1.0, 1.2), 2)
    span['llm_profanity_check'] = False
    span['llm_stream'] = bool(ctx.rng.getrandbits(1))
    ctx.step += 1
    _attach_langgraph(span, ctx.step, 'llm_node')
    span['end_time_ns'] = ctx.advance(80_000)
    return span


def _make_tool(ctx: _TraceCtx, parent_id: str, name: str,
               args: Dict[str, Any], result: Any) -> Dict[str, Any]:
    span = ctx.base_span(parent_id=parent_id)
    span['aef_kind'] = 'tool'
    span['span_name'] = name
    span['status_code'] = 'STATUS_CODE_OK'
    span['start_time_ns'] = ctx.advance(15_000)
    span['input_text'] = json.dumps(args, ensure_ascii=False)
    span['output_text'] = json.dumps(result, ensure_ascii=False)
    ctx.step += 1
    _attach_langgraph(span, ctx.step, f'tool_{name}')
    span['end_time_ns'] = ctx.advance(50_000)
    return span


def _make_retriever(ctx: _TraceCtx, parent_id: str, query: str,
                    docs: List[str]) -> Dict[str, Any]:
    span = ctx.base_span(parent_id=parent_id)
    span['aef_kind'] = 'retriever'
    span['span_name'] = 'vector_search'
    span['status_code'] = 'STATUS_CODE_OK'
    span['start_time_ns'] = ctx.advance(10_000)
    span['input_text'] = json.dumps({'query': query, 'top_k': len(docs)},
                                    ensure_ascii=False)
    span['output_text'] = json.dumps({'documents': docs}, ensure_ascii=False)
    ctx.step += 1
    _attach_langgraph(span, ctx.step, 'retriever_node')
    span['end_time_ns'] = ctx.advance(35_000)
    return span


def _make_chain(ctx: _TraceCtx, parent_id: str, name: str) -> Dict[str, Any]:
    span = ctx.base_span(parent_id=parent_id)
    span['aef_kind'] = 'chain'
    span['span_name'] = name
    span['status_code'] = 'STATUS_CODE_OK'
    span['start_time_ns'] = ctx.advance(5_000)
    span['input_text'] = json.dumps({'step': name})
    span['output_text'] = json.dumps({'ok': True})
    ctx.step += 1
    _attach_langgraph(span, ctx.step, name)
    span['end_time_ns'] = ctx.advance(8_000)
    return span


def _make_guard(ctx: _TraceCtx, parent_id: str) -> Dict[str, Any]:
    span = ctx.base_span(parent_id=parent_id)
    span['aef_kind'] = 'guard'
    span['span_name'] = 'safety_guard'
    span['status_code'] = 'STATUS_CODE_OK'
    span['start_time_ns'] = ctx.advance(3_000)
    span['input_text'] = json.dumps({'check': 'profanity'})
    span['output_text'] = json.dumps({'passed': True})
    span['end_time_ns'] = ctx.advance(5_000)
    return span


def _make_kafka(ctx: _TraceCtx, parent_id: str, kind: str,
                topic: str) -> Dict[str, Any]:
    """kind: 'kafka_produce' или 'kafka_consume'."""
    span = ctx.base_span(parent_id=parent_id)
    span['aef_kind'] = kind
    span['otel_kind'] = ('SPAN_KIND_PRODUCER' if kind == 'kafka_produce'
                        else 'SPAN_KIND_CONSUMER')
    span['span_name'] = f'kafka:{topic}'
    span['status_code'] = 'STATUS_CODE_OK'
    span['start_time_ns'] = ctx.advance(8_000)
    span['input_text'] = json.dumps({'event': 'agent_step', 'topic': topic})
    span['output_text'] = json.dumps({'offset': ctx.rng.randint(1, 10000)})
    span['kafka_topic'] = topic
    span['kafka_cluster'] = 'cluster-prod-01'
    span['kafka_bootstrap_servers'] = json.dumps(
        ['kafka1.prod:9092', 'kafka2.prod:9092']
    )
    if kind == 'kafka_consume':
        span['kafka_consumer_group'] = 'mas-events-consumer'
    span['end_time_ns'] = ctx.advance(12_000)
    return span


def _make_output_request(ctx: _TraceCtx, parent_id: str,
                         answer: str) -> Dict[str, Any]:
    """output_request: финальный HTTP-ответ (опционально - последний)."""
    span = ctx.base_span(parent_id=parent_id)
    span['agent_id'] = 'outside'
    span['aef_kind'] = 'output_request'
    span['otel_kind'] = 'SPAN_KIND_CLIENT'
    span['span_name'] = 'callback_response'
    span['status_code'] = 'STATUS_CODE_OK'
    span['start_time_ns'] = ctx.advance(10_000)
    span['input_text'] = json.dumps(
        {'callback_url': 'https://client/api/cb'}, ensure_ascii=False
    )
    span['output_text'] = json.dumps({'answer': answer}, ensure_ascii=False)
    span['http_method'] = 'POST'
    span['http_path'] = '/api/cb'
    span['http_status_code'] = 200
    span['request_headers'] = json.dumps({'Content-Type': 'application/json'})
    span['response_headers'] = json.dumps({'Content-Type': 'application/json'})
    span['end_time_ns'] = ctx.advance(30_000)
    return span


def _build_one_trace(
    ctx: _TraceCtx,
    scenario: Scenario,
) -> List[Dict[str, Any]]:
    """Сгенерировать спаны одного выполнения МАС по сценарию."""
    # 1. Корневой input_request
    inp = _make_input_request(ctx, scenario)
    ctx.add(inp)

    # 2. (опционально) guard перед агентом
    if scenario.has_guard:
        ctx.add(_make_guard(ctx, parent_id=inp['span_id']))

    # 3. Корневой start_agent
    root_agent = _make_start_agent(
        ctx,
        parent_id=inp['span_id'],
        span_name=scenario.root_agent,
        input_payload={'goal': scenario.user_query},
        output_payload={'summary': scenario.final_answer},
    )
    ctx.spans.append(root_agent)
    ctx.agent_stack.append((root_agent['span_id'], scenario.root_agent))
    parent = root_agent['span_id']

    # 4. LLM-вызов рассуждения
    ctx.add(_make_llm(
        ctx, parent_id=parent,
        system=f'You are a helpful {scenario.root_agent}.',
        user=scenario.user_query,
        response=f'Plan: I will solve "{scenario.user_query[:40]}..." step by step.',
    ))

    # 5. Retriever (опционально)
    if scenario.use_retriever:
        ctx.add(_make_retriever(
            ctx, parent_id=parent,
            query=scenario.user_query,
            docs=[f'doc_{i}: related fact about {scenario.name}' for i in range(3)],
        ))

    # 6. Tools
    for tool_name in scenario.tools:
        ctx.add(_make_tool(
            ctx, parent_id=parent,
            name=tool_name,
            args={'query': scenario.user_query},
            result={'ok': True, 'data': f'{tool_name}_result'},
        ))

    # 7. Дочерние агенты
    for sub_name in scenario.sub_agents:
        sub = _make_start_agent(
            ctx, parent_id=parent,
            span_name=sub_name,
            input_payload={'task': f'sub-task for {sub_name}'},
            output_payload={'confirmation': f'{sub_name} done'},
        )
        ctx.spans.append(sub)
        ctx.agent_stack.append((sub['span_id'], sub_name))
        # Дочерний агент вызывает свой LLM и/или один tool
        ctx.add(_make_llm(
            ctx, parent_id=sub['span_id'],
            system=f'You are {sub_name}.',
            user=f'Process sub-task for {sub_name}',
            response=f'{sub_name}: completed.',
        ))
        ctx.add(_make_chain(ctx, parent_id=sub['span_id'], name='post_process'))
        _close_agent_span(ctx, sub)
        ctx.agent_stack.pop()

    # 8. Kafka (опционально)
    if scenario.use_kafka:
        ctx.add(_make_kafka(ctx, parent_id=parent,
                            kind='kafka_produce', topic='mas.events.out'))
        ctx.add(_make_kafka(ctx, parent_id=parent,
                            kind='kafka_consume', topic='mas.events.in'))

    # 9. Закрываем корневой start_agent (его end_time_ns должен быть
    # >= max(end_time_ns) всех потомков).
    _close_agent_span(ctx, root_agent)
    ctx.agent_stack.pop()  # снимаем root agent

    # 10. Финальный output_request (внешний вызов колбэка)
    ctx.add(_make_output_request(
        ctx, parent_id=inp['span_id'], answer=scenario.final_answer))

    # 11. После завершения трейса в input_request зафиксируем финальный ответ
    # и расширим end_time_ns корневого HTTP-спана до конца трейса.
    inp['output_text'] = json.dumps({'answer': scenario.final_answer},
                                    ensure_ascii=False)
    inp['end_time_ns'] = max(inp['end_time_ns'], ctx.now_ns)

    return ctx.spans


# ============================================================================
# Публичный генератор датасета
# ============================================================================

def generate_dataset(
    num_traces: int = 12,
    scenarios: Tuple[Scenario, ...] = DEFAULT_SCENARIOS,
    sessions: Optional[int] = None,
    seed: int = 42,
    service_name: str = 'mas-runtime',
    service_version: str = '1.0.0',
    base_time_ns: int = 1_777_700_000_000_000_000,
) -> Dict[str, pd.DataFrame]:
    """Сгенерировать synthetic spans + согласованную validation_data.

    Args:
        num_traces: общее число генерируемых трейсов (>=1).
        scenarios: пул сценариев, между которыми round-robin распределяются трейсы.
        sessions: число уникальных session_id (по умолчанию = max(1, num_traces//3),
                  чтобы получить мульти-туровые сессии для режима 'dialogue').
        seed: фиксирует ГПСЧ.
        service_name/service_version: для всех спанов.
        base_time_ns: начальный момент времени.

    Returns:
        {'spans': DataFrame, 'validation_data': DataFrame}
    """
    if num_traces < 1:
        raise ValueError('num_traces must be >= 1')
    if not scenarios:
        raise ValueError('scenarios must be non-empty')

    rng = random.Random(seed)
    n_sessions = sessions if sessions is not None else max(1, num_traces // 3)
    session_ids = [f'sess_{rng.randint(1000, 9999)}_{i}'
                   for i in range(n_sessions)]

    all_spans: List[Dict[str, Any]] = []
    validation_rows: List[Dict[str, Any]] = []
    cur_time = base_time_ns

    for i in range(num_traces):
        scenario = scenarios[i % len(scenarios)]
        session_id = session_ids[i % n_sessions]
        trace_id = _b64_id(rng, 12)
        ctx = _TraceCtx(
            rng=rng,
            trace_id=trace_id,
            session_id=session_id,
            service_name=service_name,
            service_version=service_version,
            now_ns=cur_time,
        )
        trace_spans = _build_one_trace(ctx, scenario)
        all_spans.extend(trace_spans)
        cur_time = ctx.now_ns + 1_000_000  # отступ между трейсами

        # query_id - span_id корневого input_request
        root_inp = next(s for s in trace_spans if s['aef_kind'] == 'input_request'
                        and s['parent_span_id'] == 'root')
        validation_rows.append({
            'query_id': root_inp['span_id'],
            'session_id': session_id,
            'scenario': scenario.name,
            'reference_answer': scenario.reference_answer,
            'quality_metric': scenario.quality_metric,
            'relevance_metric': scenario.relevance_metric,
        })

    spans_df = pd.DataFrame(all_spans, columns=list(SPAN_COLUMNS))
    # Принудительные dtype'ы для соответствия ТЗ (Int64/Float32/Boolean).
    spans_df = _coerce_dtypes(spans_df)
    validation_df = pd.DataFrame(validation_rows)

    return {'spans': spans_df, 'validation_data': validation_df}


_INT_COLS = (
    'start_time_ns', 'end_time_ns',
    'llm_prompt_tokens', 'llm_completion_tokens', 'llm_total_tokens',
    'llm_precached_prompt_tokens', 'llm_max_tokens',
    'http_status_code', 'meta_langgraph_step',
)
_FLOAT_COLS = (
    'llm_temperature', 'llm_top_p', 'llm_repetition_penalty',
)
_BOOL_COLS = ('llm_profanity_check', 'llm_stream')


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Привести колонки к строгим типам ТЗ (Int64/Float32/Boolean)."""
    for c in _INT_COLS:
        df[c] = df[c].astype('int64')
    for c in _FLOAT_COLS:
        df[c] = df[c].astype('float32')
    for c in _BOOL_COLS:
        df[c] = df[c].astype('bool')
    # Все строковые - явный str (на случай, если попали float NaN).
    str_cols = [c for c in df.columns
                if c not in _INT_COLS + _FLOAT_COLS + _BOOL_COLS]
    for c in str_cols:
        df[c] = df[c].astype(str)
    return df


# ============================================================================
# Валидация согласно ТЗ (lite-вариант, без polars)
# ============================================================================

def validate_against_spec(spans: pd.DataFrame) -> List[str]:
    """Проверить соответствие сгенерированных spans ключевым правилам ТЗ.

    Возвращает список ошибок; пустой = строгое соответствие.
    """
    errs: List[str] = []
    if list(spans.columns) != list(SPAN_COLUMNS):
        errs.append(f'Колонки/порядок не соответствуют ТЗ: '
                    f'{list(spans.columns)[:5]}... vs {SPAN_COLUMNS[:5]}...')
        # Дальше нет смысла проверять подробности, если схема нарушена.
        return errs

    # NOT NULL
    if spans.isna().any().any():
        nulls = spans.isna().sum()
        errs.append(f'Обнаружены NULL: {nulls[nulls > 0].to_dict()}')

    # Enum-поля
    for col, allowed in (
        ('status_code', STATUS_CODES),
        ('aef_kind', AEF_KINDS),
        ('otel_kind', OTEL_KINDS),
        ('http_method', HTTP_METHODS),
    ):
        bad = set(spans[col].unique()) - set(allowed)
        if bad:
            errs.append(f'Недопустимые значения {col}: {bad}')

    # Идентификаторы - base64 или sentinel
    for col, sentinel in (('parent_span_id', 'root'), ('agent_id', 'outside')):
        bad_idx = spans[(spans[col] != sentinel)
                        & (~spans[col].str.match(_BASE64_RE.pattern))].index
        if len(bad_idx):
            errs.append(f'{col}: {len(bad_idx)} значений не base64 и не "{sentinel}"')

    for col in ('trace_id', 'span_id'):
        bad_idx = spans[~spans[col].str.match(_BASE64_RE.pattern)].index
        if len(bad_idx):
            errs.append(f'{col}: {len(bad_idx)} значений не base64')

    # Время: end >= start, оба >= 0
    if (spans['start_time_ns'] < 0).any():
        errs.append('start_time_ns содержит отрицательные значения')
    if (spans['end_time_ns'] < spans['start_time_ns']).any():
        errs.append('end_time_ns < start_time_ns в части спанов')

    # LLM-стражи: для не-llm всё должно быть -1/-1.0/False
    non_llm = spans[spans['aef_kind'] != 'llm']
    for c, sentinel in (
        ('llm_model', ''), ('llm_prompt_tokens', -1),
        ('llm_completion_tokens', -1), ('llm_total_tokens', -1),
        ('llm_precached_prompt_tokens', -1), ('llm_max_tokens', -1),
        ('llm_temperature', -1.0), ('llm_top_p', -1.0),
        ('llm_repetition_penalty', -1.0),
        ('llm_profanity_check', False), ('llm_stream', False),
    ):
        if (non_llm[c] != sentinel).any():
            cnt = int((non_llm[c] != sentinel).sum())
            errs.append(f'Не-LLM спанов с непустым {c}: {cnt}')

    # HTTP-стражи: только input_request/output_request содержат HTTP-поля
    http_kinds = {'input_request', 'output_request'}
    non_http = spans[~spans['aef_kind'].isin(http_kinds)]
    for c, sentinel in (
        ('http_method', 'NONE'), ('http_path', ''),
        ('http_status_code', -1),
        ('request_headers', ''), ('response_headers', ''),
    ):
        if (non_http[c] != sentinel).any():
            cnt = int((non_http[c] != sentinel).sum())
            errs.append(f'Не-HTTP спанов с непустым {c}: {cnt}')

    # Kafka-стражи
    kafka_kinds = {'kafka_produce', 'kafka_consume'}
    non_kafka = spans[~spans['aef_kind'].isin(kafka_kinds)]
    for c, sentinel in (
        ('kafka_topic', ''), ('kafka_cluster', ''),
        ('kafka_bootstrap_servers', ''),
    ):
        if (non_kafka[c] != sentinel).any():
            cnt = int((non_kafka[c] != sentinel).sum())
            errs.append(f'Не-Kafka спанов с непустым {c}: {cnt}')

    # status_message: только при ERROR
    bad = spans[(spans['status_code'] != 'STATUS_CODE_ERROR')
                & (spans['status_message'] != '')]
    if len(bad):
        errs.append(f'status_message не пуст для не-ERROR: {len(bad)} спан(ов)')

    # input_text: "" или валидный JSON (для output_text по ТЗ требований нет).
    def _ok_json(s: str) -> bool:
        if s == '':
            return True
        try:
            json.loads(s)
            return True
        except (ValueError, TypeError):
            return False

    bad_cnt = int((~spans['input_text'].apply(_ok_json)).sum())
    if bad_cnt:
        errs.append(f'input_text: {bad_cnt} спан(ов) с невалидным JSON')

    # Уникальность (trace_id, span_id)
    dup = spans.duplicated(subset=['trace_id', 'span_id']).sum()
    if dup:
        errs.append(f'Дубли (trace_id, span_id): {dup}')

    # parent_span_id ссылается на существующий span_id того же trace_id (или 'root')
    for tid, group in spans.groupby('trace_id'):
        ids = set(group['span_id'])
        bad_par = group[(group['parent_span_id'] != 'root')
                        & (~group['parent_span_id'].isin(ids))]
        if len(bad_par):
            errs.append(
                f'trace {tid}: {len(bad_par)} спан(ов) с невалидным parent_span_id'
            )
            break

    return errs


def _aggregate_validation_to_session(validation: pd.DataFrame) -> pd.DataFrame:
    """Свести query-level разметку к session-level: среднее по числовым
    метрикам, первое значение для строковых полей. Используется при
    e2e-прогоне в режиме 'dialogue'."""
    if 'session_id' not in validation.columns:
        return validation
    num_cols = [c for c in validation.columns
                if c.endswith('_metric') and c != 'session_id']
    other_cols = [c for c in validation.columns
                  if c not in num_cols and c not in ('session_id', 'query_id')]
    agg: Dict[str, Any] = {c: 'mean' for c in num_cols}
    agg.update({c: 'first' for c in other_cols})
    return (validation.groupby('session_id', as_index=False)
                      .agg(agg))


# ============================================================================
# CLI
# ============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Сгенерировать synthetic AEF-телеметрию по ТЗ '
                    'и (опционально) прогнать end-to-end через converter.'
    )
    p.add_argument('--num-traces', type=int, default=12,
                   help='Сколько трейсов сгенерировать (default: 12).')
    p.add_argument('--sessions', type=int, default=None,
                   help='Число уникальных session_id (default: num-traces/3).')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out-spans', type=str, default='synthetic_spans.parquet',
                   help='Куда сохранить spans-таблицу.')
    p.add_argument('--out-validation', type=str,
                   default='synthetic_validation.parquet',
                   help='Куда сохранить validation_data.')
    p.add_argument('--mode', choices=('queries', 'dialogue'), default='queries')
    p.add_argument('--e2e', action='store_true',
                   help='После генерации запустить convert_and_verify '
                        'и завершиться с кодом 1 при FAIL self-check.')
    p.add_argument('--strict-spec', action='store_true',
                   help='Дополнительно проверить strict-соответствие ТЗ '
                        'и завершиться с ошибкой при нарушениях.')
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    print(f'[gen] num_traces={args.num_traces} sessions={args.sessions} '
          f'seed={args.seed}')
    out = generate_dataset(
        num_traces=args.num_traces,
        sessions=args.sessions,
        seed=args.seed,
    )
    spans = out['spans']
    validation = out['validation_data']
    print(f'[gen] spans: {len(spans)} rows, '
          f'{spans["trace_id"].nunique()} trace_id, '
          f'{spans["session_id"].nunique()} session_id')
    print(f'[gen] validation_data: {len(validation)} rows, '
          f'columns={list(validation.columns)}')

    spec_errors = validate_against_spec(spans)
    if spec_errors:
        print('[gen] STRICT-SPEC validation errors:', file=sys.stderr)
        for e in spec_errors:
            print('  - ' + e, file=sys.stderr)
        if args.strict_spec:
            return 2
    else:
        print('[gen] STRICT-SPEC validation: OK (43 поля, страж-семантика, '
              'enum-домены, base64, иерархия, времена, JSON).')

    Path(args.out_spans).parent.mkdir(parents=True, exist_ok=True)
    spans.to_parquet(args.out_spans, index=False)
    validation.to_parquet(args.out_validation, index=False)
    print(f'[gen] saved: {args.out_spans} + {args.out_validation}')

    if args.e2e:
        # Импорт здесь, чтобы модуль был самодостаточен при использовании
        # только генератора.
        from .self_checks import convert_and_verify

        # В dialogue-режиме validation_data соединяется с корзиной по
        # session_id (одна строка на сессию). Для этого агрегируем
        # query-level метрики до session-level (среднее, взятие первого).
        e2e_validation = validation
        if args.mode == 'dialogue':
            e2e_validation = _aggregate_validation_to_session(validation)

        print('\n[e2e] Running convert_and_verify on generated data...')
        result = convert_and_verify(
            traces=spans,
            validation_data=e2e_validation,
            mode=args.mode,
            raise_on_fail=False,
            verbose=True,
        )
        report = result['report']
        basket = result['auto_assessor_df']
        print(f'\n[e2e] basket shape={basket.shape} '
              f'columns={list(basket.columns)}')
        print(f'[e2e] report.summary={report.summary}')
        if report.has_failures:
            print('[e2e] FAIL self-check', file=sys.stderr)
            return 1
        print('[e2e] OK: все обязательные self-check\'и прошли.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
