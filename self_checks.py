"""
LAIM Test-Basket Self-Check Module
==================================

Модуль самопроверки тестовой корзины, сгенерированной конвертером
(см. converter.py). Проверяет соответствие выхода целевому формату
(Формат_тестового_датасета.xlsx) и корректность извлечения данных
из исходных трейсов формата ТЗ.

Также содержит функцию `convert_and_verify`, которая:
  1. Запускает конвертацию через converter.convert(...)
  2. Прогоняет полный набор self-check'ов на полученной корзине
  3. Возвращает итоговый отчёт

Public API:

    self_check(basket, mode, validation_data=None, original_traces=None,
               raise_on_fail=False) -> CheckReport

    convert_and_verify(traces, validation_data=None, mode='queries',
                       raise_on_fail=False, verbose=True) -> dict
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

from . import converter as _converter

CheckStatus = Literal['PASS', 'FAIL', 'WARN', 'SKIP']
Mode = Literal['queries', 'dialogue']

_STATUS_MARKERS = {'PASS': '✓', 'FAIL': '✗', 'WARN': '⚠', 'SKIP': '○'}

# Целевые суффиксы/префиксы по xlsx-формату
METRIC_SUFFIX = '_metric'
REFERENCE_PREFIX = 'reference_answer'
MODULE_INPUT_SUFFIX = '_input_query'
MODULE_OUTPUT_SUFFIX = '_output_answer'

# Колонки, которые НЕ должны попасть в финальную корзину
FORBIDDEN_INTERNAL_COLUMNS = frozenset({
    'traceid', 'spanid', 'parentspanid', 'trace_id', 'span_id', 'parent_span_id',
    'span_count', 'message_count', 'starttime', 'endtime',
    'history', 'question', 'answer', 'aefkind', 'aefinput', 'aefoutput',
})


# ============================================================================
# Структуры результатов
# ============================================================================

@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_blocking(self) -> bool:
        return self.status == 'FAIL'


@dataclass
class CheckReport:
    """Отчёт по полному набору self-check'ов."""
    checks: List[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)

    @property
    def has_failures(self) -> bool:
        return any(c.status == 'FAIL' for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(c.status == 'WARN' for c in self.checks)

    @property
    def summary(self) -> Dict[str, int]:
        return dict(Counter(c.status for c in self.checks))

    def render(self) -> str:
        lines = [f'=== Self-check report ({len(self.checks)} checks) ===']
        for c in self.checks:
            marker = _STATUS_MARKERS.get(c.status, '?')
            lines.append(f'  {marker} [{c.status}] {c.name}: {c.message}')
        lines.append(f'--- Summary: {self.summary} ---')
        return '\n'.join(lines)

    def failures(self) -> List[CheckResult]:
        return [c for c in self.checks if c.status == 'FAIL']


# ============================================================================
# Чекер
# ============================================================================

class BasketSelfChecker:
    """Прогон полного набора self-check'ов против сгенерированной корзины."""

    def __init__(
        self,
        basket: pd.DataFrame,
        mode: Mode,
        validation_data: Optional[pd.DataFrame] = None,
        original_traces: Optional[pd.DataFrame] = None,
        records: Optional[List[Any]] = None,
    ):
        self.basket = basket
        self.mode = mode
        self.validation_data = validation_data
        self.original_traces = original_traces
        self.records = records or []
        self.report = CheckReport()

    # ---- Орекстратор ----

    def run_all(self) -> CheckReport:
        # Базовые проверки структуры
        self._check_basket_not_empty()
        self._check_required_columns()
        self._check_no_nulls_in_required()
        self._check_required_types()

        # Режим-специфичные проверки
        if self.mode == 'queries':
            self._check_query_id_uniqueness()
            self._check_input_output_nonempty()
        elif self.mode == 'dialogue':
            self._check_dialogue_structure()
            self._check_session_id_uniqueness()
            self._check_dialogue_nonempty()

        # Проверки соответствия целевому xlsx-формату
        self._check_metric_columns_present()
        self._check_module_field_pairing()
        self._check_no_lingering_internal_columns()
        self._check_no_unknown_object_types()

        # Проверки относительно validation_data
        if self.validation_data is not None:
            self._check_validation_coverage()
            self._check_metrics_carried_over()
            self._check_reference_answers_carried_over()

        # Проверки относительно исходных трейсов
        if self.original_traces is not None:
            self._check_trace_coverage()
            self._check_session_id_consistency()

        return self.report

    def assert_all(self) -> None:
        """Прогнать проверки; выбросить AssertionError при FAIL'ах."""
        report = self.run_all()
        if report.has_failures:
            failures = report.failures()
            messages = '\n  '.join(f'{c.name}: {c.message}' for c in failures)
            raise AssertionError(f'Self-checks failed:\n  {messages}')

    # ---- Базовые проверки ----

    def _check_basket_not_empty(self) -> None:
        if len(self.basket) == 0:
            self.report.add(CheckResult(
                'basket_not_empty', 'FAIL',
                'Корзина пуста - конвертация не дала строк',
            ))
        else:
            self.report.add(CheckResult(
                'basket_not_empty', 'PASS',
                f'Корзина содержит {len(self.basket)} строк',
            ))

    def _check_required_columns(self) -> None:
        if self.mode == 'queries':
            required = ['query_id', 'input_query', 'output_answer']
        else:
            required = ['dialogue']
        missing = [c for c in required if c not in self.basket.columns]
        if missing:
            self.report.add(CheckResult(
                'required_columns', 'FAIL',
                f"Отсутствуют обязательные колонки для режима '{self.mode}': {missing}",
                {'missing': missing, 'present': list(self.basket.columns)},
            ))
        else:
            self.report.add(CheckResult(
                'required_columns', 'PASS',
                f"Все обязательные колонки режима '{self.mode}' присутствуют",
            ))

    def _check_no_nulls_in_required(self) -> None:
        if self.mode == 'queries':
            required = ['query_id', 'input_query', 'output_answer']
        else:
            required = ['dialogue']
        problems: Dict[str, int] = {}
        for col in required:
            if col not in self.basket.columns:
                continue
            nulls = int(self.basket[col].isna().sum())
            if nulls > 0:
                problems[col] = nulls
        if problems:
            self.report.add(CheckResult(
                'no_nulls_in_required', 'FAIL',
                f'Обязательные колонки содержат NULL: {problems}',
                problems,
            ))
        else:
            self.report.add(CheckResult(
                'no_nulls_in_required', 'PASS',
                'NULL в обязательных колонках отсутствуют',
            ))

    def _check_required_types(self) -> None:
        type_errors: List[str] = []
        if self.mode == 'queries':
            for col in ['query_id', 'input_query', 'output_answer']:
                if col not in self.basket.columns:
                    continue
                non_str = int(
                    self.basket[col].apply(lambda x: not isinstance(x, str)).sum()
                )
                if non_str > 0:
                    type_errors.append(f'{col}: {non_str} не-строковых значений')
        else:
            if 'dialogue' in self.basket.columns:
                non_list = int(
                    self.basket['dialogue']
                    .apply(lambda x: not isinstance(x, list)).sum()
                )
                if non_list > 0:
                    type_errors.append(f'dialogue: {non_list} не-list значений')
        if type_errors:
            self.report.add(CheckResult(
                'required_types', 'FAIL',
                f'Нарушения типов: {type_errors}',
            ))
        else:
            self.report.add(CheckResult(
                'required_types', 'PASS',
                'Типы обязательных колонок корректны',
            ))

    # ---- Проверки режима 'queries' ----

    def _check_query_id_uniqueness(self) -> None:
        if 'query_id' not in self.basket.columns:
            self.report.add(CheckResult(
                'query_id_uniqueness', 'SKIP',
                'Колонка query_id отсутствует',
            ))
            return
        dups = int(self.basket['query_id'].duplicated().sum())
        if dups > 0:
            self.report.add(CheckResult(
                'query_id_uniqueness', 'FAIL',
                f'Найдено {dups} дублирующихся значений query_id',
            ))
        else:
            self.report.add(CheckResult(
                'query_id_uniqueness', 'PASS',
                f'Все {len(self.basket)} значений query_id уникальны',
            ))

    def _check_input_output_nonempty(self) -> None:
        for col in ['input_query', 'output_answer']:
            if col not in self.basket.columns:
                continue
            empty = int(
                (self.basket[col].astype(str).str.strip() == '').sum()
            )
            if empty > 0:
                status: CheckStatus = 'FAIL' if col == 'input_query' else 'WARN'
                self.report.add(CheckResult(
                    f'nonempty_{col}', status,
                    f'{empty} пустых значений в {col}',
                ))
            else:
                self.report.add(CheckResult(
                    f'nonempty_{col}', 'PASS',
                    f'Все значения {col} непусты',
                ))

    # ---- Проверки режима 'dialogue' ----

    def _check_dialogue_structure(self) -> None:
        if 'dialogue' not in self.basket.columns:
            self.report.add(CheckResult(
                'dialogue_structure', 'SKIP',
                'Колонка dialogue отсутствует',
            ))
            return
        bad: List[str] = []
        for idx, val in enumerate(self.basket['dialogue']):
            if not isinstance(val, list):
                bad.append(f'row {idx}: не list')
                continue
            for j, entry in enumerate(val):
                if not isinstance(entry, tuple) or len(entry) != 3:
                    bad.append(f'row {idx} entry {j}: не 3-tuple')
                    break
                if not all(isinstance(x, str) for x in entry):
                    bad.append(f'row {idx} entry {j}: содержит не-строки')
                    break
        if bad:
            self.report.add(CheckResult(
                'dialogue_structure', 'FAIL',
                f'Некорректная структура dialogue в {len(bad)} строках',
                {'sample': bad[:5]},
            ))
        else:
            self.report.add(CheckResult(
                'dialogue_structure', 'PASS',
                'Все dialogue имеют тип list[(str, str, str)]',
            ))

    def _check_dialogue_nonempty(self) -> None:
        if 'dialogue' not in self.basket.columns:
            return
        empty = int(
            self.basket['dialogue']
            .apply(lambda x: not isinstance(x, list) or len(x) == 0).sum()
        )
        if empty > 0:
            self.report.add(CheckResult(
                'dialogue_nonempty', 'FAIL',
                f'{empty} пустых dialogue-списков',
            ))
        else:
            self.report.add(CheckResult(
                'dialogue_nonempty', 'PASS',
                'Все dialogue-списки непусты',
            ))

    def _check_session_id_uniqueness(self) -> None:
        if 'session_id' not in self.basket.columns:
            self.report.add(CheckResult(
                'session_id_uniqueness', 'SKIP',
                'Колонка session_id отсутствует',
            ))
            return
        non_empty = self.basket[self.basket['session_id'].astype(str) != '']
        dups = int(non_empty['session_id'].duplicated().sum())
        if dups > 0:
            self.report.add(CheckResult(
                'session_id_uniqueness', 'FAIL',
                f'{dups} дублирующихся непустых session_id в режиме dialogue',
            ))
        else:
            self.report.add(CheckResult(
                'session_id_uniqueness', 'PASS',
                'Все непустые session_id уникальны',
            ))

    # ---- Проверки соответствия целевому формату ----

    def _check_metric_columns_present(self) -> None:
        metric_cols = [c for c in self.basket.columns if c.endswith(METRIC_SUFFIX)]
        if not metric_cols:
            self.report.add(CheckResult(
                'metric_columns_present', 'WARN',
                'Не найдено ни одной *_metric колонки. Целевой формат требует ≥1. '
                'Метрики ожидаются из validation_data или будут проставлены асессорами.',
            ))
        else:
            self.report.add(CheckResult(
                'metric_columns_present', 'PASS',
                f'Найдено {len(metric_cols)} метрических колонок: {metric_cols}',
                {'columns': metric_cols},
            ))

    def _check_module_field_pairing(self) -> None:
        """Каждый префикс модуля должен иметь и _input_query, и _output_answer."""
        # Извлекаем префиксы, исключая корневые input_query/output_answer
        in_prefixes = set()
        out_prefixes = set()
        for col in self.basket.columns:
            if col == 'input_query' or col == 'output_answer':
                continue
            if col.endswith(MODULE_INPUT_SUFFIX):
                p = col[: -len(MODULE_INPUT_SUFFIX)]
                if p:
                    in_prefixes.add(p)
            elif col.endswith(MODULE_OUTPUT_SUFFIX):
                p = col[: -len(MODULE_OUTPUT_SUFFIX)]
                if p:
                    out_prefixes.add(p)

        unpaired_in = in_prefixes - out_prefixes
        unpaired_out = out_prefixes - in_prefixes

        if unpaired_in or unpaired_out:
            self.report.add(CheckResult(
                'module_field_pairing', 'WARN',
                f'Непарные модульные поля. Без output: {sorted(unpaired_in)}. '
                f'Без input: {sorted(unpaired_out)}',
            ))
        elif in_prefixes:
            self.report.add(CheckResult(
                'module_field_pairing', 'PASS',
                f'Все {len(in_prefixes)} модулей имеют парные input/output: '
                f'{sorted(in_prefixes)}',
            ))
        else:
            self.report.add(CheckResult(
                'module_field_pairing', 'SKIP',
                'Модульные поля отсутствуют',
            ))

    def _check_no_lingering_internal_columns(self) -> None:
        present = FORBIDDEN_INTERNAL_COLUMNS & set(self.basket.columns)
        # session_id - валидное опциональное поле, не считается lingering
        if present:
            self.report.add(CheckResult(
                'no_internal_columns', 'WARN',
                f'В корзину попали внутренние/устаревшие колонки: {present}',
            ))
        else:
            self.report.add(CheckResult(
                'no_internal_columns', 'PASS',
                'Внутренние/устаревшие колонки в выходе отсутствуют',
            ))

    def _check_no_unknown_object_types(self) -> None:
        """В строковых полях не должно быть object-значений, кроме списка диалога."""
        problems: Dict[str, int] = {}
        for col in self.basket.columns:
            if col == 'dialogue':
                continue
            if self.basket[col].dtype != object:
                continue
            non_basic = int(
                self.basket[col]
                .dropna()
                .apply(lambda x: not isinstance(x, (str, int, float, bool)))
                .sum()
            )
            if non_basic > 0:
                problems[col] = non_basic
        if problems:
            self.report.add(CheckResult(
                'no_unknown_object_types', 'WARN',
                f'Колонки с нестандартными object-типами: {problems}',
            ))
        else:
            self.report.add(CheckResult(
                'no_unknown_object_types', 'PASS',
                'Типы значений соответствуют ожиданиям',
            ))

    # ---- Проверки относительно validation_data ----

    def _check_validation_coverage(self) -> None:
        vd = self.validation_data
        assert vd is not None
        if self.mode == 'queries':
            if 'query_id' not in vd.columns or 'query_id' not in self.basket.columns:
                self.report.add(CheckResult(
                    'validation_coverage', 'SKIP',
                    'Нет общего ключа query_id для проверки покрытия',
                ))
                return
            vd_keys = set(vd['query_id'].astype(str))
            basket_keys = set(self.basket['query_id'].astype(str))
            missing = vd_keys - basket_keys
            extra = basket_keys - vd_keys
            if missing:
                self.report.add(CheckResult(
                    'validation_coverage', 'WARN',
                    f'{len(missing)} строк validation_data не нашли пары в корзине '
                    f'(нет соответствующего трейса). Пример: {list(missing)[:3]}',
                    {'missing_count': len(missing), 'extra_count': len(extra)},
                ))
            else:
                self.report.add(CheckResult(
                    'validation_coverage', 'PASS',
                    f'Все {len(vd_keys)} строк validation_data покрыты корзиной',
                ))
        else:  # dialogue
            if 'session_id' not in vd.columns or 'session_id' not in self.basket.columns:
                self.report.add(CheckResult(
                    'validation_coverage', 'SKIP',
                    'Нет общего ключа session_id для dialogue-режима',
                ))
                return
            vd_keys = set(vd['session_id'].astype(str))
            basket_keys = set(self.basket['session_id'].astype(str))
            missing = vd_keys - basket_keys
            if missing:
                self.report.add(CheckResult(
                    'validation_coverage', 'WARN',
                    f'{len(missing)} сессий validation_data не нашли пары в корзине',
                ))
            else:
                self.report.add(CheckResult(
                    'validation_coverage', 'PASS',
                    f'Все {len(vd_keys)} сессий validation_data покрыты корзиной',
                ))

    def _check_metrics_carried_over(self) -> None:
        vd = self.validation_data
        assert vd is not None
        vd_metrics = [c for c in vd.columns if c.endswith(METRIC_SUFFIX)]
        basket_metrics = [c for c in self.basket.columns if c.endswith(METRIC_SUFFIX)]
        missing = set(vd_metrics) - set(basket_metrics)
        if missing:
            self.report.add(CheckResult(
                'metrics_carried_over', 'FAIL',
                f'Метрики из validation_data отсутствуют в корзине: {missing}',
            ))
        elif vd_metrics:
            self.report.add(CheckResult(
                'metrics_carried_over', 'PASS',
                f'Все {len(vd_metrics)} метрик validation_data присутствуют в корзине',
            ))
        else:
            self.report.add(CheckResult(
                'metrics_carried_over', 'SKIP',
                'В validation_data нет метрических колонок',
            ))

    def _check_reference_answers_carried_over(self) -> None:
        vd = self.validation_data
        assert vd is not None
        vd_refs = [c for c in vd.columns if c.startswith(REFERENCE_PREFIX)]
        basket_refs = [c for c in self.basket.columns if c.startswith(REFERENCE_PREFIX)]
        missing = set(vd_refs) - set(basket_refs)
        if missing:
            self.report.add(CheckResult(
                'reference_answers_carried_over', 'FAIL',
                f'Reference-колонки из validation_data отсутствуют в корзине: {missing}',
            ))
        elif vd_refs:
            self.report.add(CheckResult(
                'reference_answers_carried_over', 'PASS',
                f'Все {len(vd_refs)} reference-колонок присутствуют в корзине',
            ))
        else:
            self.report.add(CheckResult(
                'reference_answers_carried_over', 'SKIP',
                'В validation_data нет reference-колонок',
            ))

    # ---- Проверки относительно исходных трейсов ----

    def _check_trace_coverage(self) -> None:
        traces = self.original_traces
        assert traces is not None
        if 'aef_kind' not in traces.columns or 'parent_span_id' not in traces.columns:
            self.report.add(CheckResult(
                'trace_coverage', 'SKIP',
                'Нет колонок aef_kind/parent_span_id для проверки покрытия трейсов',
            ))
            return

        root_mask = (
            (traces['aef_kind'] == 'input_request')
            & (traces['parent_span_id'] == 'root')
        )
        valid_trace_ids = set(traces[root_mask]['trace_id'].astype(str))

        if self.mode == 'queries':
            if 'query_id' not in self.basket.columns:
                self.report.add(CheckResult(
                    'trace_coverage', 'SKIP',
                    "Колонка query_id отсутствует в режиме 'queries'",
                ))
                return
            # Связь query_id (=span_id input_request) -> trace_id
            span_to_trace = dict(zip(
                traces['span_id'].astype(str),
                traces['trace_id'].astype(str),
            ))
            covered = {
                span_to_trace.get(qid)
                for qid in self.basket['query_id'].astype(str)
            } - {None}
            missing = valid_trace_ids - covered
            if missing:
                self.report.add(CheckResult(
                    'trace_coverage', 'WARN',
                    f'{len(missing)} трейсов с input_request не представлены в корзине. '
                    f'Пример: {list(missing)[:3]}',
                    {'missing_count': len(missing),
                     'total_valid': len(valid_trace_ids)},
                ))
            else:
                self.report.add(CheckResult(
                    'trace_coverage', 'PASS',
                    f'Все {len(valid_trace_ids)} валидных трейсов представлены в корзине',
                ))
        else:
            self.report.add(CheckResult(
                'trace_coverage', 'SKIP',
                "Покрытие трейсов проверяется только в режиме 'queries'",
            ))

    def _check_session_id_consistency(self) -> None:
        """Сессии в корзине должны соответствовать session_id из исходных трейсов."""
        traces = self.original_traces
        assert traces is not None
        if 'session_id' not in traces.columns:
            self.report.add(CheckResult(
                'session_id_consistency', 'SKIP',
                'В трейсах нет колонки session_id',
            ))
            return
        traces_sessions = set(
            s for s in traces['session_id'].astype(str).unique() if s != ''
        )
        if 'session_id' not in self.basket.columns:
            self.report.add(CheckResult(
                'session_id_consistency', 'SKIP',
                'В корзине нет колонки session_id',
            ))
            return
        basket_sessions = set(
            s for s in self.basket['session_id'].astype(str).unique() if s != ''
        )
        unknown = basket_sessions - traces_sessions
        if unknown:
            self.report.add(CheckResult(
                'session_id_consistency', 'FAIL',
                f'В корзине есть session_id, отсутствующие в исходных трейсах: '
                f'{list(unknown)[:5]}',
            ))
        else:
            self.report.add(CheckResult(
                'session_id_consistency', 'PASS',
                'Все session_id корзины присутствуют в исходных трейсах',
            ))


# ============================================================================
# Public API
# ============================================================================

def self_check(
    basket: pd.DataFrame,
    mode: Mode,
    validation_data: Optional[pd.DataFrame] = None,
    original_traces: Optional[pd.DataFrame] = None,
    records: Optional[List[Any]] = None,
    raise_on_fail: bool = False,
) -> CheckReport:
    """Прогнать полный набор self-check'ов на корзине.

    Args:
        basket: сгенерированная тестовая корзина.
        mode: режим конвертации, 'queries' или 'dialogue'.
        validation_data: исходная разметка (если есть) - для проверки покрытия.
        original_traces: исходные трейсы (если есть) - для проверки полноты.
        records: список TraceRecord (опционально, для отладочных проверок).
        raise_on_fail: при True вызывает AssertionError при наличии FAIL'ов.

    Returns:
        CheckReport с результатами всех проверок.
    """
    checker = BasketSelfChecker(
        basket=basket,
        mode=mode,
        validation_data=validation_data,
        original_traces=original_traces,
        records=records,
    )
    report = checker.run_all()
    if raise_on_fail and report.has_failures:
        checker.assert_all()
    return report


def convert_and_verify(
    traces: pd.DataFrame,
    validation_data: Optional[pd.DataFrame] = None,
    mode: Mode = 'queries',
    raise_on_fail: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Полный пайплайн: конвертация + self-checks + вывод отчёта.

    Это рекомендуемая точка входа для production-использования.

    Args:
        traces: входные трейсы в формате ТЗ.
        validation_data: размеченная корзина (опционально).
        mode: режим - 'queries' или 'dialogue'.
        raise_on_fail: при True падает с AssertionError при первом же FAIL'е.
        verbose: при True печатает рендеренный отчёт.

    Returns:
        dict со ключами:
            'auto_assessor_df' (pd.DataFrame): сгенерированная корзина
            'report' (CheckReport): отчёт по self-check'ам
            'stats' (dict): статистика конвертации
            'errors' (list): ошибки валидации входной схемы
            'mode' (str): использованный режим
    """
    conv_result = _converter.convert(
        traces=traces,
        validation_data=validation_data,
        mode=mode,
        strict=False,
    )
    basket = conv_result['auto_assessor_df']
    records = conv_result.get('records', [])
    schema_errors = conv_result.get('errors', [])

    if verbose and schema_errors:
        print('--- Schema validation warnings ---')
        for e in schema_errors:
            print(f'  ⚠ {e}')

    report = self_check(
        basket=basket,
        mode=mode,
        validation_data=validation_data,
        original_traces=traces,
        records=records,
        raise_on_fail=False,
    )

    if verbose:
        print(report.render())
        print(f"--- Conversion stats: {conv_result['stats']} ---")

    if raise_on_fail and report.has_failures:
        raise AssertionError(
            'Self-checks failed:\n  '
            + '\n  '.join(f'{c.name}: {c.message}' for c in report.failures())
        )

    return {
        'auto_assessor_df': basket,
        'report': report,
        'stats': conv_result['stats'],
        'errors': schema_errors,
        'mode': mode,
        'skipped_trace_ids': conv_result.get('skipped_trace_ids', []),
        'records': records,
    }
