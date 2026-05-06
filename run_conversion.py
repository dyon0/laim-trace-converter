"""
Demo / entry-point script for LAIM Trace-to-Basket converter.

Запускается как:
    python run_conversion.py <traces.parquet> [--validation <validation.parquet>]
                              [--mode queries|dialogue]
                              [--output <output.parquet>]
                              [--strict]

Полный пайплайн:
    1. Загрузка трейсов из parquet (формат ТЗ).
    2. Опциональная загрузка validation_data (csv/parquet).
    3. Конвертация в формат тестовой корзины.
    4. Прогон self-check'ов.
    5. Сохранение результата (опционально) и вывод отчёта.

Без аргументов работает в demo-режиме на example_spans.parquet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from laim_converter import convert_and_verify


def _load_table(path: str) -> pd.DataFrame:
    """Загрузить DataFrame из parquet/csv по расширению."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Файл не найден: {path}')
    suffix = p.suffix.lower()
    if suffix == '.parquet':
        return pd.read_parquet(p)
    if suffix == '.csv':
        return pd.read_csv(p)
    if suffix in ('.xlsx', '.xls'):
        return pd.read_excel(p)
    raise ValueError(f'Неподдерживаемый формат: {suffix}')


def _save_basket(basket: pd.DataFrame, path: str) -> None:
    """Сохранить корзину в parquet/csv по расширению."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == '.parquet':
        # dialogue (list of tuples) сериализуется в parquet через pyarrow
        if 'dialogue' in basket.columns:
            # Преобразуем tuple -> list для совместимости с parquet
            basket = basket.copy()
            basket['dialogue'] = basket['dialogue'].apply(
                lambda lst: [list(t) for t in lst] if isinstance(lst, list) else lst
            )
        basket.to_parquet(p, index=False)
    elif suffix == '.csv':
        basket.to_csv(p, index=False)
    elif suffix in ('.xlsx',):
        basket.to_excel(p, index=False)
    else:
        raise ValueError(f'Неподдерживаемый формат вывода: {suffix}')


def run(
    traces_path: str,
    validation_path: Optional[str] = None,
    mode: str = 'queries',
    output_path: Optional[str] = None,
    strict: bool = False,
) -> int:
    """Основной точка входа. Возвращает 0 при успехе, 1 при FAIL'ах."""
    print(f'Loading traces from: {traces_path}')
    traces = _load_table(traces_path)
    print(
        f'  -> {len(traces)} spans across '
        f"{traces['trace_id'].nunique() if 'trace_id' in traces.columns else 0} traces"
    )

    validation_data = None
    if validation_path:
        print(f'Loading validation_data from: {validation_path}')
        validation_data = _load_table(validation_path)
        print(f'  -> {len(validation_data)} rows, columns: {list(validation_data.columns)}')

    print(f'\nRunning conversion in mode={mode!r}...\n')
    result = convert_and_verify(
        traces=traces,
        validation_data=validation_data,
        mode=mode,  # type: ignore[arg-type]
        raise_on_fail=False,
        verbose=True,
    )

    basket = result['auto_assessor_df']
    report = result['report']

    print(f'\nResulting basket: shape={basket.shape}, columns={list(basket.columns)}')

    if output_path and len(basket) > 0:
        _save_basket(basket, output_path)
        print(f'\nSaved basket to: {output_path}')

    if strict and report.has_failures:
        print('\nStrict mode: FAILed checks present, exiting with code 1', file=sys.stderr)
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Convert LAIM telemetry traces to test basket format.',
    )
    parser.add_argument(
        'traces',
        nargs='?',
        default='/mnt/user-data/uploads/example_spans.parquet',
        help='Путь к parquet/csv с трейсами в формате ТЗ '
             '(default: example_spans.parquet)',
    )
    parser.add_argument(
        '--validation',
        default=None,
        help='Путь к parquet/csv с разметкой (validation_data)',
    )
    parser.add_argument(
        '--mode',
        choices=['queries', 'dialogue'],
        default='queries',
        help="Режим конвертации (default: 'queries')",
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Путь для сохранения корзины (parquet/csv)',
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Завершать с ненулевым кодом при FAIL self-check',
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    return run(
        traces_path=args.traces,
        validation_path=args.validation,
        mode=args.mode,
        output_path=args.output,
        strict=args.strict,
    )


if __name__ == '__main__':
    sys.exit(main())
