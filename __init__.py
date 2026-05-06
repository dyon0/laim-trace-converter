"""LAIM Trace-to-Basket conversion package."""

from .converter import (
    convert,
    main,
    extract_trace_record,
    extract_user_query,
    extract_agent_answer,
    extract_llm_messages,
    validate_input_schema,
    TraceRecord,
    AEF_KINDS,
    REQUIRED_TRACE_COLUMNS,
)

from .self_checks import (
    self_check,
    convert_and_verify,
    BasketSelfChecker,
    CheckReport,
    CheckResult,
)

__all__ = [
    'convert',
    'main',
    'convert_and_verify',
    'self_check',
    'extract_trace_record',
    'extract_user_query',
    'extract_agent_answer',
    'extract_llm_messages',
    'validate_input_schema',
    'TraceRecord',
    'BasketSelfChecker',
    'CheckReport',
    'CheckResult',
    'AEF_KINDS',
    'REQUIRED_TRACE_COLUMNS',
]
