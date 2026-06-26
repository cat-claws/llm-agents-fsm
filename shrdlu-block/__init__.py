"""OpenAI-compatible SHRDLU agents."""

from importlib import import_module

__all__ = [
    'DEFAULT_MAX_STEPS',
    'DEFAULT_OPENAI_API_KEY',
    'DEFAULT_OPENAI_BASE_URL',
    'DEFAULT_OPENAI_MODEL',
    'DEFAULT_TRACE_DIR',
    'OpenAICompatibleShrdluAgent',
    'PredictivePreplannedOpenAICompatibleShrdluAgent',
    'PreplannedOpenAICompatibleShrdluAgent',
    'SuffixPredictivePreplannedOpenAICompatibleShrdluAgent',
]

_EXPORT_MODULES = {
    'DEFAULT_MAX_STEPS': 'shrdlu_agents.shrdlu_agent_basic',
    'DEFAULT_OPENAI_API_KEY': 'shrdlu_agents.shrdlu_agent_basic',
    'DEFAULT_OPENAI_BASE_URL': 'shrdlu_agents.shrdlu_agent_basic',
    'DEFAULT_OPENAI_MODEL': 'shrdlu_agents.shrdlu_agent_basic',
    'DEFAULT_TRACE_DIR': 'shrdlu_agents.shrdlu_agent_basic',
    'OpenAICompatibleShrdluAgent': 'shrdlu_agents.shrdlu_agent_basic',
    'PreplannedOpenAICompatibleShrdluAgent': 'shrdlu_agents.shrdlu_agent_basic',
    'PredictivePreplannedOpenAICompatibleShrdluAgent': 'shrdlu_agents.shrdlu_agent_plan',
    'SuffixPredictivePreplannedOpenAICompatibleShrdluAgent': 'shrdlu_agents.shrdlu_agent_fsm',
}


def __getattr__(name):
    if name not in _EXPORT_MODULES:
        raise AttributeError("module 'shrdlu_agents' has no attribute %r" % name)
    module = import_module(_EXPORT_MODULES[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
