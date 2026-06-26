"""OpenAI-compatible SHRDLU agents."""

from importlib import import_module

__all__ = [
    'DEFAULT_MAX_STEPS',
    'DEFAULT_OPENAI_API_KEY',
    'DEFAULT_OPENAI_BASE_URL',
    'DEFAULT_OPENAI_MODEL',
    'DEFAULT_TRACE_DIR',
    'FsmOpenAICompatibleShrdluAgent',
    'OpenAICompatibleShrdluAgent',
]

_EXPORT_MODULES = {
    'DEFAULT_MAX_STEPS': 'shrdlu_agents.shrdlu_agent_basic',
    'DEFAULT_OPENAI_API_KEY': 'shrdlu_agents.shrdlu_agent_basic',
    'DEFAULT_OPENAI_BASE_URL': 'shrdlu_agents.shrdlu_agent_basic',
    'DEFAULT_OPENAI_MODEL': 'shrdlu_agents.shrdlu_agent_basic',
    'DEFAULT_TRACE_DIR': 'shrdlu_agents.shrdlu_agent_basic',
    'FsmOpenAICompatibleShrdluAgent': 'shrdlu_agents.shrdlu_agent_fsm',
    'OpenAICompatibleShrdluAgent': 'shrdlu_agents.shrdlu_agent_basic',
}


def __getattr__(name):
    if name not in _EXPORT_MODULES:
        raise AttributeError("module 'shrdlu_agents' has no attribute %r" % name)
    module = import_module(_EXPORT_MODULES[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
