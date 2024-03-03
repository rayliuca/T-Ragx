from .InternLM2Model import InternLM2Model
from .LlamaCppPythonModel import LlamaCppPythonModel
from .MistralModel import MistralModel
from .OllamaModel import OllamaModel
from .OpenAIModel import OpenAIModel

"""Import all modules that exist in the current directory."""
# Ref https://stackoverflow.com/a/60861023/
from importlib import import_module
from pathlib import Path

__all__ = ["MistralModel", "InternLM2Model", "OllamaModel", "OpenAIModel", "LlamaCppPythonModel"]

for f in Path(__file__).parent.glob("*.py"):
    module_name = f.stem
    if (not module_name.startswith("_")) and (module_name not in globals()):
        import_module(f".{module_name}", __package__)
        __all__.append(module_name)
    del f, module_name
del import_module, Path
