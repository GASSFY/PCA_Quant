from Sublink.models.llava_onevision.llava_onevision import LLaVA_onevision  # noqa: F401
from Sublink.models.llava_v15.llava_v15 import LLaVA_v15  # noqa: F401
from Sublink.models.internVL2.internvl2 import InternVL2  # noqa: F401
from Sublink.utils.registry import MODEL_REGISTRY


def get_process_model(model_name: str):
    return MODEL_REGISTRY[model_name]


__all__ = ["get_process_model", "LLaVA_onevision", "LLaVA_v15", "InternVL2"]
