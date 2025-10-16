import importlib
import pytorch_lightning as pyl

MODEL_REGISTRY = {
    "NeuralInnerProduct": "recsys.models.neural_inner_product",
    "NeuralColabFilter": "recsys.models.neural_colab_filter",
    "MultVAE": "recsys.models.mult_vae",
    "RQVAE": "recsys.models.rq_vae",
    "DCN": "recsys.models.dcn",
    "DCNV2": "recsys.models.dcnv2",
    "LightGCN": "recsys.models.lightgcn",
    "WideAndDeep": "recsys.models.wide_and_deep",
    "AutoInt": "recsys.models.autoint",
    "FactorizationMachines": "recsys.models.factorization_machines",
    "HybridModel": "recsys.models.hybrid_model",
}


def get_model(architecture: str, **kwargs) -> pyl.LightningModule:
    """Dynamically loads the specified model."""
    if architecture not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {architecture}")

    module_path = MODEL_REGISTRY[architecture]
    module = importlib.import_module(module_path)
    model_class = getattr(module, "Model")  # Ensure each module has a 'Model' class
    return model_class(**kwargs)
