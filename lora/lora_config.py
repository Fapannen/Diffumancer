class LoraConfig:
    """Class for definition of LORA adapter. Simplifies the
    handling of LORAs in the pipelines.
    """

    def __init__(
        self,
        lora_path: str,
        weights_file: str | None,
        triggerword: str | None,
        name: str,
        weight: float = 0.5,
    ):
        self.lora_path = lora_path
        self.weights_file = weights_file if weights_file is not None else self.lora_path
        self.name = name
        self.triggerword = triggerword
        self.weight = weight
