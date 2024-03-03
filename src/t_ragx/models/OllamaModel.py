from .API_Model import APIModel


class OllamaModel(APIModel):
    def __init__(self, host='localhost', port=11434, endpoint='/api/generate', model="t_ragx_mistral",
                 protocol="http"):
        super().__init__(host=host, port=port, endpoint=endpoint, model=model,
                         protocol=protocol)
