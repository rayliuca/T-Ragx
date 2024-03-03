class DummyTokenizer:
    eos_token_id = None
    eos_token = None

    pad_token_id = None
    pad_token = None

    unk_token_id = None
    unk_token = None

    def __init__(self, *args, **kwargs):
        pass

    def batch_decode(self, *args, **kwargs):
        raise NotImplementedError

    def batch_encode_plus(self, *args, **kwargs):
        raise NotImplementedError

    def apply_chat_template(self, conversation, *args, **kwargs):
        return conversation[-1]['content']
