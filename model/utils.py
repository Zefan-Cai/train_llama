
from transformers import (
    BertModel,
    RobertaModel,
    AlbertModel,
    DebertaV2Model,
    XLNetModel,
    DebertaV2Model,
    AutoConfig,
    LlamaForCausalLM
)


MODEL_CLASS = {
    "llama": LlamaForCausalLM
}


def get_model(model_args, config: AutoConfig, fix_model: bool = False):

    model_class = MODEL_CLASS[config.model_type]
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        config=config
    )

    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
    if fix_model:
        training_param = 0
        for param in model.parameters():
            param.requires_grad = False

        for param in model.layers[-3:].parameters():
            param.requires_grad = True
            training_param+=param.numel()

    print('***** total param is {} *****'.format(all_param))
    if fix_model:
        all_param = training_param
    print('***** training param is {} *****'.format(all_param))
    return model
