from models import get_models
from models.get_models import weights_init


def get_model(config=None,export=False,cfg=None,num_class=78):
    modelName = config.MODEL.NAME
    class_name = getattr(get_models, modelName)


    model = class_name(num_classes=num_class,export=export,cfg=cfg)
    model.apply(weights_init)
    return model