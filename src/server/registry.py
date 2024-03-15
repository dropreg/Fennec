class ServerRegistry:
    _registry = {}

    @classmethod
    def register_class(cls, class_name, registered_class):
        if class_name in cls._registry:
            raise ValueError(f"duplicated registry: {class_name}")
        cls._registry[class_name] = registered_class

    @classmethod
    def get_class(cls, class_name):
        return cls._registry.get(class_name, None)

    @classmethod
    def create_instance(cls, class_name, *args, **kwargs):
        target_class = cls.get_class(class_name)
        if not target_class:
            raise ValueError(f"not support create instance for : {class_name}")
        return target_class(*args, **kwargs)


def auto_register(class_name):
    def inner(cls):
        ServerRegistry.register_class(class_name, cls)
        return cls

    return inner
