import queue


class FrozenManager:
    """
    FrozenManager is used to manage ZeroPP parameters gathering and gradient calculation.
    """

    cached_wp_params = {}

    @classmethod
    def cache_full_wp_parameters(cls, shard_param, full_param):
        cls.cached_wp_params[shard_param] = full_param

    @classmethod
    def clear_cached_wp_parameters(cls):
        for shard_param, full_param in cls.cached_wp_params.items():
            del full_param
            cls.cached_wp_params[shard_param] = None

    @classmethod
    def check_postpond_grad_accum(cls, param):
        return param in cls.cached_wp_params