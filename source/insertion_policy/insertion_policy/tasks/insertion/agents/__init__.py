"""Agent configs for the insertion task.

Importing this package registers the custom ``insertion_hybrid`` rl_games network (CNN image branch
+ proprio vector) so it is available before any Runner builds a model.
"""

from . import insertion_hybrid_network  # noqa: F401  (runs model_builder.register_network)
