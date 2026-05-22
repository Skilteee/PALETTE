# Re-export for backward compatibility
from utils.utils import evaluate, NativeScalerWithGradNormCount, create_logger
from utils.hook_utils import (
    add_hooks,
    get_direction_ablation_input_pre_hook,
    get_direction_ablation_output_hook,
    get_all_direction_ablation_hooks,
    get_directional_patching_input_pre_hook,
    get_activation_addition_input_pre_hook,
)
