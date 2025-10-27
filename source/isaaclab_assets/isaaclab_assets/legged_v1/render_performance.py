import isaaclab.sim as sim_utils

def set_performance_render_settings(
    rendering_mode: str = "performance",
    carb_settings: dict | None = None,
):
    """
    Configure and apply custom rendering settings in Isaac Lab.

    Args:
        rendering_mode (str): One of {"performance", "balanced", "quality"}.
            Determines base rendering preset.
        carb_settings (dict, optional): RTX carb settings to override default
            rendering mode presets. Example:
                {"rtx.reflections.enabled": True}

    Returns:
        sim_utils.RenderCfg: Configured RenderCfg instance.
    """
    # Default override settings (if none provided)
    if carb_settings is None:
        carb_settings = {
            "rtx.reflections.enabled": False,    # enable reflections
            "rtx.translucency.enabled": False,  # disable translucency for speed
            "rtx.directLighting.enabled": False  # ensure direct lighting on
        }

    # Initialize render configuration
    render_cfg = sim_utils.RenderCfg(
        rendering_mode=rendering_mode,
        carb_settings=carb_settings,
    )

    return render_cfg
