"""The package must import and be usable on a machine with no GPU."""
import canopi


def test_top_level_api():
    for name in ("SceneConfig", "render", "render_backward_image",
                 "render_backward_image_deriv", "RenderResult"):
        assert hasattr(canopi, name), name
    assert isinstance(canopi.__version__, str)


def test_gpu_module_imports_without_cuda():
    # The import-time CUDA probe must degrade to a flag, never raise.
    from canopi.gpu import GPU_AVAILABLE
    assert isinstance(GPU_AVAILABLE, bool)


def test_submodules_import():
    import canopi.derivative      # noqa: F401
    import canopi.geometry        # noqa: F401
    import canopi.intersection    # noqa: F401
    import canopi.lad             # noqa: F401
    import canopi.paths           # noqa: F401
    import canopi.path_tracer     # noqa: F401
    import canopi.scattering      # noqa: F401
    import canopi.scene           # noqa: F401
    import canopi.scenes.jbs_loader          # noqa: F401
    import canopi.scenes.jbs_species_loader  # noqa: F401
