"""GPU/CPU UMAP detection utilities."""


def get_umap_class(prefer_gpu: bool = True):
    """Return the best available UMAP class.

    Parameters
    ----------
    prefer_gpu : bool
        If True, try to import cuML's GPU UMAP first.

    Returns
    -------
    UMAP class
        Either ``cuml.UMAP`` or ``umap.UMAP``.
    """
    if prefer_gpu:
        try:
            from cuml import UMAP as GPU_UMAP
            return GPU_UMAP
        except ImportError:
            pass

    import umap as cpu_umap_module
    return cpu_umap_module.UMAP
