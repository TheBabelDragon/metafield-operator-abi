"""FPGA backend placeholder — empty on purpose until first D_W device pass."""


class FPGABackend:
    name = "fpga"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "FPGABackend is provisional. Implement wilson_dirac first; pass L2 goldens."
        )
