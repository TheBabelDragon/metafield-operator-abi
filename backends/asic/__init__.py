"""ASIC backend placeholder — atomic operators only, never RUN_HMC."""


class ASICBackend:
    name = "asic"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ASICBackend exposes WILSON_DIRAC / DOT / NORM / … atoms only."
        )
