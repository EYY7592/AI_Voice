"""ScamLens-TW 執行階段例外。"""


class ScamLensError(Exception):
    pass


class AudioLoadError(ScamLensError):
    pass


class AudioDenoiseError(ScamLensError):
    pass


class WhisperModelError(ScamLensError):
    pass
