from tolkien_voice.training.personaplex_train import mimi_codebooks_for_config


def test_personaplex_uses_eight_mimi_codebooks_per_stereo_channel() -> None:
    assert mimi_codebooks_for_config({"version": "personaplex-7b-v1", "dep_q": 16}) == 8


def test_non_personaplex_uses_upstream_codebook_rule() -> None:
    assert mimi_codebooks_for_config({"version": "moshiko", "n_q": 16, "dep_q": 8}) == 8
    assert mimi_codebooks_for_config({"version": "other", "n_q": 16, "dep_q": 16}) == 16
