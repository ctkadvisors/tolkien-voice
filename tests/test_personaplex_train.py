from types import SimpleNamespace

from tolkien_voice.training.personaplex_train import (
    install_reiterable_eval_loader_patch,
    load_personaplex_mimi,
    mimi_codebooks_for_config,
)


def test_personaplex_uses_eight_mimi_codebooks_per_stereo_channel() -> None:
    assert mimi_codebooks_for_config({"model_type": "personaplex", "dep_q": 16}) == 8


def test_non_personaplex_uses_upstream_codebook_rule() -> None:
    assert mimi_codebooks_for_config({"version": "moshiko", "n_q": 16, "dep_q": 8}) == 8
    assert mimi_codebooks_for_config({"version": "other", "n_q": 16, "dep_q": 16}) == 16


def test_load_personaplex_mimi_uses_pinned_moshi_api() -> None:
    calls: list[tuple[object, int, str]] = []

    class FakeLoaders:
        @staticmethod
        def get_mimi(filename, *, num_codebooks: int, device: str):
            calls.append((filename, num_codebooks, device))
            return "mimi-model"

    checkpoint_info = SimpleNamespace(
        mimi_weights="mimi.safetensors",
        raw_config={"model_type": "personaplex", "n_q": 16, "dep_q": 16},
    )

    result = load_personaplex_mimi(FakeLoaders, checkpoint_info, "cuda")

    assert result == "mimi-model"
    assert calls == [("mimi.safetensors", 8, "cuda")]


def test_each_checkpoint_gets_a_fresh_finite_eval_loader() -> None:
    calls: list[bool] = []

    def build_data_loader(*_args, **kwargs):
        is_eval = kwargs["is_eval"]
        calls.append(is_eval)
        return iter([len(calls)])

    upstream_train = SimpleNamespace(build_data_loader=build_data_loader)
    install_reiterable_eval_loader_patch(upstream_train)

    train_loader = upstream_train.build_data_loader(is_eval=False)
    eval_loader = upstream_train.build_data_loader(is_eval=True)

    assert list(train_loader) == [1]
    assert list(eval_loader) == [2]
    assert list(eval_loader) == [3]
    assert calls == [False, True, True]
