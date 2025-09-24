from __future__ import annotations

import os

from vibevoice_api.voice_map import VoiceMapper


def _touch(path: os.PathLike[str] | str) -> None:
    with open(path, "wb") as f:
        f.write(b"voice")


def test_voice_map_directories_and_prefixes(tmp_path) -> None:
    root = tmp_path

    demo_dir = root / "demo" / "voices"
    demo_dir.mkdir(parents=True)
    _touch(demo_dir / "base.wav")

    custom_dir = root / "custom_dir"
    custom_dir.mkdir()
    _touch(custom_dir / "hero.wav")

    nested_dir = root / "more" / "nested"
    nested_dir.mkdir(parents=True)
    _touch(nested_dir / "beta.wav")

    alt_dir = root / "more_exact"
    alt_dir.mkdir()
    _touch(alt_dir / "gamma.wav")

    (root / "voice_map.yaml").write_text(
        "\n".join(
            [
                "aliases:",
                "  alias_base: base",
                "directories:",
                "  - custom_dir",
                "  - path: more",
                "    prefix: promo_",
                "    recursive: true",
                "  - path: more_exact",
                "    prefix: alt_",
                "    normalize: true",
            ]
        ),
        encoding="utf-8",
    )

    mapper = VoiceMapper(str(root))
    available = mapper.available()

    assert available["hero"] == str(custom_dir / "hero.wav")
    assert available["promo_beta"] == str(nested_dir / "beta.wav")
    assert "beta" not in available
    assert available["alt_gamma"] == str(alt_dir / "gamma.wav")
    assert available["gamma"] == str(alt_dir / "gamma.wav")
    assert mapper.resolve("alias_base") == str(demo_dir / "base.wav")
