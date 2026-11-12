from scripts.train import ROOT, portable_metadata


def test_portable_metadata_removes_workspace_from_git_root_and_nested_paths():
    original = {"git": {"root": str(ROOT), "commit": "abc"},
                "paths": [ROOT / "data" / "train.txt", str(ROOT / "runs" / "train")]}
    assert portable_metadata(original) == {
        "git": {"root": ".", "commit": "abc"},
        "paths": ["data/train.txt", "runs/train"],
    }
    assert original["git"]["root"] == str(ROOT)
