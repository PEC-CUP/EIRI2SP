"""Validate the shape and scope of the public EIRI2SP data package."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main() -> None:
    library = pd.read_csv(DATA / "library" / "compound_library.csv")
    fingerprints = pd.read_csv(DATA / "fingerprints" / "fingerprints_47bit.csv")
    properties = pd.read_csv(DATA / "properties" / "property_labels_8.csv")
    selections = pd.read_csv(DATA / "mixtures" / "virtual_mixture_selections.csv")

    assert len(library) == 5227, len(library)
    assert len(fingerprints) == 5227, len(fingerprints)
    assert len(properties) == 5227, len(properties)
    assert fingerprints.shape[1] == 49, fingerprints.shape
    assert library["split"].value_counts().to_dict() == {"train": 4181, "test": 1046}
    assert selections["identifier"].nunique() == 2392
    assert not any(c.startswith("mz_") for c in library.columns)
    assert not any(c.startswith("mz_") for c in fingerprints.columns)
    assert not any(c.startswith("mz_") for c in properties.columns)

    print("public data validation passed")
    print("molecules:", len(library))
    print("fingerprint columns:", fingerprints.shape[1] - 2)
    print("split:", library["split"].value_counts().to_dict())
    print("virtual-mixture compounds:", selections["identifier"].nunique())


if __name__ == "__main__":
    main()
