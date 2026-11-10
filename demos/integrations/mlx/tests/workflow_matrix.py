"""Expand a GitHub Actions ``strategy.matrix`` the way the runner service does.

The axes form a Cartesian product, ``exclude`` removes the combinations it
matches, and each ``include`` entry is then merged into every combination none
of whose original axis values it would change. A merged entry may overwrite
values an earlier ``include`` entry added, and an entry that fits no
combination becomes a combination of its own. So several entries that set the
same added key, with no axis to tell them apart, leave only the last one's
values in every combination. Combinations that ``include`` itself added take
no later entries.
"""

from itertools import product

RESERVED = ("include", "exclude")


def expand_matrix(matrix):
    """The list of combinations ``matrix`` runs, in GitHub's order."""
    axes = [key for key in matrix if key not in RESERVED]
    for key in axes:
        if not isinstance(matrix[key], list):
            raise ValueError(f"Matrix axis {key} is not a literal list")
    # A matrix of include entries alone runs each entry as written.
    combinations = (
        [dict(zip(axes, values)) for values in product(*(matrix[key] for key in axes))]
        if axes
        else []
    )
    for excluded in matrix.get("exclude") or ():
        combinations = [
            combination
            for combination in combinations
            if any(combination.get(key) != value for key, value in excluded.items())
        ]
    expanded = [dict(combination) for combination in combinations]
    added = []
    for entry in matrix.get("include") or ():
        if not isinstance(entry, dict):
            raise ValueError("Matrix include entries must be literal mappings")
        merged = False
        # Only the axes' combinations take entries; added ones stay as written.
        for original, combination in zip(combinations, expanded):
            if all(original[key] == entry[key] for key in entry if key in original):
                combination.update(entry)
                merged = True
        if not merged:
            added.append(dict(entry))
    return expanded + added


def unrealised_includes(matrix):
    """The ``include`` entries no expanded combination carries in full."""
    expanded = expand_matrix(matrix)
    return [
        entry
        for entry in matrix.get("include") or ()
        if not any(
            all(combination.get(key) == value for key, value in entry.items())
            for combination in expanded
        )
    ]
