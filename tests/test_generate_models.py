"""Tests for popai.generate_models.ModelBuilder (step 1 of the workflow).

Configs are built directly in code, so these tests exercise model generation
on its own, without alignments, popfiles or the config parser. Migration timing
is checked with msprime's DemographyDebugger, i.e. against what each model
*means*, not how the code happens to build it.

Run with:  python -m pytest tests/test_generate_models.py
"""

import io
import itertools
import logging

import dendropy
import msprime
import numpy as np
import pandas as pd
import pytest

from popai.generate_models import ModelBuilder

logging.disable(logging.INFO)

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

NE = (10000, 50000)
DIV_AB = (10000, 50000)
DIV_ABC = (100000, 500000)
MIG = (1e-5, 1e-4)

TREE3 = ("((A[&ne=10000-50000],B[&ne=10000-50000])AB[&ne=10000-50000,div=10000-50000],"
         "C[&ne=10000-50000])ABC[&ne=10000-50000,div=100000-500000];")
# Same tree as TREE3, with the children written in a different order.
TREE3_SWAPPED = ("(C[&ne=10000-50000],(B[&ne=10000-50000],A[&ne=10000-50000])"
                 "AB[&ne=10000-50000,div=10000-50000])ABC[&ne=10000-50000,div=100000-500000];")
# A genuinely different topology on the same tips.
TREE3_AC = ("((A[&ne=10000-50000],C[&ne=10000-50000])AC[&ne=10000-50000,div=10000-50000],"
            "B[&ne=10000-50000])ABC[&ne=10000-50000,div=100000-500000];")
TREE5 = ("(((A[&ne=10000-50000],B[&ne=10000-50000])AB[&ne=10000-50000,div=1000-5000],"
         "C[&ne=10000-50000])ABC[&ne=10000-50000,div=10000-15000],(D[&ne=10000-50000],"
         "E[&ne=10000-50000])DE[&ne=10000-50000,div=1000-5000])ABCDE[&ne=10000-50000,div=100000-200000];")


def caterpillar(n_tips):
    newick = "T0[&ne=1000-2000]"
    for i in range(1, n_tips):
        newick = (f"({newick},T{i}[&ne=1000-2000])N{i}"
                  f"[&ne=1000-2000,div={i * 1000}-{i * 1000 + 500}]")
    return newick + ";"


def balanced(n_tips):
    counter = itertools.count(1)

    def rec(lo, hi, depth):
        if hi - lo == 1:
            return f"T{lo}[&ne=1000-2000]"
        label = next(counter)
        mid = (lo + hi) // 2
        div = (20 - depth) * 1000
        return (f"({rec(lo, mid, depth + 1)},{rec(mid, hi, depth + 1)})N{label}"
                f"[&ne=1000-2000,div={div}-{div + 500}]")

    return rec(0, n_tips, 0) + ";"


def tree_list(*newicks):
    body = "".join(f"Tree t{i}={nwk}\n" for i, nwk in enumerate(newicks))
    return dendropy.TreeList.get(data=f"#NEXUS\nBEGIN TREES;\n{body}END;", schema="nexus")


def migration_matrix(pops, allowed=None):
    """All pairs allowed by default; otherwise only the (unordered) pairs listed."""
    allowed = None if allowed is None else {frozenset(p) for p in allowed}
    rows = []
    for a in pops:
        row = []
        for b in pops:
            ok = a != b and (allowed is None or frozenset((a, b)) in allowed)
            row.append("T" if ok else "F")
        rows.append(f"{a}," + ",".join(row))
    return pd.read_csv(io.StringIO("," + ",".join(pops) + "\n" + "\n".join(rows)), index_col=0)


def all_populations(newick):
    tree = tree_list(newick)[0]
    return [str(n.taxon.label if n.is_leaf() else n.label).strip("'")
            for n in tree.postorder_node_iter()]


def make_config(trees=(TREE3,), migration=None, **overrides):
    config = {
        "species tree": tree_list(*trees),
        "migration df": [migration if migration is not None
                         else migration_matrix(all_populations(t)) for t in trees],
        "seed": 1234,
        "replicates": 5,
        "symmetric": True,
        "constant Ne": True,
        "secondary contact": True,
        "divergence with gene flow": False,
        "max migration events": 2,
        "migration rate": list(MIG),
    }
    config.update(overrides)
    return config


def build(config):
    builder = ModelBuilder(config)
    div, sc, dwg = builder.build_models()
    models, labels, sp_index = builder.draw_parameters(div, sc, dwg)
    return builder, (div, sc, dwg), models, labels, sp_index


# ---------------------------------------------------------------------------
# Helpers for inspecting drawn models
# ---------------------------------------------------------------------------

def split_times(model):
    return {e.ancestral: e.time for e in model.events if hasattr(e, "ancestral")}


def present_day_populations(model):
    ancestral = {e.ancestral for e in model.events if hasattr(e, "ancestral")}
    return [p.name for p in model.populations if p.name not in ancestral]


def migration_epochs(model):
    """Yield (start, end, source, dest, rate) for every epoch with migration."""
    names = [p.name for p in model.populations]
    for epoch in model.debug().epochs:
        matrix = epoch.migration_matrix
        for j, k in zip(*np.nonzero(matrix)):
            yield epoch.start_time, epoch.end_time, names[j], names[k], matrix[j, k]


def fingerprint(model):
    return (tuple(p.initial_size for p in model.populations),
            tuple((type(e).__name__, e.time, getattr(e, "rate", None)) for e in model.events),
            model.migration_matrix.tobytes())


def kind_of(model):
    """Classify a drawn model by when its migration happens."""
    epochs = list(migration_epochs(model))
    if not epochs:
        return "divergence"
    return "secondary contact" if min(start for start, *_ in epochs) == 0 else "gene flow"


# ---------------------------------------------------------------------------
# 1. Which models get enumerated
# ---------------------------------------------------------------------------

def test_divergence_models_for_three_tips():
    _, (div, _, _), *_ = build(make_config(**{"secondary contact": False}))
    population_sets = {frozenset(p.name for p in d.populations) for d in div[0]}
    assert population_sets == {
        frozenset({"A", "B", "C", "AB", "ABC"}),  # full tree
        frozenset({"AB", "C", "ABC"}),            # A and B collapsed
        frozenset({"ABC"}),                       # panmixia
    }


@pytest.mark.parametrize("n_tips", [3, 4, 5, 6, 8])
def test_caterpillar_tree_has_one_model_per_tip(n_tips):
    tree = caterpillar(n_tips)
    config = make_config(trees=(tree,), **{"secondary contact": False})
    div, _, _ = ModelBuilder(config).build_models()
    assert len(div[0]) == n_tips


@pytest.mark.parametrize("n_tips,expected", [(4, 5), (8, 26), (16, 677)])
def test_balanced_tree_model_count(n_tips, expected):
    # Valid collapse sets satisfy f(node) = 1 + product of f(children), f(tip) = 1.
    tree = balanced(n_tips)
    config = make_config(trees=(tree,), **{"secondary contact": False})
    div, _, _ = ModelBuilder(config).build_models()
    assert len(div[0]) == expected


@pytest.mark.parametrize("symmetric,max_events,sc_expected,dwg_expected", [
    # full tree: contemporary pairs A-B, A-C, B-C; sister pairs A-B, AB-C
    # A,B collapsed: contemporary pair AB-C; sister pair AB-C
    (True, 2, 3 + 3 + 1, 2 + 1 + 1),
    (True, 1, 3 + 1, 2 + 1),
    (False, 1, 6 + 2, 4 + 2),
])
def test_migration_model_counts(symmetric, max_events, sc_expected, dwg_expected):
    config = make_config(symmetric=symmetric, **{
        "max migration events": max_events, "divergence with gene flow": True})
    _, sc, dwg = ModelBuilder(config).build_models()
    assert len(sc[0]) == sc_expected
    assert len(dwg[0]) == dwg_expected


def test_migration_matrix_limits_which_pairs_get_models():
    mig = migration_matrix(all_populations(TREE3), allowed=[("A", "C")])
    config = make_config(migration=mig, **{"divergence with gene flow": True})
    _, sc, dwg = ModelBuilder(config).build_models()
    assert len(sc[0]) == 1    # only A<->C, only in the full tree
    assert len(dwg[0]) == 0   # A and C are not sisters


# ---------------------------------------------------------------------------
# 2. Drawing parameters: every option combination works
# ---------------------------------------------------------------------------

OPTION_COMBOS = [
    pytest.param(sym, const, sc, dwg, id=f"sym={sym}-constNe={const}-sc={sc}-dwg={dwg}")
    for sym, const, (sc, dwg) in itertools.product(
        [True, False], [True, False], [(True, False), (False, True), (True, True)])
]


@pytest.mark.parametrize("symmetric,constant_ne,sc,dwg", OPTION_COMBOS)
def test_draw_parameters_shapes_and_labels(symmetric, constant_ne, sc, dwg):
    reps = 4
    config = make_config(replicates=reps, symmetric=symmetric, **{
        "constant Ne": constant_ne, "secondary contact": sc, "divergence with gene flow": dwg})
    _, (div, scm, dwgm), models, labels, sp_index = build(config)

    n_models = len(div[0]) + len(scm[0]) + len(dwgm[0])
    assert len(models) == len(labels) == len(sp_index) == n_models * reps
    assert sorted(set(labels)) == list(range(n_models))
    assert all(labels.count(k) == reps for k in range(n_models))
    assert set(sp_index) == {0}


@pytest.mark.parametrize("symmetric,constant_ne,sc,dwg", OPTION_COMBOS)
def test_every_model_runs_in_msprime(symmetric, constant_ne, sc, dwg):
    config = make_config(replicates=2, symmetric=symmetric, **{
        "constant Ne": constant_ne, "secondary contact": sc, "divergence with gene flow": dwg})
    *_, models, _, _ = build(config)
    for model in models:
        samples = {name: 1 for name in present_day_populations(model)}
        msprime.sim_ancestry(samples=samples, demography=model,
                             sequence_length=1000, random_seed=1)


# ---------------------------------------------------------------------------
# 3. Drawn values respect the priors
# ---------------------------------------------------------------------------

def test_drawn_values_within_priors():
    config = make_config(replicates=300, **{"constant Ne": False})
    *_, models, _, _ = build(config)
    for model in models:
        for pop in model.populations:
            assert NE[0] <= pop.initial_size <= NE[1]
        times = split_times(model)
        if "AB" in times:
            assert DIV_AB[0] <= times["AB"] <= DIV_AB[1]
            assert times["ABC"] > times["AB"]
        if "ABC" in times:
            assert DIV_ABC[0] <= times["ABC"] <= DIV_ABC[1]


def test_constant_ne_gives_every_population_the_same_size():
    *_, models, _, _ = build(make_config(**{"constant Ne": True}))
    for model in models:
        assert len({p.initial_size for p in model.populations}) == 1


def test_variable_ne_gives_populations_different_sizes():
    *_, models, _, _ = build(make_config(**{"constant Ne": False}))
    full = [m for m in models if len(m.populations) == 5]
    assert any(len({p.initial_size for p in m.populations}) > 1 for m in full)


def test_ancestor_older_than_descendant_when_priors_overlap():
    tree = ("((A[&ne=1000-2000],B[&ne=1000-2000])AB[&ne=1000-2000,div=10000-90000],"
            "C[&ne=1000-2000])ABC[&ne=1000-2000,div=20000-100000];")
    config = make_config(trees=(tree,), replicates=500, **{"secondary contact": False})
    *_, models, _, _ = build(config)
    for model in models:
        times = split_times(model)
        if "AB" in times:
            assert times["ABC"] >= times["AB"]


def test_impossible_prior_nesting_gives_a_clear_error():
    # AB can be drawn older (up to 90000) than ABC's maximum (30000).
    tree = ("((A[&ne=1000-2000],B[&ne=1000-2000])AB[&ne=1000-2000,div=10000-90000],"
            "C[&ne=1000-2000])ABC[&ne=1000-2000,div=20000-30000];")
    config = make_config(trees=(tree,), **{"secondary contact": False})
    with pytest.raises(ValueError, match=r"(?s)AB.*ABC|ABC.*AB"):
        build(config)


# ---------------------------------------------------------------------------
# 4. Migration happens when (and between whom) it should
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("symmetric", [True, False])
def test_secondary_contact_is_recent(symmetric):
    config = make_config(replicates=20, symmetric=symmetric, **{"max migration events": 1})
    *_, models, _, _ = build(config)
    sc_models = [m for m in models if kind_of(m) == "secondary contact"]
    assert sc_models
    for model in sc_models:
        youngest_split = min(t for t in split_times(model).values() if t > 0)
        epochs = list(migration_epochs(model))
        for start, end, source, dest, rate in epochs:
            assert start == 0
            assert end <= youngest_split
            assert MIG[0] <= rate <= MIG[1]
        if symmetric:
            rates = {(s, d): r for _, _, s, d, r in epochs}
            for (s, d), r in rates.items():
                assert rates[(d, s)] == r


@pytest.mark.parametrize("symmetric", [True, False])
def test_divergence_with_gene_flow_happens_after_the_split(symmetric):
    config = make_config(replicates=20, symmetric=symmetric, **{
        "max migration events": 1, "secondary contact": False,
        "divergence with gene flow": True})
    *_, models, _, _ = build(config)
    dwg_models = [m for m in models if kind_of(m) == "gene flow"]
    assert dwg_models
    for model in dwg_models:
        times = split_times(model)
        parent = {d: e.ancestral for e in model.events if hasattr(e, "ancestral")
                  for d in e.derived}
        for start, end, source, dest, rate in migration_epochs(model):
            ancestor = parent[source]
            assert parent[dest] == ancestor                       # sisters only
            daughters_formed = max(times.get(source, 0), times.get(dest, 0))
            assert daughters_formed < start < end <= times[ancestor]
            assert MIG[0] <= rate <= MIG[1]


def test_nm_prior_sets_number_of_migrants_into_the_source_population():
    # msprime's M[j, k] is the fraction of population j (forward in time) made up of
    # migrants from k, so the number of migrants into j is N_j * M[j, k].
    config = make_config(replicates=50, symmetric=False, **{
        "constant Ne": False, "migration rate": ["Nm0.5", "Nm2"],
        "max migration events": 1})
    *_, models, _, _ = build(config)
    sizes_checked = 0
    for model in models:
        size = {p.name: p.initial_size for p in model.populations}
        for _, _, source, dest, rate in migration_epochs(model):
            assert 0.5 - 1e-6 <= size[source] * rate <= 2 + 1e-6
            sizes_checked += 1
    assert sizes_checked > 0


# ---------------------------------------------------------------------------
# 5. Reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_gives_identical_models():
    opts = {"constant Ne": False, "divergence with gene flow": True, "symmetric": False}
    *_, first, labels1, _ = build(make_config(**opts))
    *_, second, labels2, _ = build(make_config(**opts))
    assert labels1 == labels2
    assert [fingerprint(m) for m in first] == [fingerprint(m) for m in second]


def test_different_seed_gives_different_models():
    *_, first, _, _ = build(make_config(seed=1))
    *_, second, _, _ = build(make_config(seed=2))
    assert [fingerprint(m) for m in first] != [fingerprint(m) for m in second]


# ---------------------------------------------------------------------------
# 6. Multiple species trees
# ---------------------------------------------------------------------------

def test_multiple_trees_get_distinct_labels_and_correct_tree_index():
    config = make_config(trees=(TREE3, TREE3_AC), **{"secondary contact": False})
    _, (div, _, _), models, labels, sp_index = build(config)
    # The panmixia model is shared, so the second tree contributes 2, not 3.
    assert [len(d) for d in div] == [3, 2]
    assert sorted(set(labels)) == list(range(5))
    for model, tree_ix in zip(models, sp_index):
        names = {p.name for p in model.populations}
        if "AC" in names:
            assert tree_ix == 1
        if "AB" in names:
            assert tree_ix == 0


def test_same_tree_written_differently_is_not_duplicated():
    config = make_config(trees=(TREE3, TREE3_SWAPPED), **{"secondary contact": False})
    div, _, _ = ModelBuilder(config).build_models()
    assert [len(d) for d in div] == [3, 0]


# ---------------------------------------------------------------------------
# 7. Plotting
# ---------------------------------------------------------------------------

def test_validate_models_writes_pdf(tmp_path):
    builder, _, models, labels, _ = build(make_config(replicates=2))
    out = tmp_path / "models.pdf"
    builder.validate_models(models, labels, outplot=str(out))
    assert out.stat().st_size > 0


@pytest.mark.parametrize("dwg", [False, True])
def test_symmetric_nm_gives_same_number_of_migrants_both_ways(dwg):
    config = make_config(replicates=30, symmetric=True, **{
        "constant Ne": False, "migration rate": ["Nm0.5", "Nm2"], "max migration events": 1,
        "secondary contact": not dwg, "divergence with gene flow": dwg})
    *_, models, _, _ = build(config)
    checked = 0
    for model in models:
        size = {p.name: p.initial_size for p in model.populations}
        nm = {}
        for start, end, source, dest, rate in migration_epochs(model):
            nm[(start, source, dest)] = size[source] * rate
        for (start, source, dest), value in nm.items():
            assert value == pytest.approx(nm[(start, dest, source)])
            assert 0.5 - 1e-6 <= value <= 2 + 1e-6
            checked += 1
    assert checked > 0


def test_identical_trees_give_a_warning(caplog):
    config = make_config(trees=(TREE3, TREE3_SWAPPED), **{"secondary contact": False})
    logging.disable(logging.NOTSET)
    try:
        with caplog.at_level(logging.WARNING, logger="popai.generate_models"):
            ModelBuilder(config).build_models()
    finally:
        logging.disable(logging.INFO)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "Species trees 1 and 2 are identical" in warnings[0].getMessage()


def test_different_trees_give_no_warning(caplog):
    config = make_config(trees=(TREE3, TREE3_AC), **{"secondary contact": False})
    logging.disable(logging.NOTSET)
    try:
        with caplog.at_level(logging.WARNING, logger="popai.generate_models"):
            ModelBuilder(config).build_models()
    finally:
        logging.disable(logging.INFO)
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]



def _matrix_from_text(pops, allowed_text, blocked_text):
    csv = "," + ",".join(pops) + "\n" + "\n".join(
        p + "," + ",".join(blocked_text if p == q else allowed_text for q in pops) for p in pops)
    return pd.read_csv(io.StringIO(csv), index_col=0)


@pytest.mark.parametrize("allowed,blocked", [("T", "F"), ("TRUE", "FALSE"), ("true", "false")])
def test_migration_matrix_accepts_t_and_true(allowed, blocked):
    mig = _matrix_from_text(all_populations(TREE3), allowed, blocked)
    _, sc, _ = ModelBuilder(make_config(migration=mig)).build_models()
    assert len(sc[0]) == 7


def test_unrecognized_migration_matrix_value_gives_a_clear_error():
    mig = _matrix_from_text(all_populations(TREE3), "Y", "N")
    with pytest.raises(ValueError, match="must be T or F"):
        ModelBuilder(make_config(migration=mig)).build_models()

TREE3_MIXED_NE = ("((A[&ne=10000-50000],B[&ne=10000-50000])AB[&ne=20000-60000,div=10000-50000],"
                  "C[&ne=5000-9000])ABC[&ne=10000-50000,div=100000-500000];")


def _warnings_from_drawing(config, caplog):
    logging.disable(logging.NOTSET)
    try:
        with caplog.at_level(logging.WARNING, logger="popai.generate_models"):
            build(config)
    finally:
        logging.disable(logging.INFO)
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_constant_ne_warns_about_ignored_ne_priors(caplog):
    config = make_config(trees=(TREE3_MIXED_NE,), **{"constant Ne": True})
    warnings = _warnings_from_drawing(config, caplog)
    assert len(warnings) == 1
    assert "Ne prior for A (10000-50000)" in warnings[0]
    assert "given for AB, C are ignored" in warnings[0]


def test_constant_ne_with_matching_priors_gives_no_warning(caplog):
    assert _warnings_from_drawing(make_config(**{"constant Ne": True}), caplog) == []


def test_variable_ne_gives_no_ne_warning(caplog):
    config = make_config(trees=(TREE3_MIXED_NE,), **{"constant Ne": False})
    assert _warnings_from_drawing(config, caplog) == []