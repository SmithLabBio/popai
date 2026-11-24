"""Tests for popai.process_user_models.ModelReader (user-specified models).

Model files are written in code into a temporary folder, so each test shows exactly
what goes in. Most tests check properties that must hold whatever the random draws
are (for example, a migration window always starts while its populations exist), over
many replicates. Each model is also checked against msprime, by simulating from it.

Run with:  python -m pytest tests/test_user_models.py
"""

import copy
import logging
import os
import re

import matplotlib
matplotlib.use("Agg")   # draw plots without a screen
import matplotlib.pyplot as plt
import msprime
import numpy as np
import pytest

from popai.process_user_models import ModelReader

logging.disable(logging.WARNING)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Building inputs
# ---------------------------------------------------------------------------

POPULATIONS = """[Populations]
A=[10000,50000]
B=[10000,50000]
C=[10000,50000]
AB=[10000,50000]
ABC=[10000,50000]
"""

SPLITS = """div_AB = split{time=[10000,20000], descendants=[A,B], ancestor=AB}
div_ABC = split{time=[50000,100000], descendants=[AB,C], ancestor=ABC}
"""


def model(events="", populations=POPULATIONS, splits=SPLITS, migration=None):
    """The text of a model file: populations, an optional [Migration] table, and events."""
    text = populations + "\n"
    if migration is not None:
        text += "[Migration]\nmatrix =\n" + migration + "\n"
    return text + "[Events]\n" + splits + events


def matrix(*rows):
    """A [Migration] table from rows of cells, e.g. matrix(["", "A", "B"], ["A", "0", "1e-5"], ...)."""
    return "".join("    " + "\t".join(row) + "\n" for row in rows)


def read(tmp_path, files, replicates=20, seed=1):
    """Write {file name: text} to a folder and read it. Returns (demographies, labels, reader)."""
    folder = tmp_path / "models"
    folder.mkdir(exist_ok=True)
    for name, text in files.items():
        (folder / name).write_text(text)
    reader = ModelReader({"seed": seed, "replicates": replicates, "user models": str(folder)})
    demographies, labels = reader.read_models()
    return demographies, labels, reader


def read_one(tmp_path, text, replicates=20, seed=1):
    """Read a single model file; returns its demographies."""
    return read(tmp_path, {"model_0.model": text}, replicates, seed)[0]


def read_error(tmp_path, text):
    """Read a single model file that should fail; returns the error message."""
    with pytest.raises(ValueError) as error:
        read_one(tmp_path, text, replicates=2)
    return " ".join(str(error.value).split())


# ---------------------------------------------------------------------------
# Inspecting drawn models
# ---------------------------------------------------------------------------

def sizes(demography):
    return {p.name: p.initial_size for p in demography.populations}


def split_times(demography):
    """{ancestor: time} for each split."""
    return {e.ancestral: e.time for e in demography.events if hasattr(e, "derived")}


def lifespan(demography, populations):
    """(from, until): when all the populations exist, backwards in time."""
    splits = [e for e in demography.events if hasattr(e, "derived")]
    start = max([e.time for e in splits if e.ancestral in populations] or [0])
    end = min([e.time for e in splits if set(e.derived) & set(populations)] or [np.inf])
    return start, end


def migration_window(demography, source, dest):
    """(start, stop, rate) of a migration event from source to dest."""
    changes = [e for e in demography.events if type(e).__name__ == "MigrationRateChange"
               and e.source == source and e.dest == dest]
    start = [e for e in changes if e.rate > 0]
    stop = [e for e in changes if e.rate == 0]
    assert len(start) == 1 and len(stop) == 1
    return start[0].time, stop[0].time, start[0].rate


def present_rate(demography, source, dest):
    names = [p.name for p in demography.populations]
    return demography.migration_matrix[names.index(source), names.index(dest)]


def simulate(demography):
    """Simulate a few samples from each population that is sampled at the present (not one
    that merges into another at time 0). Fails if msprime can't use the model."""
    merged = {d for e in demography.events if hasattr(e, "derived") and e.time == 0 for d in e.derived}
    samples = {p.name: 1 for p in demography.populations
               if p.default_sampling_time in (None, 0) and p.name not in merged}
    return msprime.sim_ancestry(samples, demography=demography, sequence_length=100, random_seed=1)


# ---------------------------------------------------------------------------
# Model files and labels
# ---------------------------------------------------------------------------

def test_one_label_per_replicate_in_number_order(tmp_path):
    files = {f"model_{i}.model": model() for i in (0, 1, 10, 2)}
    demographies, labels, _ = read(tmp_path, files, replicates=3)
    assert len(demographies) == 12
    assert labels == ["model_0"] * 3 + ["model_1"] * 3 + ["model_2"] * 3 + ["model_10"] * 3


def test_any_name_ending_in_a_number_is_labelled_model_number(tmp_path):
    files = {"isolation_0.model": model(), "my_model_1.model": model(), "notes.txt": "ignored",
             "models.pdf": "ignored"}
    _, labels, _ = read(tmp_path, files, replicates=1)
    assert labels == ["model_0", "model_1"]


@pytest.mark.parametrize("files, message", [
    ({"Model1.model": model()}, "must end in _x.model"),
    ({"a_0.model": model(), "b_0.model": model()}, "both end in _0"),
    ({"notes.txt": "no models here"}, "There are no .model files"),
])
def test_bad_model_file_names(tmp_path, files, message):
    with pytest.raises(ValueError, match=message):
        read(tmp_path, files)


def test_same_seed_same_models_and_different_seed_different_models(tmp_path):
    text = model("mig = symmetric_migration{start=[0,5000], stop=div_AB, populations=[A,B], rate=[1e-5,1e-4]}\n")
    first = read(tmp_path, {"model_0.model": text}, seed=3)[0]
    again = read(tmp_path, {"model_0.model": text}, seed=3)[0]
    other = read(tmp_path, {"model_0.model": text}, seed=4)[0]
    assert [str(d) for d in first] == [str(d) for d in again]
    assert [str(d) for d in first] != [str(d) for d in other]


# ---------------------------------------------------------------------------
# Population sizes
# ---------------------------------------------------------------------------

def test_sizes_are_whole_numbers_in_their_ranges(tmp_path):
    for demography in read_one(tmp_path, model(), replicates=50):
        for size in sizes(demography).values():
            assert 10000 <= size <= 50000 and size == round(size)


def test_other_ways_to_give_a_size(tmp_path):
    populations = "[Populations]\nA=[10000,50000]\nB=A\nC=20000\nAB=0.5*pop_C\nABC=[10000,50000]\n"
    for demography in read_one(tmp_path, model(populations=populations)):
        size = sizes(demography)
        assert size["B"] == size["A"]
        assert size["C"] == 20000 and size["AB"] == 10000


@pytest.mark.parametrize("value", ["[10000,5000O]", "5000O", "[10000,20000,30000]", "D"])
def test_unreadable_size(tmp_path, value):
    populations = POPULATIONS.replace("B=[10000,50000]", f"B={value}")
    assert "can't read the size of population B" in read_error(tmp_path, model(populations=populations))


def test_size_can_only_copy_a_population_listed_above_it(tmp_path):
    populations = POPULATIONS.replace("A=[10000,50000]", "A=B")
    assert "can't read the size of population A, 'B'" in read_error(tmp_path, model(populations=populations))


def test_no_populations_section(tmp_path):
    assert "there is no [Populations] section" in read_error(tmp_path, "[Events]\n" + SPLITS)


def test_no_events_section_for_one_population(tmp_path):
    for demography in read_one(tmp_path, "[Populations]\nA=[10000,50000]\n"):
        assert len(demography.events) == 0
        simulate(demography)


# ---------------------------------------------------------------------------
# Reading events
# ---------------------------------------------------------------------------

def test_spaces_and_a_trailing_comma_are_fine(tmp_path):
    splits = ("div_AB = split { time = [ 10000 , 20000 ] , descendants = [ A , B ] , ancestor = AB , }\n"
              "div_ABC = split{time=[50000,100000], descendants=[AB,C], ancestor=ABC}\n")
    for demography in read_one(tmp_path, model(splits=splits)):
        assert 10000 <= split_times(demography)["AB"] <= 20000


@pytest.mark.parametrize("event, message", [
    ("p = pulse{time:500, source=A, dest=B, prop=0.1}", "'time:500' should be written as name=value"),
    ("p = pulse time=500, source=A, dest=B, prop=0.1}", "Write events as type{name=value, ...}"),
    ("p = pulsee{time=500, source=A, dest=B, prop=0.1}", "has the event type 'pulsee', which isn't valid"),
    ("p = pulse{tme=500, time=600, source=A, dest=B, prop=0.1}", "the pulse event p has tme"),
    ("p = pulse{time=500, source=A, prop=0.1}", "the pulse event p is missing dest"),
    ("p = pulse{time=500, source=[A,B], dest=C, prop=0.1}", "source should be one population"),
    ("p = pulse{time=500, source=A, dest=D, prop=0.1}", "event p includes D, which isn't listed in [Populations]"),
    ("p = pulse{time=[500,600,700], source=A, dest=B, prop=0.1}", "a range for time needs two values"),
    ("m = symmetric_migration{start=0, stop=div_AB, populations=[A,B,C], rate=1e-5}",
     "populations should be two populations"),
    ("b = bottleneck{time=500, population=A}", "the bottleneck event b is missing prop"),
])
def test_badly_written_events(tmp_path, event, message):
    assert message in read_error(tmp_path, model(event + "\n"))


def test_unknown_ancestor(tmp_path):
    populations = "[Populations]\nA=[10000,50000]\nB=[10000,50000]\n"
    splits = "div_AB = split{time=[10000,20000], descendants=[A,B], ancestor=AB}\n"
    assert "includes AB, which isn't listed" in read_error(tmp_path, model(populations=populations, splits=splits))


def test_events_can_be_listed_in_any_order(tmp_path):
    events = [
        "div_AB = split{time=[10000,20000], descendants=[A,B], ancestor=AB}\n",
        "div_ABC = split{time=max(div_AB, 45000), descendants=[AB,C], ancestor=ABC}\n",
        "mig = symmetric_migration{start=0, stop=div_AB, populations=[A,B], rate=[1e-5,1e-4]}\n",
    ]
    in_order = read_one(tmp_path, model(splits="", events="".join(events)))
    backwards = read_one(tmp_path, model(splits="", events="".join(reversed(events))))
    assert [str(d) for d in in_order] == [str(d) for d in backwards]


def test_events_that_refer_to_each_other_in_a_loop(tmp_path):
    splits = ("div_AB = split{time=div_ABC/2, descendants=[A,B], ancestor=AB}\n"
              "div_ABC = split{time=2*div_AB, descendants=[AB,C], ancestor=ABC}\n")
    assert "depend on each other in a loop" in read_error(tmp_path, model(splits=splits))


def test_a_migration_event_name_can_not_be_used_as_a_value(tmp_path):
    events = ("m = symmetric_migration{start=0, stop=div_AB, populations=[A,B], rate=1e-5}\n"
              "ps = popsize{time=m, population=A, size=1000}\n")
    assert "ps uses m, which is a migration event" in read_error(tmp_path, model(events))


# ---------------------------------------------------------------------------
# Numbers written in the file
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("event, message", [
    ("div_X = split{time=[20000,10000], descendants=[A,B], ancestor=AB}", "Put the smaller value first"),
    ("m = symmetric_migration{start=0, stop=div_AB, populations=[A,B], rate=[2Nm,0.5Nm]}", "Put the smaller value first"),
    ("b = bottleneck{time=500, population=A, prop=[0.2,0.1]}", "Put the smaller value first"),
    ("p = pulse{time=-5, source=A, dest=B, prop=0.1}", "times can't be negative"),
    ("b = bottleneck{time=500, population=A, prop=1.5}", "proportions must be between 0 and 1"),
    ("m = asymmetric_migration{start=0, stop=div_AB, source=A, dest=B, rate=-1e-5}", "migration rates can't be negative"),
])
def test_numbers_that_can_not_be_right(tmp_path, event, message):
    splits = SPLITS if "div_X" not in event else ""
    text = model(event.replace("div_X", "div_AB") + "\n", splits=splits)
    if "div_X" in event:
        text += "div_ABC = split{time=[50000,100000], descendants=[AB,C], ancestor=ABC}\n"
    assert message in read_error(tmp_path, text)


def test_size_of_zero(tmp_path):
    populations = POPULATIONS.replace("A=[10000,50000]", "A=0")
    assert "sizes must be above 0" in read_error(tmp_path, model(populations=populations))


def test_negative_growth_rates_are_allowed(tmp_path):
    for demography in read_one(tmp_path, model("g = popgrowth{time=0, population=C, rate=[-0.001,0.001]}\n")):
        rate = [e.growth_rate for e in demography.events if getattr(e, "growth_rate", None) is not None][0]
        assert -0.001 <= rate <= 0.001


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

def test_expressions(tmp_path):
    splits = ("div_AB = split{time=[10000,20000], descendants=[A,B], ancestor=AB}\n"
              "div_ABC = split{time=mean(div_AB, 3*1e4, median(40000, 50000, 90000)), descendants=[AB,C], ancestor=ABC}\n")
    events = "ps = popsize{time=max(div_AB/10, 500), population=C, size=0.01*pop_C}\n"
    for demography in read_one(tmp_path, model(events, splits=splits)):
        times = split_times(demography)
        assert times["ABC"] == pytest.approx((times["AB"] + 30000 + 50000) / 3)
        change = [e for e in demography.events if getattr(e, "initial_size", None)][0]
        assert change.time == pytest.approx(max(times["AB"] / 10, 500))
        assert change.initial_size == pytest.approx(0.01 * sizes(demography)["C"])


@pytest.mark.parametrize("value, name", [("max(div_AC, 45000)", "div_AC"), ("maxx(div_AB, 45000)", "maxx")])
def test_unknown_name_in_an_expression(tmp_path, value, name):
    splits = SPLITS.replace("[50000,100000]", value)
    message = read_error(tmp_path, model(splits=splits))
    assert f"name '{name}' is not defined" in message


# ---------------------------------------------------------------------------
# Split times
# ---------------------------------------------------------------------------

def test_split_times_in_their_ranges(tmp_path):
    for demography in read_one(tmp_path, model(), replicates=50):
        times = split_times(demography)
        assert 10000 <= times["AB"] <= 20000 and 50000 <= times["ABC"] <= 100000


def test_an_ancestor_is_never_younger_than_its_descendants(tmp_path):
    splits = ("div_AB = split{time=[30000,90000], descendants=[A,B], ancestor=AB}\n"
              "div_ABC = split{time=[50000,100000], descendants=[AB,C], ancestor=ABC}\n")
    for demography in read_one(tmp_path, model(splits=splits), replicates=200):
        times = split_times(demography)
        assert times["ABC"] >= times["AB"] and 50000 <= times["ABC"] <= 100000
        simulate(demography)


def test_impossible_split_time(tmp_path):
    splits = ("div_AB = split{time=[10000,20000], descendants=[A,B], ancestor=AB}\n"
              "div_ABC = split{time=[5000,8000], descendants=[AB,C], ancestor=ABC}\n")
    assert "no time works for div_ABC" in read_error(tmp_path, model(splits=splits))


def test_merges_at_time_zero(tmp_path):
    splits = ("div_AB = split{time=0, descendants=[A,B], ancestor=AB}\n"
              "div_ABC = split{time=0, descendants=[AB,C], ancestor=ABC}\n")
    for demography in read_one(tmp_path, model(splits=splits), replicates=3):
        assert split_times(demography) == {"AB": 0, "ABC": 0}
        simulate(demography)


# ---------------------------------------------------------------------------
# Migration events: when
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("event, source, dest", [
    # start and stop ranges overlap
    ("symmetric_migration{start=[5000,15000], stop=[8000,12000], populations=[A,B], rate=1e-5}", "A", "B"),
    # start can fall after A and B split
    ("symmetric_migration{start=[5000,25000], stop=div_AB, populations=[A,B], rate=1e-5}", "A", "B"),
    # A ends at div_AB, so the start must come before that
    ("asymmetric_migration{start=[0,30000], stop=[20000,40000], source=A, dest=C, rate=1e-5}", "A", "C"),
    # AB only exists after div_AB, so the start must come after that
    ("symmetric_migration{start=[0,60000], stop=div_ABC, populations=[AB,C], rate=1e-5}", "AB", "C"),
])
def test_migration_starts_while_its_populations_exist_and_stops_after_it_starts(tmp_path, event, source, dest):
    for demography in read_one(tmp_path, model(f"mig = {event}\n"), replicates=200):
        start, stop, _ = migration_window(demography, source, dest)
        exists_from, exists_until = lifespan(demography, [source, dest])
        assert exists_from <= start < exists_until
        assert start < stop
        simulate(demography)


@pytest.mark.parametrize("event, message", [
    ("symmetric_migration{start=0, stop=div_ABC, populations=[AB,C], rate=1e-5}", "no start time works"),
    ("symmetric_migration{start=30000, stop=40000, populations=[A,B], rate=1e-5}", "no start time works"),
    ("symmetric_migration{start=[1000,5000], stop=500, populations=[A,B], rate=1e-5}", "no start time works"),
])
def test_impossible_migration_window(tmp_path, event, message):
    assert message in read_error(tmp_path, model(f"mig = {event}\n"))


# ---------------------------------------------------------------------------
# Migration events: rates
# ---------------------------------------------------------------------------

def windows(tmp_path, event, replicates=20):
    """For each replicate: {(source, dest): rate} and population sizes."""
    result = []
    for demography in read_one(tmp_path, model(f"mig = {event}\n"), replicates=replicates):
        rates = {(e.source, e.dest): e.rate for e in demography.events
                 if type(e).__name__ == "MigrationRateChange" and e.rate > 0}
        result.append((rates, sizes(demography)))
    return result


def test_symmetric_fixed_rate(tmp_path):
    for rates, _ in windows(tmp_path, "symmetric_migration{start=0, stop=div_AB, populations=[A,B], rate=1e-5}"):
        assert rates == {("B", "A"): 1e-5, ("A", "B"): 1e-5}


def test_symmetric_rate_range_is_one_draw_for_both_directions(tmp_path):
    for rates, _ in windows(tmp_path, "symmetric_migration{start=0, stop=div_AB, populations=[A,B], rate=[1e-5,1e-4]}"):
        assert rates[("A", "B")] == rates[("B", "A")] and 1e-5 <= rates[("A", "B")] <= 1e-4


@pytest.mark.parametrize("rate", ["[0.5Nm,2Nm]", "[0.5 Nm, 2 Nm]"])
def test_symmetric_nm_gives_the_same_number_of_migrants_each_way(tmp_path, rate):
    for rates, size in windows(tmp_path, f"symmetric_migration{{start=0, stop=div_AB, populations=[A,B], rate={rate}}}"):
        nm_ab = rates[("A", "B")] * size["A"]   # m = Nm / N_source
        nm_ba = rates[("B", "A")] * size["B"]
        assert nm_ab == pytest.approx(nm_ba) and 0.5 <= nm_ab <= 2


def test_asymmetric_fixed_nm(tmp_path):
    for rates, size in windows(tmp_path, "asymmetric_migration{start=0, stop=div_AB, source=A, dest=B, rate=2Nm}"):
        assert rates[("A", "B")] * size["A"] == pytest.approx(2)


@pytest.mark.parametrize("rate, message", [
    ("Nm1", "Write Nm after the number"),
    ("[1Nm,1e-4]", "Both ends of a range must be rates, or both Nm values"),
])
def test_badly_written_rates(tmp_path, rate, message):
    event = f"mig = symmetric_migration{{start=0, stop=div_AB, populations=[A,B], rate={rate}}}\n"
    assert message in read_error(tmp_path, model(event))


# ---------------------------------------------------------------------------
# Migration from the present: the [Migration] table
# ---------------------------------------------------------------------------

def test_matrix_rates_and_nm(tmp_path):
    table = matrix(["", "A", "B", "C"],
                   ["A", "0", "[1e-5,1e-4]", "0"],
                   ["B", "2e-5", "0", "0"],
                   ["C", "1 Nm", "0", "0"])
    for demography in read_one(tmp_path, model(migration=table), replicates=50):
        assert 1e-5 <= present_rate(demography, "A", "B") <= 1e-4
        assert present_rate(demography, "B", "A") == 2e-5
        assert present_rate(demography, "C", "A") * sizes(demography)["C"] == pytest.approx(1)
        assert present_rate(demography, "A", "C") == 0


def test_matrix_migration_stops_when_the_populations_split(tmp_path):
    table = matrix(["", "A", "B"], ["A", "0", "1e-5"], ["B", "1e-5", "0"])
    for demography in read_one(tmp_path, model(migration=table), replicates=5):
        names = [p.name for p in demography.populations]
        split = split_times(demography)["AB"]
        for epoch in demography.debug().epochs:
            on = epoch.migration_matrix[names.index("A"), names.index("B")] > 0
            assert on == (epoch.end_time <= split)
        simulate(demography)


def test_matrix_with_the_name_of_a_population_merged_at_time_zero(tmp_path):
    splits = ("div_AB = split{time=0, descendants=[A,B], ancestor=AB}\n"
              "div_ABC = split{time=[50000,100000], descendants=[AB,C], ancestor=ABC}\n")
    table = matrix(["", "AB", "C"], ["AB", "0", "2e-5"], ["C", "2e-5", "0"])
    for demography in read_one(tmp_path, model(splits=splits, migration=table), replicates=3):
        assert present_rate(demography, "AB", "C") == 2e-5
        simulate(demography)


@pytest.mark.parametrize("splits, table, message", [
    (SPLITS, matrix(["", "AB", "C"], ["AB", "0", "2e-5"], ["C", "0", "0"]), "AB is an ancestral population"),
    (SPLITS.replace("[10000,20000]", "0"), matrix(["", "A", "C"], ["A", "0", "2e-5"], ["C", "0", "0"]),
     "A merges with another population at time 0"),
    (SPLITS, matrix(["", "A", "B"], ["A", "0", "[1e-5,abc]"], ["B", "0", "0"]), "entry for A to B is '[1e-5,abc]'"),
    (SPLITS, matrix(["", "A", "B"], ["A", "1e-5", "0"], ["B", "0", "0"]), "Diagonal entries must be 0"),
    (SPLITS, matrix(["", "A", "D"], ["A", "0", "0"], ["D", "0", "0"]), "includes ['D'], which are not listed"),
    (SPLITS, matrix(["", "A", "B"], ["A", "0"], ["B", "0", "0"]), "row A of the [Migration] table has 1 entries"),
])
def test_bad_migration_tables(tmp_path, splits, table, message):
    assert message in read_error(tmp_path, model(splits=splits, migration=table))


# ---------------------------------------------------------------------------
# Other events
# ---------------------------------------------------------------------------

def test_other_event_types(tmp_path):
    events = ("bn_A = bottleneck{time=[500,700], population=A, prop=[0.1,0.2]}\n"
              "pulse_AB = pulse{time=[1000,2000], source=A, dest=B, prop=[0.05,0.1]}\n"
              "size_C = popsize{time=[3000,4000], population=C, size=[1000,2000]}\n"
              "grow_B = popgrowth{time=0, population=B, rate=0.001}\n")
    for demography in read_one(tmp_path, model(events)):
        kinds = {type(e).__name__: e for e in demography.events}
        bottleneck, pulse = kinds["SimpleBottleneck"], kinds["MassMigration"]
        assert 500 <= bottleneck.time <= 700 and 0.1 <= bottleneck.proportion <= 0.2
        assert (pulse.source, pulse.dest) == ("A", "B")
        assert 1000 <= pulse.time <= 2000 and 0.05 <= pulse.proportion <= 0.1
        changes = [e for e in demography.events if type(e).__name__ == "PopulationParametersChange"]
        size_change = [e for e in changes if e.population == "C"][0]
        growth_change = [e for e in changes if e.population == "B"][0]
        assert 3000 <= size_change.time <= 4000 and 1000 <= size_change.initial_size <= 2000
        assert growth_change.growth_rate == 0.001
        simulate(demography)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

MERGED_WITH_TABLE = model(splits=SPLITS.replace("[10000,20000]", "0"),
                          migration=matrix(["", "AB", "C"], ["AB", "0", "2e-5"], ["C", "2e-5", "0"]))
WITH_BOTTLENECK = model("bn_A = bottleneck{time=500, population=A, prop=0.1}\n")


def pdf_pages(path):
    return len(re.findall(rb"/Type\s*/Page\b", open(path, "rb").read()))


def test_plot_has_one_page_per_model_even_when_one_can_not_be_drawn(tmp_path):
    # model 0 has a bottleneck, which the plotting package can't draw
    files = {"model_0.model": WITH_BOTTLENECK, "model_1.model": model(), "model_2.model": MERGED_WITH_TABLE}
    demographies, labels, reader = read(tmp_path, files, replicates=2)
    reader.validate_models(demographies, labels, outplot=str(tmp_path / "models.pdf"))
    assert pdf_pages(tmp_path / "models.pdf") == 3
    assert plt.get_fignums() == []


def test_time_zero_merges_can_be_drawn_and_the_model_is_not_changed(tmp_path):
    demographies, _, reader = read(tmp_path, {"model_0.model": MERGED_WITH_TABLE}, replicates=1)
    before = str(demographies[0])
    reader._nonzero(copy.deepcopy(demographies[0])).to_demes()   # raises if it can't be drawn
    assert str(demographies[0]) == before


# ---------------------------------------------------------------------------
# The models that come with popai
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("folder", ["example_models", "tutorial_data/tutorial_3_data/models"])
def test_example_and_tutorial_models(folder):
    reader = ModelReader({"seed": 1, "replicates": 3, "user models": os.path.join(REPO, folder)})
    demographies, labels = reader.read_models()
    assert labels == ["model_0"] * 3 + ["model_1"] * 3 + ["model_2"] * 3
    for demography in demographies:
        simulate(demography)

# ---------------------------------------------------------------------------
# Log-uniform ranges
# ---------------------------------------------------------------------------

def is_log_uniform(values, low, high):
    """Every value is in [low, high], and about half are below the geometric midpoint, as
    expected on a log scale (a uniform draw over a range spanning 100x would put ~9% there)."""
    below = np.mean([v < np.sqrt(low * high) for v in values])
    return all(low <= v <= high for v in values) and 0.35 < below < 0.65


LOG_TABLE = matrix(["", "A", "B"], ["A", "0", "loguniform[1e-6, 1e-4]"], ["B", "0", "0"])
LOG_EVENTS = ("mig = symmetric_migration{start=loguniform[10,1000], stop=loguniform[2000,9000], "
              "populations=[A,C], rate=loguniform[0.02Nm,2Nm]}\n"
              "g = popgrowth{time=loguniform[10,1000], population=C, rate=loguniform[1e-5,1e-3]}\n"
              "b = bottleneck{time=[500,700], population=B, prop=loguniform[0.001,0.1]}\n")
LOG_POPULATIONS = POPULATIONS.replace("A=[10000,50000]", "A=loguniform[1000,100000]")
LOG_SPLITS = SPLITS.replace("[10000,20000]", "loguniform[10000,1000000]").replace("[50000,100000]", "[1000000,2000000]")


def test_loguniform_in_every_kind_of_value(tmp_path):
    demographies = read_one(tmp_path, model(LOG_EVENTS, populations=LOG_POPULATIONS, splits=LOG_SPLITS,
                                            migration=LOG_TABLE), replicates=400)
    found = {key: [] for key in ("size", "split", "start", "stop", "nm", "growth time", "growth rate",
                                 "prop", "matrix")}
    for demography in demographies:
        size = sizes(demography)
        found["size"].append(size["A"])
        found["split"].append(split_times(demography)["AB"])
        start, stop, rate = migration_window(demography, "A", "C")
        found["start"].append(start)
        found["stop"].append(stop)
        found["nm"].append(rate * size["A"])   # m = Nm / N_source
        growth = [e for e in demography.events if getattr(e, "growth_rate", None) is not None][0]
        found["growth time"].append(growth.time)
        found["growth rate"].append(growth.growth_rate)
        found["prop"].append([e for e in demography.events if hasattr(e, "proportion")][0].proportion)
        found["matrix"].append(present_rate(demography, "A", "B"))
    limits = {"size": (1000, 100000), "split": (10000, 1000000), "start": (10, 1000), "stop": (2000, 9000),
              "nm": (0.02, 2), "growth time": (10, 1000), "growth rate": (1e-5, 1e-3), "prop": (0.001, 0.1),
              "matrix": (1e-6, 1e-4)}
    for key, (low, high) in limits.items():
        assert is_log_uniform(found[key], low, high), key
    # times and sizes are still whole numbers
    for key in ("size", "split", "start", "stop", "growth time"):
        assert all(v == round(v) for v in found[key]), key


def test_loguniform_rate_is_one_draw_for_both_directions(tmp_path):
    event = "symmetric_migration{start=0, stop=div_AB, populations=[A,B], rate=loguniform[1e-6,1e-4]}"
    for rates, _ in windows(tmp_path, event):
        assert rates[("A", "B")] == rates[("B", "A")] and 1e-6 <= rates[("A", "B")] <= 1e-4


def test_loguniform_with_spaces_in_the_migration_table(tmp_path):
    table = matrix(["", "A", "B"], ["A", "0", "loguniform[1e-6, 1e-4]"], ["B", "loguniform[ 1Nm , 2Nm ]", "0"])
    for demography in read_one(tmp_path, model(migration=table), replicates=50):
        assert 1e-6 <= present_rate(demography, "A", "B") <= 1e-4
        assert 1 <= present_rate(demography, "B", "A") * sizes(demography)["B"] <= 2


def test_models_without_loguniform_draw_as_before(tmp_path):
    # values recorded from popai 4c0d66a (before log-uniform ranges), seed 1
    table = matrix(["", "A", "B"], ["A", "0", "[1e-5,1e-4]"], ["B", "0", "0"])
    events = ("mig = symmetric_migration{start=[0,5000], stop=div_AB, populations=[A,C], rate=[0.5Nm,2Nm]}\n"
              "g = popgrowth{time=[100,500], population=C, rate=[-0.001,0.001]}\n")
    demography = read_one(tmp_path, model(events, migration=table), replicates=1, seed=1)[0]
    assert sizes(demography) == {"A": 30473, "B": 48019, "C": 15766, "AB": 47946, "ABC": 22473}
    assert split_times(demography) == {"AB": 14233, "ABC": 91385}
    assert migration_window(demography, "C", "A") == (2046, 14233, pytest.approx(8.400295138333053e-05))
    assert migration_window(demography, "A", "C") == (2046, 14233, pytest.approx(4.3461114150546033e-05))
    growth = [e for e in demography.events if getattr(e, "growth_rate", None) is not None][0]
    assert (growth.time, growth.growth_rate) == (111, pytest.approx(0.0005070262173496132))
    assert present_rate(demography, "A", "B") == pytest.approx(5.8432898189735043e-05)


@pytest.mark.parametrize("change, message", [
    (("populations", "A=[10000,50000]", "A=loguniform[0,50000]"), "a loguniform range must be above 0"),
    (("events", "b = bottleneck{time=500, population=A, prop=loguniform[0,0.5]}\n"), "a loguniform range must be above 0"),
    (("events", "m = symmetric_migration{start=loguniform[0,10], stop=div_AB, populations=[A,B], rate=1e-5}\n"),
     "a loguniform range must be above 0"),
    (("migration", matrix(["", "A", "B"], ["A", "0", "loguniform[0,1e-4]"], ["B", "0", "0"])),
     "A loguniform range must be above 0"),
    (("splits", "div_AB = split{time=[10000,20000], descendants=loguniform[A,B], ancestor=AB}\n"
                "div_ABC = split{time=[50000,100000], descendants=[AB,C], ancestor=ABC}\n"),
     "descendants can't be loguniform"),
])
def test_impossible_loguniform_values(tmp_path, change, message):
    kind = change[0]
    if kind == "populations":
        text = model(populations=POPULATIONS.replace(change[1], change[2]))
    elif kind == "events":
        text = model(change[1])
    elif kind == "migration":
        text = model(migration=change[1])
    else:
        text = model(splits=change[1])
    assert message in read_error(tmp_path, text)


def test_a_loguniform_range_drawn_from_an_expression_must_be_above_0(tmp_path):
    events = "b = popsize{time=loguniform[div_AB-div_AB, 100], population=A, size=1000}\n"
    assert "a loguniform range must be above 0" in read_error(tmp_path, model(events))


# ---------------------------------------------------------------------------
# The [Parameters] section, log and exp
# ---------------------------------------------------------------------------

EXPANSION = """[Parameters]
fold = loguniform[2,20]
lnfold = log(fold)

[Populations]
A=[10000,50000]
C=1000*fold
AB=1000*exp(lnfold)

[Events]
div_AB = split{time=[50000,100000], descendants=[A,C], ancestor=AB}
stop_C = popgrowth{time=[1000,20000], population=C, rate=0}
grow_C = popgrowth{time=0, population=C, rate=log(fold)/stop_C}
"""


def test_parameters_in_sizes_and_event_values(tmp_path):
    folds = []
    for demography in read_one(tmp_path, EXPANSION, replicates=400):
        size = sizes(demography)
        fold = size["C"] / 1000                     # C = 1000 * fold
        folds.append(fold)
        assert size["AB"] == pytest.approx(size["C"])  # exp(log(fold)) = fold
        stop, grow = sorted((e for e in demography.events if getattr(e, "growth_rate", None) is not None),
                            key=lambda e: e.time, reverse=True)
        assert grow.time == 0 and stop.growth_rate == 0
        assert grow.growth_rate == pytest.approx(np.log(fold) / stop.time)
        # C grows by the fold-change between stop_C and the present
        names = [p.name for p in demography.populations]
        trajectory = demography.debug().population_size_trajectory([0, stop.time])
        assert trajectory[0, names.index("C")] / trajectory[1, names.index("C")] == pytest.approx(fold)
    assert is_log_uniform(folds, 2, 20)
    assert any(f != round(f) for f in folds)        # parameters aren't rounded
    simulate(read_one(tmp_path, EXPANSION, replicates=1)[0])


def test_parameters_can_use_parameters_above_them(tmp_path):
    text = model(populations="[Parameters]\nx = [1,2]\ny = 2*x\nz = 7\n\n" +
                 POPULATIONS.replace("C=[10000,50000]", "C=10000*y+z"))
    for demography in read_one(tmp_path, text, replicates=50):
        c = sizes(demography)["C"]
        assert 20007 <= c <= 40007


@pytest.mark.parametrize("parameters, message", [
    ("x = [20,2]", "Put the smaller value first"),
    ("x = loguniform[0,2]", "a loguniform range must be above 0"),
    ("x = [1,2,3]", "A range needs two values"),
    ("2x = 3", "the parameter name 2x can't be used"),
    ("log = 3", "the parameter name log can't be used"),
    ("pop_A = 3", "the parameter name pop_A can't be used"),
    ("div_AB = 3", "the parameter name div_AB can't be used"),
    ("A = 3", "the parameter name A can't be used"),
    ("x = y*2", "can't work out parameter x"),
    ("x = [1,y]", "can't work out parameter x"),
])
def test_bad_parameters(tmp_path, parameters, message):
    assert message in read_error(tmp_path, model(populations=f"[Parameters]\n{parameters}\n\n" + POPULATIONS))


def test_a_parameter_can_not_use_a_parameter_below_it(tmp_path):
    text = model(populations="[Parameters]\nx = 2*y\ny = 3\n\n" + POPULATIONS)
    assert "can't work out parameter x" in read_error(tmp_path, text)