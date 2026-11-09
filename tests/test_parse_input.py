"""Tests for popai.parse_input.ModelConfigParser (reading the configuration file).

Each test writes a config file, and any small input files it needs, into a temporary
folder, so it shows exactly what goes in. The FASTA alignments and popfile come from
tests/alignments and tests/populations.txt. Most tests change one setting of a good
config and check that it is read correctly, or that it gives a clear error.

Run with:  python -m pytest tests/test_parse_input.py
"""

import logging
import os
import shutil

import pytest

from popai.generate_models import ModelBuilder
from popai.parse_input import ModelConfigParser

logging.disable(logging.WARNING)

TESTS = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Building inputs
# ---------------------------------------------------------------------------

TREE = ("((A[&ne=10000-50000],B[&ne=10000-50000])AB[&ne=10000-50000,div=10000-50000],"
        "C[&ne=10000-50000])ABC[&ne=10000-50000,div=100000-500000];")
MIGRATION = ",A,B,C,AB,ABC\nA,F,F,T,F,F\nB,F,F,F,F,F\nC,T,F,F,T,F\nAB,F,F,T,F,F\nABC,F,F,F,F,F\n"

# A small VCF: three diploid samples, with data on contigs 1 and 2
VCF_SAMPLES = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tn0\tn1\tn2\n"
VCF_SITES = ("1\t1\t.\tA\tT\t.\tPASS\t.\tGT\t0|1\t0|0\t1|1\n"
             "1\t7\t.\tA\tT\t.\tPASS\t.\tGT\t0|0\t0|1\t1|1\n"
             "2\t5\t.\tA\tT\t.\tPASS\t.\tGT\t0|1\t0|0\t1|0\n")
VCF_POPFILE = "individual\tpopulation\nn0\tA\nn1\tB\nn2\tB\n"


def nexus(*newicks):
    trees = "".join(f"Tree t{i}={newick}\n" for i, newick in enumerate(newicks))
    return f"#NEXUS\nBEGIN TREES;\n{trees}END;\n"


def vcf(header=""):
    return ("##fileformat=VCFv4.2\n" + header +
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n' + VCF_SAMPLES + VCF_SITES)


def settings(**changes):
    """The settings of a good config, as {section: {key: value}}. Pass section__key=value to
    change one (e.g. Other__replicates="0"), or None to leave it out."""
    sections = {
        "Model": {"species tree file": "tree.nex", "migration matrix": "migration.csv",
                  "symmetric": "True", "secondary contact": "True",
                  "divergence with gene flow": "False", "max migration events": "1",
                  "migration rate": "U(1e-5, 1e-4)", "constant Ne": "True"},
        "Other": {"seed": "1234", "replicates": "10"},
        "Simulations": {"mutation rate": "U(5e-9, 5e-8)", "substitution model": "JC69"},
        "Data": {"alignments": os.path.join(TESTS, "alignments"),
                 "popfile": os.path.join(TESTS, "populations.txt")},
    }
    for name, value in changes.items():
        section, key = name.split("__")
        sections[section][key.replace("_", " ")] = value
    return sections


def parse(tmp_path, files=None, **changes):
    """Write the config (and {file name: text} for any other files) to tmp_path, and parse it.
    Relative paths in the config are read from tmp_path."""
    files = {"tree.nex": nexus(TREE), "migration.csv": MIGRATION, **(files or {})}
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    text = ""
    for section, values in settings(**changes).items():
        text += f"[{section}]\n" + "".join(f"{k} = {v}\n" for k, v in values.items() if v is not None) + "\n"
    (tmp_path / "config.txt").write_text(text)
    cwd = os.getcwd()
    os.chdir(tmp_path)   # paths in a config are read from the folder popai is run in
    try:
        return ModelConfigParser("config.txt").parse_config()
    finally:
        os.chdir(cwd)


def parse_vcf(tmp_path, header="", files=None, **changes):
    """Parse a config that reads the small VCF, with the given header lines."""
    files = {"data.vcf": vcf(header), "pop.txt": VCF_POPFILE, **(files or {})}
    changes = {"Data__alignments": None, "Data__vcf": "data.vcf", "Data__popfile": "pop.txt", **changes}
    return parse(tmp_path, files, **changes)


def error(tmp_path, kind=ValueError, parser=parse, **kwargs):
    """Parse a config that should fail; returns the message."""
    with pytest.raises(kind) as raised:
        parser(tmp_path, **kwargs)
    return " ".join(str(raised.value).split())


# ---------------------------------------------------------------------------
# A good config
# ---------------------------------------------------------------------------

def test_good_fasta_config(tmp_path):
    config = parse(tmp_path)
    assert config["user models"] is None
    assert config["replicates"] == 10 and config["seed"] == 1234
    assert config["max migration events"] == 1
    assert config["mutation rate"] == [5e-9, 5e-8]
    assert config["migration rate"] == [1e-5, 1e-4]
    assert config["substitution model"] == "JC69"
    assert config["symmetric"] is True and config["divergence with gene flow"] is False
    assert len(config["species tree"]) == 1 and len(config["migration df"]) == 1
    assert len(config["fastas"]) == 10 and len(config["lengths"]) == 10
    assert config["population dictionary"]["n0"] == "A"


def test_populations_are_in_popfile_order(tmp_path):
    # tests/populations.txt lists 12 A, then 8 B, then 10 C
    assert list(parse(tmp_path)["sampling dict"].items()) == [("A", 12), ("B", 8), ("C", 10)]


def test_alignments_are_read_in_alphabetical_order(tmp_path):
    folder = tmp_path / "aln"
    folder.mkdir()
    for name in ["alignment_2.fa", "alignment_0.fa", "alignment_1.fa"]:
        shutil.copy(os.path.join(TESTS, "alignments", name), folder / name)
    lengths = [len(open(folder / name).read().split("\n")[1])
               for name in ["alignment_0.fa", "alignment_1.fa", "alignment_2.fa"]]
    assert parse(tmp_path, Data__alignments="aln")["lengths"] == lengths


def test_fas_files_and_other_files(tmp_path):
    folder = tmp_path / "aln"
    folder.mkdir()
    for i in range(3):
        shutil.copy(os.path.join(TESTS, "alignments", f"alignment_{i}.fa"), folder / f"alignment_{i}.fas")
    (folder / "notes.txt").write_text("ignored")
    assert len(parse(tmp_path, Data__alignments="aln")["fastas"]) == 3


def test_parsed_config_builds_models(tmp_path):
    config = parse(tmp_path, Model__migration_rate="U(0.5Nm, 2Nm)")
    builder = ModelBuilder(config)
    divergence, secondary_contact, gene_flow = builder.build_models()
    models, labels, _ = builder.draw_parameters(divergence, secondary_contact, gene_flow)
    assert len(models) == len(labels) > 0


# ---------------------------------------------------------------------------
# Settings that can be None or left out
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["None", "none", "NONE", ""])
def test_user_models_none_in_any_case(tmp_path, value):
    assert parse(tmp_path, Model__user_models=value)["user models"] is None


def test_user_models_do_not_need_a_tree_or_migration_matrix(tmp_path):
    config = parse(tmp_path, Model__user_models="my_models", Model__species_tree_file=None,
                   Model__migration_matrix=None)
    assert config["user models"] == "my_models" and "species tree" not in config


@pytest.mark.parametrize("alignments", ["None", "none", None])
def test_vcf_with_alignments_none_or_left_out(tmp_path, alignments):
    config = parse_vcf(tmp_path, "##contig=<ID=1,length=100>\n##contig=<ID=2,length=200>\n",
                       Data__alignments=alignments)
    assert config["lengths"] == [100, 200]


def test_neither_alignments_nor_vcf(tmp_path):
    message = error(tmp_path, Data__alignments="None")
    assert "needs alignments (a folder of fasta files) or vcf" in message


# ---------------------------------------------------------------------------
# VCF data
# ---------------------------------------------------------------------------

def test_vcf_samples_become_two_haplotypes(tmp_path):
    config = parse_vcf(tmp_path, "##contig=<ID=1,length=100>\n##contig=<ID=2,length=200>\n")
    assert list(config["sampling dict"].items()) == [("A", 2), ("B", 4)]
    assert config["population dictionary"] == {"n0_a": "A", "n0_b": "A", "n1_a": "B", "n1_b": "B",
                                                "n2_a": "B", "n2_b": "B"}


@pytest.mark.parametrize("header, lengths", [
    # contigs in the header without data are left out
    ("".join(f"##contig=<ID={i},length={i}00>\n" for i in range(1, 6)), [100, 200]),
    # extra fields, and fields in any order
    ("##contig=<ID=1,length=100,assembly=x>\n##contig=<length=200,ID=2>\n", [100, 200]),
    # header order doesn't matter: lengths follow the order of the data
    ("##contig=<ID=2,length=200>\n##contig=<ID=1,length=100>\n", [100, 200]),
])
def test_vcf_contig_lengths_from_the_header(tmp_path, header, lengths):
    assert parse_vcf(tmp_path, header)["lengths"] == lengths


def test_vcf_length_setting(tmp_path):
    # a header line that mentions "length" but isn't a contig line is ignored
    header = '##INFO=<ID=L,Number=1,Type=Integer,Description="Read length">\n'
    assert parse_vcf(tmp_path, header, Data__length="500")["lengths"] == [500, 500]


def test_vcf_header_lengths_and_length_setting_together(tmp_path):
    assert parse_vcf(tmp_path, "##contig=<ID=1,length=100>\n", Data__length="50")["lengths"] == [100, 50]


def test_vcf_contig_without_a_length(tmp_path):
    message = error(tmp_path, parser=parse_vcf, header="##contig=<ID=1,length=100>\n")
    assert "The VCF has no length for contig(s) 2" in message


# ---------------------------------------------------------------------------
# The popfile
# ---------------------------------------------------------------------------

def test_spaces_around_popfile_names_are_removed(tmp_path):
    popfile = "individual \t population\nn0 \t A\nn1\tB \nn2\tB\n"
    config = parse_vcf(tmp_path, "##contig=<ID=1,length=100>\n##contig=<ID=2,length=200>\n",
                       files={"pop.txt": popfile})
    assert list(config["sampling dict"]) == ["A", "B"]
    assert config["population dictionary"]["n0_a"] == "A"


def test_sample_names_that_are_numbers(tmp_path):
    files = {"data.vcf": vcf("##contig=<ID=1,length=100>\n##contig=<ID=2,length=200>\n")
             .replace("\tn0\tn1\tn2", "\t1\t2\t3"),
             "pop.txt": "individual\tpopulation\n1\tA\n2\tB\n3\tB\n"}
    config = parse_vcf(tmp_path, files=files)
    assert config["population dictionary"]["1_a"] == "A"


@pytest.mark.parametrize("popfile, message", [
    ("individual,population\nn0,A\nn1,B\nn2,B\n", "must be tab-separated, with a header row"),
    ("individual\tpopulation\nn0\tA\tx\nn1\tB\tx\nn2\tB\tx\n", "more than two columns"),
    ("individual\tpopulation\nn0\tA\nn1\tB\nn2\n", "missing an individual or a population"),
    ("individual\tpopulation\nn0\tA\nn1\tB\nn2\tB\nn9\tB\n", "in the popfile but not in the vcf file: n9"),
    ("individual\tpopulation\nn0\tA\nn1\tB\n", "in the vcf file but not in the popfile: n2"),
])
def test_bad_popfiles(tmp_path, popfile, message):
    header = "##contig=<ID=1,length=100>\n##contig=<ID=2,length=200>\n"
    assert message in error(tmp_path, parser=parse_vcf, header=header, files={"pop.txt": popfile})


def test_folder_without_alignments(tmp_path):
    (tmp_path / "aln").mkdir()
    assert "There are no .fa, .fas, or .fasta files" in error(tmp_path, Data__alignments="aln")


# ---------------------------------------------------------------------------
# Species trees
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tree, message", [
    (TREE.replace("C[&ne=10000-50000]", "C"), "C needs ne=min-max with whole numbers"),
    (TREE.replace("AB[&ne=10000-50000,div=10000-50000]", "AB[&ne=10000-50000]"), "AB needs div=min-max"),
    (TREE.replace("10000-50000", "1e4-5e4"), "A needs ne=min-max with whole numbers"),
    (TREE.replace("A[&ne=10000-50000]", "A[&ne=50000-10000]"), "the ne prior of A is 50000-10000"),
    (TREE.replace("div=100000-500000", "div=500000-100000"), "the div prior of ABC is 500000-100000"),
    (TREE.replace(")AB[", ")["), "an internal node has no name"),
])
def test_bad_tree_priors(tmp_path, tree, message):
    assert message in error(tmp_path, files={"tree.nex": nexus(tree)})


def test_different_ne_priors_need_constant_ne_false(tmp_path):
    tree = TREE.replace("C[&ne=10000-50000]", "C[&ne=1000-5000]")
    files = {"tree.nex": nexus(TREE, tree), "migration2.csv": MIGRATION}
    message = error(tmp_path, files=files, Model__migration_matrix="migration.csv;migration2.csv")
    assert "species tree 2 has different ne priors (1000-5000, 10000-50000)" in message
    config = parse(tmp_path, files=files, Model__migration_matrix="migration.csv; migration2.csv",
                   Model__constant_Ne="False")
    assert len(config["species tree"]) == 2 and len(config["migration df"]) == 2


def test_empty_tree_file(tmp_path):
    assert "has no trees" in error(tmp_path, files={"tree.nex": "#NEXUS\nBEGIN TREES;\nEND;\n"})


# ---------------------------------------------------------------------------
# Migration matrices
# ---------------------------------------------------------------------------

def test_one_migration_matrix_per_tree(tmp_path):
    message = error(tmp_path, files={"tree.nex": nexus(TREE, TREE)})
    assert "There are 2 species trees but 1 migration tables" in message


@pytest.mark.parametrize("matrix, message", [
    (MIGRATION.replace("C", "D"), "The rows of the migration table migration.csv are A, B, D, AB, ABD"),
    (",A,B,C,AB\nA,F,F,T,F\nB,F,F,F,F\nC,T,F,F,T\nAB,F,F,T,F\n", "They must match"),
    (MIGRATION.replace("A,F,F,T,F,F", "A,F,,T,F,F"), "should only contain T or F, but it has nan"),
    (MIGRATION.replace("A,F,F,T,F,F", "A,F,yes,T,F,F"), "should only contain T or F, but it has yes"),
])
def test_bad_migration_matrices(tmp_path, matrix, message):
    assert message in error(tmp_path, files={"migration.csv": matrix})


def test_spaces_around_migration_matrix_names_are_removed(tmp_path):
    matrix = MIGRATION.replace(",A,B", ", A ,B").replace("\nB,", "\n B ,")
    table = parse(tmp_path, files={"migration.csv": matrix})["migration df"][0]
    assert list(table.index) == list(table.columns) == ["A", "B", "C", "AB", "ABC"]


def test_true_and_false_work_in_migration_matrices(tmp_path):
    matrix = MIGRATION.replace(",T,", ",TRUE,").replace(",F,", ",false,")
    parse(tmp_path, files={"migration.csv": matrix})


# ---------------------------------------------------------------------------
# Priors and other settings
# ---------------------------------------------------------------------------

def test_nm_migration_prior(tmp_path):
    assert parse(tmp_path, Model__migration_rate="U(0.5Nm, 2Nm)")["migration rate"] == ["0.5Nm", "2Nm"]


@pytest.mark.parametrize("changes, message", [
    ({"Simulations__mutation_rate": "U(5e-8, 5e-9)"}, "Put the smaller value first"),
    ({"Simulations__mutation_rate": "5e-9"}, "Write it as U(5e-9, 5e-8)"),
    ({"Simulations__mutation_rate": "5e-9-5e-8"}, "Write it as U(5e-9, 5e-8)"),
    ({"Simulations__mutation_rate": "U(1Nm, 2Nm)"}, "Nm values are only allowed for migration rates"),
    ({"Model__migration_rate": "U(1e-5,abc)"}, "Write it as U(1e-5, 1e-4) or U(0.5Nm, 2Nm)"),
    ({"Model__migration_rate": "U(0.5Nm, 1e-4)"}, "Both values must be rates, or both Nm values"),
    ({"Model__migration_rate": "U(-1e-5, 1e-4)"}, "can't be negative"),
    ({"Other__replicates": "0"}, "'replicates' in [Other] is 0, but it must be at least 1"),
    ({"Other__replicates": "ten"}, "'replicates' in [Other] is 'ten', but it must be a whole number"),
    ({"Other__seed": "-1"}, "'seed' in [Other] is -1, but it must be at least 0"),
    ({"Model__max_migration_events": "-1"}, "'max migration events' in [Model] is -1"),
    ({"Simulations__substitution_model": "HKY"}, "The substitution model is 'HKY'. Use JC69 or binary"),
])
def test_bad_settings(tmp_path, changes, message):
    assert message in error(tmp_path, **changes)


@pytest.mark.parametrize("model", ["JC69", "jc69", "binary"])
def test_substitution_models(tmp_path, model):
    assert parse(tmp_path, Simulations__substitution_model=model)["substitution model"] == model


# ---------------------------------------------------------------------------
# Missing settings and files
# ---------------------------------------------------------------------------

def test_missing_setting(tmp_path):
    message = error(tmp_path, KeyError, Model__max_migration_events=None)
    assert "Missing key in configuration file: 'max migration events'" in message


@pytest.mark.parametrize("changes", [{"Model__species_tree_file": "nope.nex"}, {"Data__popfile": "nope.txt"},
                                     {"Model__migration_matrix": "nope.csv"}])
def test_missing_files(tmp_path, changes):
    assert "nope" in error(tmp_path, FileNotFoundError, **changes)


def test_missing_config_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        ModelConfigParser(str(tmp_path / "nope.txt")).parse_config()
