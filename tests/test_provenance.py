"""inventory/provenance: FreeSurfer version, recon-all flags, truncation and hole counts.

The files in ``tests/data/recon_logs/`` are cut from the logs of two real subjects by
``make_excerpts.py`` there (``fs6_*``: ds000102 sub-01, FreeSurfer 6.0.1; ``fs7_*``: ds003020
sub-UTS01, FreeSurfer 7.3.2), with paths and the host replaced. ``plain_recon-all.log`` is
written by hand. The network tests at the end read the six whole logs and check that the
excerpts stand for them.
"""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Callable
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from surfcensus.inventory.provenance import (
    EDIT_FLAGS,
    BuildStamp,
    Provenance,
    ReconLog,
    decode_text,
    parse_build_stamp,
    parse_recon_log,
    subject_provenance,
)
from surfcensus.sources.base import FileRef, FileRole

DATA = Path(__file__).parent / "data" / "recon_logs"

FS6_STAMP = "freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.1-f53a55a"
FS7_STAMP = "freesurfer-linux-ubuntu22_x86_64-7.3.2-20220804-6354275"
TEMPLATE_STAMP = "freesurfer-x86_64-redhat-linux-gnu-dev4-20090216"

MAIN, LH, RH, STAMP = (
    FileRole.RECON_LOG,
    FileRole.RECON_LOG_LH,
    FileRole.RECON_LOG_RH,
    FileRole.BUILD_STAMP,
)
SWITCHES = ("t2pial", "flairpial", "hires", "expert", "xopts", "edited")

FLAG = re.compile(r"-[A-Za-z][A-Za-z0-9_-]*")
ACCESSION = re.compile(r"ds\d{6}")


# ---------------------------------------------------------------------------------------
# helpers: logs in the two layouts of the real ones


def fs6_header(arguments: str, subject: str = "sub-01", stamp: str = FS6_STAMP) -> str:
    """What recon-all 6 writes at the start of an invocation: the arguments on their own line."""
    return (
        "Thu Feb 10 07:12:51 UTC 2022\n"
        f"/data/ds999999-fmriprep/sourcedata/freesurfer/{subject}\n"
        "/opt/freesurfer/bin/recon-all\n"
        f"{arguments}\n"
        f"subjid {subject}\n"
        "setenv SUBJECTS_DIR /data/ds999999-fmriprep/sourcedata/freesurfer\n"
        "FREESURFER_HOME /opt/freesurfer\n"
        "Actual FREESURFER_HOME /opt/freesurfer\n"
        f"build-stamp.txt: {stamp}\n"
        "Linux HOST 3.10.0-1160.45.1.el7.x86_64 #1 SMP x86_64 GNU/Linux\n"
    )


def fs7_header(arguments: str, subject: str = "sub-01", stamp: str = FS7_STAMP) -> str:
    """recon-all 7: the arguments after the path of the script, then a blank line."""
    return (
        "\n"
        "Sat Apr 22 01:24:39 UTC 2023\n"
        f"cd /data/ds999999-fmriprep/sourcedata/freesurfer/{subject}\n"
        "setenv SUBJECTS_DIR /data/ds999999-fmriprep/sourcedata/freesurfer\n"
        f"/opt/freesurfer/bin/recon-all {arguments}\n"
        "\n"
        f"subjid {subject}\n"
        "setenv SUBJECTS_DIR /data/ds999999-fmriprep/sourcedata/freesurfer\n"
        "FREESURFER_HOME /opt/freesurfer\n"
        "Actual FREESURFER_HOME /opt/freesurfer\n"
        f"build-stamp.txt: {stamp}\n"
        "Linux HOST 3.10.0-1160.45.1.el7.x86_64 #1 SMP x86_64 GNU/Linux\n"
    )


HEADERS: dict[str, Callable[[str], str]] = {"fs6": fs6_header, "fs7": fs7_header}
FINISHED = "recon-all -s sub-01 finished without error at Thu Feb 10 07:17:34 UTC 2022\n"
ERRORS = "recon-all -s sub-01 exited with ERRORS at Thu Feb 10 07:17:34 UTC 2022\n"


def truncation_line(first: int, second: int) -> str:
    return (
        "125000 of 148332 vertices processed\n"
        f"thickness calculation complete, {first}:{second} truncations.\n"
        "42393 vertices at 0 distance\n"
    )


def holes_line(lh: int, rh: int) -> str:
    return f"orig.nofix lheno =  -80, rheno = -50\norig.nofix lhholes = {lh:4d}, rhholes = {rh}\n"


def log(*invocations: str, layout: str = "fs6", body: str = "", end: str = FINISHED) -> str:
    """A log of several invocations, each given by its arguments, each with ``body`` and ``end``."""
    separator = "\n\n\nNew invocation of recon-all \n\n\n\n"
    return separator.join(HEADERS[layout](arguments) + body + end for arguments in invocations)


def pial_ref(name: str, variant: str | None = None) -> dict[FileRole, FileRef]:
    return {
        FileRole.LH_PIAL: FileRef(
            role=FileRole.LH_PIAL,
            location=f"surf/{name}",
            size=1,
            etag_or_mtime="1",
            variant=variant,
        )
    }


def log_ref(role: FileRole) -> FileRef:
    return FileRef(role=role, location=f"scripts/{role.value}.log", size=1, etag_or_mtime="1")


def provenance(*invocations: str, layout: str = "fs6") -> Provenance:
    """Of a subject whose only text is a ``recon-all.log`` with these invocations."""
    return subject_provenance({MAIN: log(*invocations, layout=layout)}, {}, "sub-01")


def excerpt(name: str) -> str:
    return decode_text((DATA / name).read_bytes())


def excerpts(prefix: str) -> dict[FileRole, str]:
    return {
        MAIN: excerpt(f"{prefix}_recon-all.log"),
        LH: excerpt(f"{prefix}_recon-all-lh.log"),
        RH: excerpt(f"{prefix}_recon-all-rh.log"),
    }


# ---------------------------------------------------------------------------------------
# build stamps


@pytest.mark.parametrize(
    ("stamp", "version", "major"),
    [
        # read in the bucket: ds000102 sub-01, ds003020 sub-UTS01, and the fsaverage next to
        # the subjects of ds000102
        (FS6_STAMP, "6.0.1", 6),
        (FS7_STAMP, "7.3.2", 7),
        (TEMPLATE_STAMP, None, None),
        # the form of other releases, from memory of their stamps, not read in a file
        ("freesurfer-Linux-centos6_x86_64-stable-pub-v5.3.0", "5.3.0", 5),
        ("freesurfer-Linux-centos4_x86_64-stable-pub-v5.3.0-HCP", "5.3.0", 5),
        ("freesurfer-Linux-centos6_x86_64-stable-pub-v5.3", "5.3", 5),
        ("freesurfer-linux-centos7_x86_64-7.1.1-20200723-8b40551", "7.1.1", 7),
        ("freesurfer-linux-centos7_x86_64-dev-20210310-b3d7d04", None, None),
        ("freesurfer-local-build-20190101", None, None),
        ("7.4.1", "7.4.1", 7),
        ("v10.2.0-rc", "10.2.0", 10),
    ],
)
def test_the_version_in_a_build_stamp(stamp: str, version: str | None, major: int | None) -> None:
    assert parse_build_stamp(stamp + "\n") == BuildStamp(raw=stamp, version=version, major=major)


@pytest.mark.parametrize(
    "stamp",
    [
        "x86_64",  # an architecture is not a version
        "freesurfer-6-stable",  # one number
        "freesurfer-6.0.1beta-x",  # not a whole field
        "freesurfer-a6.0.1-x",
        "freesurfer-6.0.1.2.3-x",  # more than three parts
        "freesurfer-v" + "9" * 5000 + ".1-x",  # a number int() would refuse
    ],
)
def test_a_stamp_without_a_whole_version_field_has_no_version(stamp: str) -> None:
    parsed = parse_build_stamp(stamp)
    assert (parsed.raw, parsed.version, parsed.major) == (stamp, None, None)


def test_the_stamp_is_the_first_non_blank_line_stripped() -> None:
    assert parse_build_stamp(f"\n  \r\n  {FS6_STAMP}  \r\n{FS7_STAMP}\n").raw == FS6_STAMP
    assert parse_build_stamp(f"\n  \r\n  {FS6_STAMP}  \r\n{FS7_STAMP}\n").version == "6.0.1"


@pytest.mark.parametrize("text", ["", "\n", "  \n\t\n"])
def test_an_empty_build_stamp(text: str) -> None:
    assert parse_build_stamp(text) == BuildStamp(raw="", version=None, major=None)


# ---------------------------------------------------------------------------------------
# the excerpts of the real logs

FS6_INVOCATIONS = (
    ("-autorecon1", "-i", "-noskullstrip", "-openmp", "-subjid", "-sd"),
    ("-autorecon2-volonly", "-openmp", "-subjid", "-sd"),
    ("-parallel", "-subjid", "-sd", "-cortribbon"),
    (
        *("-autorecon3", "-openmp", "-subjid", "-sd", "-nosphere", "-nosurfreg"),
        *("-nojacobian_white", "-noavgcurv", "-nocortparc", "-nopial", "-noparcstats"),
        *("-nocortparc2", "-noparcstats2", "-nocortparc3", "-noparcstats3", "-nopctsurfcon"),
        *("-nocortribbon", "-nobalabels"),
    ),
)
FS7_INVOCATIONS = (
    (
        *("-autorecon1", "-i", "-noskullstrip", "-noT2pial", "-noFLAIRpial"),
        *("-openmp", "-subjid", "-sd"),
    ),
    ("-openmp", "-subjid", "-sd", "-gcareg"),
    ("-autorecon2-volonly", "-openmp", "-subjid", "-sd", "-nogcareg"),
    ("-parallel", "-subjid", "-sd", "-cortribbon"),
    FS6_INVOCATIONS[3],
)


def hemi_invocations(*hemi_flags: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The two invocations of a per-hemisphere log; FreeSurfer 7 adds ``-?h-only`` to both."""
    first = (
        *("-autorecon-hemi", "-noparcstats", "-noparcstats2", "-noparcstats3"),
        *("-nohyporelabel", "-nobalabels", *hemi_flags, "-openmp", "-subjid", "-sd"),
    )
    second = (
        *("-autorecon-hemi", "-nohyporelabel", *hemi_flags, "-openmp", "-subjid", "-sd"),
        *("-notessellate", "-nosmooth1", "-noinflate1", "-noqsphere", "-nofix", "-nowhite"),
        *("-nosmooth2", "-noinflate2", "-nocurvHK", "-nocurvstats", "-nosphere", "-nosurfreg"),
        *("-nojacobian_white", "-noavgcurv", "-nocortparc", "-nopial", "-nocortparc2"),
        *("-nocortparc3", "-nopctsurfcon"),
    )
    return first, second


def test_the_freesurfer_6_log_has_four_invocations_with_the_arguments_on_their_own_line() -> None:
    parsed = parse_recon_log(excerpt("fs6_recon-all.log"))
    assert parsed == ReconLog(
        invocations=FS6_INVOCATIONS,
        build_stamp=FS6_STAMP,
        truncations=(),  # under fMRIPrep they are in the per-hemisphere logs only
        holes=(41, 26),
        finished=True,
    )


def test_the_freesurfer_7_log_has_five_invocations_with_the_arguments_after_the_script() -> None:
    parsed = parse_recon_log(excerpt("fs7_recon-all.log"))
    assert parsed == ReconLog(
        invocations=FS7_INVOCATIONS,
        build_stamp=FS7_STAMP,
        truncations=(),
        holes=(4, 15),
        finished=True,
    )
    # the script path is not an argument, and no line says "New invocation" without the "#"
    assert "New invocation of recon-all" not in excerpt("fs7_recon-all.log")
    assert "#New# invocation of recon-all" in excerpt("fs7_recon-all.log")


@pytest.mark.parametrize(
    ("name", "hemi_flags", "truncations"),
    [
        ("fs6_recon-all-lh.log", (), (39, 225)),
        ("fs6_recon-all-rh.log", (), (93, 434)),
        ("fs7_recon-all-lh.log", ("-lh-only",), (22, 230)),
        ("fs7_recon-all-rh.log", ("-rh-only",), (23, 325)),
    ],
)
def test_the_per_hemisphere_logs_hold_two_invocations_and_the_truncation_counts(
    name: str, hemi_flags: tuple[str, ...], truncations: tuple[int, int]
) -> None:
    parsed = parse_recon_log(excerpt(name))
    assert parsed.invocations == hemi_invocations(*hemi_flags)
    assert parsed.truncations == (truncations,)
    assert parsed.holes is None
    assert parsed.finished is True


def test_the_hand_written_plain_log_has_one_invocation_and_both_truncation_lines() -> None:
    assert parse_recon_log(excerpt("plain_recon-all.log")) == ReconLog(
        invocations=(("-s", "-i", "-all"),),
        build_stamp=FS6_STAMP,
        truncations=((12, 140), (30, 201)),
        holes=(21, 12),
        finished=True,
    )


def test_provenance_of_the_freesurfer_6_subject() -> None:
    texts = {STAMP: FS6_STAMP + "\n", **excerpts("fs6")}
    found = subject_provenance(texts, pial_ref("lh.pial", "lh.pial"), "sub-01")
    flags = {flag for invocation in (*FS6_INVOCATIONS, *hemi_invocations()) for flag in invocation}
    assert found == Provenance(
        build_stamp=FS6_STAMP,
        fs_version="6.0.1",
        fs_major=6,
        flags=tuple(sorted(flags)),
        n_invocations=8,
        finished=True,
        t2pial=False,
        flairpial=False,
        hires=False,
        expert=False,
        xopts=False,
        edited=False,
        pial_variant="pial",
        holes=(41, 26),
        truncations={"lh": (39, 225), "rh": (93, 434)},
        participant="sub-01",
        longitudinal=False,
    )
    assert len(found.flags) == 36


def test_provenance_of_the_freesurfer_7_subject() -> None:
    texts = {STAMP: FS7_STAMP + "\n", **excerpts("fs7")}
    found = subject_provenance(texts, pial_ref("lh.pial.T1", "lh.pial.T1"), "sub-UTS01")
    assert (found.fs_version, found.fs_major, found.build_stamp) == ("7.3.2", 7, FS7_STAMP)
    assert found.n_invocations == 9
    assert found.finished is True
    # fMRIPrep passes -noT2pial -noFLAIRpial: the flag is in the log and the answer is no
    assert {"-noT2pial", "-noFLAIRpial"} <= set(found.flags)
    assert (found.t2pial, found.flairpial) == (False, False)
    assert (found.hires, found.expert, found.xopts, found.edited) == (False, False, False, False)
    assert found.pial_variant == "pial.T1"
    assert found.holes == (4, 15)
    assert found.truncations == {"lh": (22, 230), "rh": (23, 325)}
    assert found.participant == "sub-UTS01"
    assert len(found.flags) == 42


@pytest.mark.parametrize("prefix", ["fs6", "fs7"])
def test_no_path_and_no_accession_reaches_the_flags_of_the_real_logs(prefix: str) -> None:
    texts = excerpts(prefix)
    assert all("/data/ds999999-fmriprep/" in text for text in texts.values())  # the canary
    found = subject_provenance(texts, {}, "sub-01")
    assert found.flags == tuple(sorted(set(found.flags)))
    for flag in found.flags:
        assert FLAG.fullmatch(flag), flag
        assert "/" not in flag and not ACCESSION.search(flag) and "sub-" not in flag


def test_the_excerpts_name_no_person_and_no_machine() -> None:
    names = sorted(path.name for path in DATA.glob("*.log"))
    assert names == [
        *(f"fs{v}_recon-all{h}.log" for v in (6, 7) for h in ("-lh", "-rh", "")),
        "plain_recon-all.log",
    ]
    allowed = ("/data/ds999999", "/data/work/ds999999_sub-", "/opt/freesurfer", "/usr/local/")
    for name in names:
        text = (DATA / name).read_text(encoding="utf-8")
        assert len(text) < 5000, name
        assert not re.search(r"scratch|tacc|frontera|utexas|\w@\w|/home|/Users", text, re.I), name
        assert set(ACCESSION.findall(text)) <= {"ds999999"}, name
        for token in text.split():
            if token.startswith("/"):
                assert token.startswith(allowed), (name, token)
        for line in text.splitlines():
            if line.startswith("Linux "):
                assert line.split()[1] == "HOST", name


# ---------------------------------------------------------------------------------------
# invocations and flag names


@pytest.mark.parametrize("layout", ["fs6", "fs7"])
@pytest.mark.parametrize("n", [1, 4, 5])
def test_each_subjid_line_is_one_invocation_in_both_layouts(layout: str, n: int) -> None:
    arguments = [f"-step{i} -subjid sub-01 -sd /data/ds999999/x -openmp {i}" for i in range(n)]
    parsed = parse_recon_log(log(*arguments, layout=layout))
    assert parsed.invocations == tuple((f"-step{i}", "-subjid", "-sd", "-openmp") for i in range(n))


def test_the_arguments_are_the_nearest_non_blank_line_above_subjid() -> None:
    text = (
        "/x/recon-all -a -b\n\n   \n\t\nsubjid s\n-c\nsubjid s\n-d /bin/recon-all -e\nsubjid\t s \n"
    )
    assert parse_recon_log(text).invocations == (("-a", "-b"), ("-c",), ("-d", "-e"))


def test_only_a_first_token_that_ends_in_recon_all_is_dropped() -> None:
    assert parse_recon_log("recon-all -all\nsubjid s\n").invocations == (("-all",),)
    assert parse_recon_log("/opt/fs/bin/recon-all -all\nsubjid s\n").invocations == (("-all",),)
    # recon-all has no such flag; the point is that a flag-shaped first token is an argument
    assert parse_recon_log("-norecon-all -all\nsubjid s\n").invocations == (("-all",),)
    assert parse_recon_log("-all /opt/fs/bin/recon-all\nsubjid s\n").invocations == (("-all",),)
    assert parse_recon_log("-recon -all\nsubjid s\n").invocations == (("-recon", "-all"),)


@pytest.mark.parametrize(
    "line",
    [
        "subjid",
        "subjid ",
        "subjid a b",
        " subjid a",
        "subjid: a",
        "Subjid a",
        "xsubjid a",
        "subjida",
    ],
)
def test_lines_that_are_not_a_subjid_line_start_no_invocation(line: str) -> None:
    assert parse_recon_log(f"-all\n{line}\n").invocations == ()


def test_a_subjid_line_without_arguments_above_is_an_invocation_without_flags() -> None:
    assert parse_recon_log("subjid s\n").invocations == ((),)
    assert parse_recon_log("\n\nsubjid s\nsubjid s\n").invocations == ((), ())
    # the line above the second one is the first, which holds no flag
    assert parse_recon_log("-a\nsubjid s\nsubjid s\n").invocations == (("-a",), ())


@pytest.mark.parametrize(
    ("token", "kept"),
    [
        ("-all", True),
        ("-T2pial", True),
        ("-autorecon2-volonly", True),
        ("-nojacobian_white", True),
        ("-cw256", True),
        ("-3T", True),  # the one flag of recon-all that starts with a digit
        ("-3t", True),
        ("-7T", False),
        ("-3", False),
        ("-1", False),
        ("-0.5", False),
        ("7", False),
        ("-", False),
        ("--all", False),
        ("-all=1", False),
        ("-a.b", False),
        ("-aparc+aseg-stats", False),
        ("all", False),
        ("sub-01", False),
        ("-sub-01", True),  # flag-shaped: a name that starts with a dash cannot be told apart
        ("/data/ds000102-fmriprep/sourcedata/freesurfer", False),
        ("-/data/x", False),
        ("-a/b", False),
        ("ds000102", False),
        ("-ds000102", False),
        ("-ds000102-fmriprep", False),
        ("-Xds000102", False),
        ("-ds00010", True),  # five digits are not an accession
        ("-\u00e9", False),
        ("-a\u00e9", False),
    ],
)
@pytest.mark.parametrize("layout", ["fs6", "fs7"])
def test_which_tokens_are_flag_names(token: str, kept: bool, layout: str) -> None:
    parsed = parse_recon_log(log(f"-first {token} -last", layout=layout))
    assert parsed.invocations == (("-first", token, "-last") if kept else ("-first", "-last"),)


def test_values_never_reach_the_flags() -> None:
    arguments = (
        "-i /data/ds000102/sub-01/anat/sub-01_T1w.nii.gz -s sub-01 -sd /data/ds000102 "
        "-expert /home/someone/my -ds000102 opts.txt -openmp 8 -T2 /data/t2.mgz -hemi lh"
    )
    found = provenance(arguments)
    assert found.flags == ("-T2", "-expert", "-hemi", "-i", "-openmp", "-s", "-sd")


def test_flags_are_the_sorted_union_over_all_invocations_of_all_logs() -> None:
    texts = {MAIN: log("-b -a", "-c -a"), LH: log("-lh-only -a"), RH: log("-rh-only -B")}
    found = subject_provenance(texts, {}, "sub-01")
    assert found.flags == ("-B", "-a", "-b", "-c", "-lh-only", "-rh-only")
    assert found.n_invocations == 4


# ---------------------------------------------------------------------------------------
# the switches

T2 = ("t2pial", "-T2pial", "-noT2pial")
FLAIR = ("flairpial", "-FLAIRpial", "-noFLAIRpial")


@pytest.mark.parametrize(("field", "on", "off"), [T2, FLAIR])
@pytest.mark.parametrize("layout", ["fs6", "fs7"])
def test_the_last_setting_of_a_command_line_decides_and_any_command_line_can_say_yes(
    field: str, on: str, off: str, layout: str
) -> None:
    def value(*invocations: str) -> bool | None:
        return getattr(provenance(*invocations, layout=layout), field)

    assert value("-all") is False
    assert value(f"-all {off}") is False
    assert value(f"-all {on}") is True
    assert value(f"{on} -all {off}") is False
    assert value(f"{off} -all {on}") is True
    assert value(f"{on} {off} {on} {off}") is False
    assert value(f"{on} {off}", "-autorecon3") is False
    assert value(f"{on} {off}", f"-autorecon3 {on}") is True  # a later invocation
    assert value(f"-all {on}", f"-autorecon3 {off}") is True  # an earlier one
    assert value("-all", "-autorecon2", f"-autorecon3 {on}") is True


@pytest.mark.parametrize(
    ("flag", "field", "value"),
    [
        # every spelling the argument parser of recon-all v6.0.0 and v7.4.1 takes
        ("-T2pial", "t2pial", True),
        ("-t2pial", "t2pial", True),
        ("-T2pial-only", "t2pial", True),
        ("-autorecon3-T2pial", "t2pial", True),
        ("-FLAIRpial", "flairpial", True),
        ("-flairpial", "flairpial", True),
        ("-hires", "hires", True),
        ("-highres", "hires", True),
        # whole tokens, case-sensitive: these are other tokens
        ("-T2PIAL", "t2pial", False),
        ("-t2Pial", "t2pial", False),
        ("-T2pialx", "t2pial", False),
        ("-xT2pial", "t2pial", False),
        ("-T2pial=1", "t2pial", False),
        ("--T2pial", "t2pial", False),
        ("T2pial", "t2pial", False),
        ("-T2", "t2pial", False),  # names the T2 image; the refinement is a flag of its own
        ("-FLAIR", "flairpial", False),
        ("-flairPial", "flairpial", False),
        ("-FLAIRPIAL", "flairpial", False),
        ("-Hires", "hires", False),
        ("-HIRES", "hires", False),
        ("-hires2", "hires", False),
        ("-conf2hires", "hires", False),
        ("-cm", "hires", False),
        ("-T2pial", "flairpial", False),
        ("-FLAIRpial", "t2pial", False),
        ("-hires", "t2pial", False),
    ],
)
def test_switch_flags_are_whole_tokens_and_case_sensitive(
    flag: str, field: str, value: bool
) -> None:
    assert getattr(provenance(f"-all {flag} -openmp 8"), field) is value


@pytest.mark.parametrize(
    ("on", "off", "field"),
    [
        *(("-T2pial", off, "t2pial") for off in ("-noT2pial", "-no-T2pial", "-no_T2pial")),
        *(("-t2pial", off, "t2pial") for off in ("-no-t2pial", "-no_t2pial")),
        *(("-FLAIRpial", o, "flairpial") for o in ("-noFLAIRpial", "-no-FLAIRpial")),
        *(("-flairpial", o, "flairpial") for o in ("-no_FLAIRpial", "-no-flairpial")),
        ("-FLAIRpial", "-no_flairpial", "flairpial"),
        ("-hires", "-nohires", "hires"),
    ],
)
def test_every_spelling_that_clears_a_switch(on: str, off: str, field: str) -> None:
    assert getattr(provenance(f"{on} -all {off}"), field) is False
    assert getattr(provenance(f"{off} -all {on}"), field) is True


def test_a_spelling_recon_all_does_not_take_clears_nothing() -> None:
    assert provenance("-T2pial -not2pial -NOT2pial -noT2Pial").t2pial is True
    assert provenance("-T2pial -noFLAIRpial").t2pial is True
    assert provenance("-FLAIRpial -noT2pial -noflairpial").flairpial is True


def test_a_switch_set_in_a_per_hemisphere_log_counts() -> None:
    texts = {MAIN: log("-autorecon1 -noT2pial"), LH: log("-autorecon-hemi lh -T2pial")}
    assert subject_provenance(texts, {}, "sub-01").t2pial is True


@pytest.mark.parametrize(
    ("name", "variant", "expected", "t2pial", "flairpial"),
    [
        ("lh.pial.T2", "lh.pial.T2", "pial.T2", True, False),
        ("lh.pial", "lh.pial.T2", "pial.T2", True, False),  # a link read under its target's name
        ("lh.pial.FLAIR", "lh.pial.FLAIR", "pial.FLAIR", False, True),
        ("lh.pial.T1", "lh.pial.T1", "pial.T1", False, False),
        ("lh.pial", "lh.pial", "pial", False, False),
        ("lh.pial.T2", None, "pial.T2", True, False),  # no variant recorded: the file name
        ("lh.pial", None, "pial", False, False),
        ("lh.pial.T2", "lh.mine", None, False, False),  # not a name of the role: not recorded
        ("lh.other", None, None, False, False),
    ],
)
def test_the_pial_variant_and_what_it_says_whatever_the_logs_say(
    name: str, variant: str | None, expected: str | None, t2pial: bool, flairpial: bool
) -> None:
    texts = {MAIN: log("-all -noT2pial -noFLAIRpial")}
    found = subject_provenance(texts, pial_ref(name, variant), "sub-01")
    assert found.pial_variant == expected
    assert (found.t2pial, found.flairpial) == (t2pial, flairpial)


def test_the_pial_variant_is_that_of_the_left_hemisphere_and_none_without_it() -> None:
    right = {
        FileRole.RH_PIAL: FileRef(
            role=FileRole.RH_PIAL,
            location="surf/rh.pial.T2",
            size=1,
            etag_or_mtime="1",
            variant="rh.pial.T2",
        )
    }
    found = subject_provenance({}, right, "sub-01")
    assert (found.pial_variant, found.t2pial) == (None, None)


def test_without_a_log_the_variant_can_say_yes_and_nothing_can_say_no() -> None:
    found = subject_provenance({}, pial_ref("lh.pial.T2", "lh.pial.T2"), "sub-01")
    assert (found.t2pial, found.flairpial, found.hires) == (True, None, None)
    found = subject_provenance({}, pial_ref("lh.pial.FLAIR", "lh.pial.FLAIR"), "sub-01")
    assert (found.t2pial, found.flairpial) == (None, True)
    found = subject_provenance({}, pial_ref("lh.pial.T1", "lh.pial.T1"), "sub-01")
    assert (found.t2pial, found.flairpial) == (None, None)


@pytest.mark.parametrize(
    ("arguments", "field"),
    [
        ("-all -expert /data/ds000102/opts.txt", "expert"),
        ("-all -xopts-use", "xopts"),
        ("-all -xopts-clean", "xopts"),
        ("-all -xopts-overwrite", "xopts"),
        *((f"{flag} -subjid sub-01", "edited") for flag in EDIT_FLAGS),
    ],
)
def test_expert_xopts_and_edit_flags(arguments: str, field: str) -> None:
    found = provenance("-autorecon1", arguments)
    assert [name for name in SWITCHES if getattr(found, name)] == [field]
    assert [name for name in SWITCHES if getattr(found, name) is False] == [
        name for name in SWITCHES if name != field
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        "-all",
        "-autorecon2 -autorecon3",
        "-autorecon2-volonly -autorecon2-perhemi -autorecon-hemi lh",
        "-autorecon2-cpx -xautorecon2-wm -AUTORECON2-WM -autorecon-Pial",
        "-Expert -experts -xopts -xoptsuse -Xopts-use",
        "-autorecon1 -i /data/-autorecon2-wm/-expert/x.nii",
        "-noexpert",
    ],
)
def test_what_is_not_an_expert_xopts_or_edit_flag(arguments: str) -> None:
    found = provenance(arguments)
    assert [getattr(found, name) for name in SWITCHES] == [False] * 6


def test_the_edit_flags_are_the_four_of_the_contract() -> None:
    assert EDIT_FLAGS == ("-autorecon2-cp", "-autorecon2-wm", "-autorecon2-pial", "-autorecon-pial")


# ---------------------------------------------------------------------------------------
# what was not seen is None, not False


def test_no_log_gives_none_everywhere() -> None:
    found = subject_provenance({}, {}, "sub-01")
    assert found == Provenance(
        build_stamp=None,
        fs_version=None,
        fs_major=None,
        flags=(),
        n_invocations=None,
        finished=None,
        t2pial=None,
        flairpial=None,
        hires=None,
        expert=None,
        xopts=None,
        edited=None,
        pial_variant=None,
        holes=None,
        truncations={"lh": None, "rh": None},
        participant="sub-01",
        longitudinal=False,
    )


@pytest.mark.parametrize("text", ["", "\n\n", "no invocation here\n-T2pial\n"])
def test_a_log_without_an_invocation_establishes_no_flag(text: str) -> None:
    found = subject_provenance({MAIN: text, LH: "", RH: ""}, {}, "sub-01")
    assert found.n_invocations == 0  # logs were read, unlike None
    assert found.flags == ()
    assert [getattr(found, name) for name in SWITCHES] == [None] * 6
    assert found.finished is None
    assert found.truncations == {"lh": None, "rh": None}


def test_other_roles_among_the_texts_are_ignored() -> None:
    texts = {FileRole.ASEG_STATS: log("-T2pial -expert x"), FileRole.LH_STATS: FS7_STAMP}
    assert subject_provenance(texts, {}, "sub-01") == subject_provenance({}, {}, "sub-01")


def test_a_log_the_subject_has_but_that_was_not_read_leaves_no_as_unknown() -> None:
    files = {role: log_ref(role) for role in (MAIN, LH, RH)}
    texts = {MAIN: log("-autorecon1 -hires"), LH: log("-autorecon-hemi lh")}

    unread = subject_provenance(texts, files, "sub-01")  # recon-all-rh.log could not be read
    assert unread.hires is True  # seen
    assert (unread.t2pial, unread.flairpial, unread.expert, unread.xopts, unread.edited) == (
        None,
        None,
        None,
        None,
        None,
    )
    assert unread.n_invocations == 2

    read = subject_provenance({**texts, RH: log("-autorecon-hemi rh")}, files, "sub-01")
    assert (read.hires, read.t2pial, read.edited) == (True, False, False)
    # a subject that has no per-hemisphere logs misses nothing
    plain = subject_provenance({MAIN: texts[MAIN]}, {MAIN: files[MAIN]}, "sub-01")
    assert (plain.hires, plain.t2pial, plain.edited) == (True, False, False)


# ---------------------------------------------------------------------------------------
# truncations, holes, the verdict


def test_truncations_come_from_each_hemisphere_s_own_log() -> None:
    texts = {
        MAIN: log("-all"),
        LH: log("-autorecon-hemi lh", body=truncation_line(39, 225)),
        RH: log("-autorecon-hemi rh", body=truncation_line(93, 434)),
    }
    assert parse_recon_log(texts[LH]).truncations == ((39, 225),)
    assert subject_provenance(texts, {}, "sub-01").truncations == {"lh": (39, 225), "rh": (93, 434)}


def test_the_last_truncation_line_of_a_hemisphere_log_is_the_one() -> None:
    lh = log("-autorecon-hemi lh", "-autorecon-hemi lh -T2pial", body=truncation_line(1, 2))
    lh += truncation_line(7, 8)
    assert parse_recon_log(lh).truncations == ((1, 2), (1, 2), (7, 8))  # every line, in order
    found = subject_provenance({LH: lh}, {}, "sub-01")
    assert found.truncations == {"lh": (7, 8), "rh": None}


def test_a_single_log_with_exactly_two_truncation_lines_is_left_then_right() -> None:
    main = log("-all", body=truncation_line(12, 140) + "more\n" + truncation_line(30, 201))
    found = subject_provenance({MAIN: main}, {}, "bert")
    assert found.truncations == {"lh": (12, 140), "rh": (30, 201)}


@pytest.mark.parametrize("n", [0, 1, 3, 4])
def test_a_single_log_with_another_number_of_truncation_lines_gives_none(n: int) -> None:
    main = log("-all", body="".join(truncation_line(i, i + 1) for i in range(n)))
    assert len(parse_recon_log(main).truncations) == n
    assert subject_provenance({MAIN: main}, {}, "bert").truncations == {"lh": None, "rh": None}


def test_a_hemisphere_log_wins_over_the_single_log_and_each_hemisphere_falls_back_alone() -> None:
    main = log("-all", body=truncation_line(12, 140) + truncation_line(30, 201))
    rh = log("-autorecon-hemi rh", body=truncation_line(93, 434))
    found = subject_provenance({MAIN: main, RH: rh, LH: log("-autorecon-hemi lh")}, {}, "s")
    assert found.truncations == {"lh": (12, 140), "rh": (93, 434)}


@pytest.mark.parametrize(
    "line",
    [
        "thickness calculation complete, 39:225 truncation.",
        "thickness calculation complete, 39;225 truncations.",
        "thickness calculation complete, :225 truncations.",
        "thickness calculation complete, -3:225 truncations.",
        "Thickness calculation complete, 39:225 truncations.",
        "thickness calculation complete, 39:1234567890123 truncations.",
        "thickness calculation complete, " + "9" * 5000 + ":1 truncations.",
    ],
)
def test_a_line_that_is_not_the_truncation_line_gives_no_counts(line: str) -> None:
    assert parse_recon_log(log("-all", body=line + "\n")).truncations == ()


def test_holes_are_those_of_the_last_holes_line() -> None:
    assert parse_recon_log(log("-all", body=holes_line(41, 26))).holes == (41, 26)
    assert parse_recon_log(log("-all", body=holes_line(41, 26) + holes_line(0, 3))).holes == (0, 3)
    assert parse_recon_log(log("-all", body="orig.nofix lheno =  -80, rheno = -50\n")).holes is None
    assert parse_recon_log(log("-all")).holes is None
    for line in (
        "orig.nofix lhholes = 41, rhholes = " + "9" * 5000,
        "orig.nofix lhholes = , rhholes = 26",
        "orig nofix lhholes = 41, rhholes = 26",
        "orig.nofix lhholes = 41 rhholes = 26",
    ):
        assert parse_recon_log(log("-all", body=line + "\n")).holes is None, line


def test_the_subject_s_holes_come_from_the_main_log_first() -> None:
    with_holes = log("-autorecon3", body=holes_line(41, 26))
    hemi = log("-autorecon-hemi lh", body=holes_line(1, 2))
    assert subject_provenance({MAIN: with_holes, LH: hemi}, {}, "s").holes == (41, 26)
    assert subject_provenance({MAIN: log("-all"), LH: hemi}, {}, "s").holes == (1, 2)
    assert subject_provenance({MAIN: log("-all")}, {}, "s").holes is None


@pytest.mark.parametrize(
    ("ends", "finished"),
    [
        ((FINISHED,), True),
        ((ERRORS,), False),
        ((ERRORS, FINISHED), True),  # a failed run, then a rerun that went through
        ((FINISHED, ERRORS), False),
        ((FINISHED, ERRORS + FINISHED), True),  # within one invocation the last line counts
        ((FINISHED, FINISHED + ERRORS), False),
        ((FINISHED, ""), None),  # the last invocation was killed: no verdict of its own
        ((ERRORS, ""), None),
        (("",), None),
        (("recon-all -s sub-01 Finished Without Error\n",), None),
        (("recon-all -s sub-01 exited with errors\n",), None),
    ],
)
def test_the_verdict_of_the_last_invocation_says_whether_the_main_log_finished(
    ends: tuple[str, ...], finished: bool | None
) -> None:
    text = "\n\n".join(log("-all", end=end) for end in ends)
    assert parse_recon_log(text).finished is finished
    assert subject_provenance({MAIN: text}, {}, "sub-01").finished is finished


def test_two_verdicts_on_one_line_are_read_in_their_order() -> None:
    good, bad = FINISHED.strip(), ERRORS.strip()
    assert parse_recon_log(log("-all", end=f"{bad} {good}\n")).finished is True
    assert parse_recon_log(log("-all", end=f"{good} {bad}\n")).finished is False
    assert parse_recon_log(log("-all", end=f"{good} {bad} {good}\n")).finished is True


def test_finished_is_read_from_the_main_log_alone() -> None:
    texts = {LH: log("-autorecon-hemi lh"), RH: log("-autorecon-hemi rh")}
    assert parse_recon_log(texts[LH]).finished is True
    assert subject_provenance(texts, {}, "sub-01").finished is None
    texts[MAIN] = log("-all", end=ERRORS)
    assert subject_provenance(texts, {}, "sub-01").finished is False


# ---------------------------------------------------------------------------------------
# version


def test_the_version_comes_from_the_stamp_file_and_else_from_the_main_log() -> None:
    main = log("-all", layout="fs7")  # its stamp line says 7.3.2
    assert parse_recon_log(main).build_stamp == FS7_STAMP

    both = subject_provenance({STAMP: FS6_STAMP + "\n", MAIN: main}, {}, "sub-01")
    assert (both.build_stamp, both.fs_version, both.fs_major) == (FS6_STAMP, "6.0.1", 6)
    only_log = subject_provenance({MAIN: main}, {}, "sub-01")
    assert (only_log.build_stamp, only_log.fs_version, only_log.fs_major) == (FS7_STAMP, "7.3.2", 7)
    only_file = subject_provenance({STAMP: FS6_STAMP}, {}, "sub-01")
    assert (only_file.build_stamp, only_file.fs_version, only_file.fs_major) == (
        FS6_STAMP,
        "6.0.1",
        6,
    )
    empty_file = subject_provenance({STAMP: "\n", MAIN: main}, {}, "sub-01")
    assert (empty_file.build_stamp, empty_file.fs_version) == (FS7_STAMP, "7.3.2")


def test_a_development_stamp_has_no_version_and_is_kept() -> None:
    found = subject_provenance({STAMP: TEMPLATE_STAMP + "\n"}, {}, "sub-01")
    assert (found.build_stamp, found.fs_version, found.fs_major) == (TEMPLATE_STAMP, None, None)
    # a log that names a version is then the better witness
    found = subject_provenance({STAMP: TEMPLATE_STAMP, MAIN: log("-all")}, {}, "sub-01")
    assert (found.build_stamp, found.fs_version, found.fs_major) == (FS6_STAMP, "6.0.1", 6)


def test_an_empty_stamp_file_is_no_stamp() -> None:
    assert subject_provenance({STAMP: ""}, {}, "sub-01").build_stamp is None
    assert subject_provenance({STAMP: " \n\n"}, {}, "sub-01").build_stamp is None
    main = log("-all").replace(FS6_STAMP, TEMPLATE_STAMP)
    found = subject_provenance({STAMP: "\n", MAIN: main}, {}, "sub-01")
    assert (found.build_stamp, found.fs_version, found.fs_major) == (TEMPLATE_STAMP, None, None)


def test_the_stamp_of_a_per_hemisphere_log_is_not_used_for_the_version() -> None:
    found = subject_provenance({LH: log("-autorecon-hemi lh")}, {}, "sub-01")
    assert (found.build_stamp, found.fs_version, found.fs_major) == (None, None, None)


def test_the_last_stamp_line_of_a_log_is_its_stamp() -> None:
    text = fs6_header("-autorecon1") + fs7_header("-autorecon2")
    assert parse_recon_log(text).build_stamp == FS7_STAMP
    assert parse_recon_log("build-stamp.txt:   \n").build_stamp is None
    assert parse_recon_log("x build-stamp.txt: abc\n").build_stamp is None
    assert parse_recon_log("build-stamp.txt: abc  \n").build_stamp == "abc"


# ---------------------------------------------------------------------------------------
# heads, odd bytes, empty files


def test_a_head_has_no_verdict_and_says_no_to_nothing() -> None:
    text = log("-autorecon1 -hires", "-autorecon2", body=truncation_line(3, 4))
    head = parse_recon_log(text, partial=True)
    assert head.finished is None
    assert parse_recon_log(text).finished is True
    assert (head.invocations, head.truncations) == (
        (("-autorecon1", "-hires"), ("-autorecon2",)),
        ((3, 4), (3, 4)),
    )

    found = subject_provenance({MAIN: text}, {}, "sub-01", partial=True)
    assert found.finished is None
    assert found.hires is True
    assert (found.t2pial, found.flairpial, found.expert, found.xopts, found.edited) == (
        None,
        None,
        None,
        None,
        None,
    )
    assert found.n_invocations == 2
    assert found.flags == ("-autorecon1", "-autorecon2", "-hires")


@pytest.mark.parametrize(
    ("cut", "field", "in_head", "in_whole"),
    [
        # the label may be cut short, so this is not yet an invocation
        ("subjid sub-0", "invocations", (("-autorecon1",),), (("-autorecon1",), ("-autorecon2",))),
        (
            "build-stamp.txt: freesurfer-linux-ubun",
            "build_stamp",
            FS6_STAMP,
            "freesurfer-linux-ubun",
        ),
        ("thickness calculation complete, 39:225 truncations.", "truncations", (), ((39, 225),)),
        ("orig.nofix lhholes =   41, rhholes = 2", "holes", None, (41, 2)),
    ],
)
def test_a_last_line_without_a_line_end_is_ignored_in_a_head(
    cut: str, field: str, in_head: object, in_whole: object
) -> None:
    text = fs6_header("-autorecon1") + "-autorecon2\n" + cut
    assert getattr(parse_recon_log(text, partial=True), field) == in_head
    assert getattr(parse_recon_log(text), field) == in_whole
    # with its line end the line is whole, and a head keeps it
    assert getattr(parse_recon_log(text + "\n", partial=True), field) == in_whole


def test_the_first_64_kb_of_a_long_log_show_only_its_first_invocations() -> None:
    body = ("x" * 99 + "\n") * 400  # 40 kB per invocation
    text = log("-autorecon1", "-autorecon2", "-autorecon3 -T2pial", body=body)
    head = text.encode()[:65536]
    assert len(text) > 65536
    found = subject_provenance({MAIN: decode_text(head)}, {}, "sub-01", partial=True)
    assert found.n_invocations == 2
    assert found.t2pial is None  # the third invocation would have said yes
    assert subject_provenance({MAIN: text}, {}, "sub-01").t2pial is True


def test_an_empty_file_parses_to_nothing() -> None:
    empty = ReconLog(invocations=(), build_stamp=None, truncations=(), holes=None, finished=None)
    assert parse_recon_log("") == empty
    assert parse_recon_log("", partial=True) == empty
    assert parse_recon_log(decode_text(b"")) == empty


def test_undecodable_bytes_are_replaced_and_the_rest_is_read() -> None:
    data = (
        b"\xff\xfe\x00junk \x80\x81\n"
        + fs6_header("-all -T2pial /data/caf\xe9/\xff.nii -hires").encode("latin-1")
        + b"\xc3\x28 \xa0\n"
        + truncation_line(5, 6).encode()
        + b"\xf0\x9f\x92\n"
        + holes_line(7, 8).encode()
        + FINISHED.encode()
        + b"\xe2\x82"
    )
    with pytest.raises(UnicodeDecodeError):
        data.decode("utf-8")
    text = decode_text(data)
    assert "\ufffd" in text
    assert parse_recon_log(text) == ReconLog(
        invocations=(("-all", "-T2pial", "-hires"),),
        build_stamp=FS6_STAMP,
        truncations=((5, 6),),
        holes=(7, 8),
        finished=True,
    )
    assert parse_build_stamp(decode_text(b"\xff" + FS6_STAMP.encode() + b"\n")).version == "6.0.1"


def test_a_byte_order_mark_is_dropped() -> None:
    assert decode_text(b"\xef\xbb\xbf" + FS6_STAMP.encode()) == FS6_STAMP
    data = b"\xef\xbb\xbfsubjid sub-01\n"
    assert parse_recon_log(decode_text(data)).invocations == ((),)
    assert parse_recon_log(data.decode("utf-8")).invocations == ()  # what the mark would cost


def test_carriage_returns_end_lines_like_line_feeds() -> None:
    unix = log("-all -T2pial", body=truncation_line(5, 6) + holes_line(7, 8))
    assert parse_recon_log(unix.replace("\n", "\r\n")) == parse_recon_log(unix)
    assert parse_recon_log(unix.replace("\n", "\r")) == parse_recon_log(unix)
    # progress output as in the real logs: many "lines" between two line feeds
    progress = "\rstep 000: RMS=0.158 (target=0.015)   \rstep 005: RMS=0.119 (target=0.015)   "
    text = fs6_header("-all") + progress + "\rsubjid again\r" + truncation_line(1, 2) + FINISHED
    parsed = parse_recon_log(text)
    assert len(parsed.invocations) == 2
    assert parsed.truncations == ((1, 2),)
    assert parse_build_stamp(FS6_STAMP + "\r\n").raw == FS6_STAMP


# ---------------------------------------------------------------------------------------
# participant, longitudinal


@pytest.mark.parametrize(
    ("label", "participant", "longitudinal"),
    [
        ("sub-01", "sub-01", False),
        ("sub-UTS01", "sub-UTS01", False),
        ("sub-01_ses-2", "sub-01", False),
        ("sub-01_ses-2.long.sub-01", "sub-01", True),
        ("sub-01.long.sub-01_base", "sub-01", True),
        ("sub-a1-b", "sub-a1", False),
        ("bert", "bert", False),
        ("bert.long.bert_base", "bert.long.bert_base", True),
        ("Sub-01", "Sub-01", False),
        ("xsub-01_ses-1", "xsub-01_ses-1", False),
        ("sub-", "sub-", False),
        ("sub-01.long", "sub-01", False),
        ("long.sub-01", "long.sub-01", False),
    ],
)
def test_participant_and_longitudinal_from_the_directory_name(
    label: str, participant: str, longitudinal: bool
) -> None:
    found = subject_provenance({}, {}, label)
    assert (found.participant, found.longitudinal) == (participant, longitudinal)


# ---------------------------------------------------------------------------------------
# properties

TOKEN_PARTS = ("-", "/", "ds000102", "ds123456", "sub-01", "T2pial", "no", "a", "_", "7", "3T", ".")
tokens = st.lists(st.sampled_from(TOKEN_PARTS), min_size=1, max_size=5).map("".join)
any_text = st.text(max_size=400)
log_like = st.lists(
    st.one_of(
        st.sampled_from(
            [
                "subjid sub-01",
                "subjid",
                "",
                "\r",
                " ",
                "-all -T2pial /data/ds000102-fmriprep/x -ds000102",
                "/opt/freesurfer/bin/recon-all -i /a/b -s sub-01",
                "build-stamp.txt: " + FS7_STAMP,
                "build-stamp.txt:",
                "thickness calculation complete, 39:225 truncations.",
                "thickness calculation complete, 9999999999:1 truncations.",
                "orig.nofix lhholes =   41, rhholes = 26",
                "recon-all -s x finished without error at",
                "recon-all -s x exited with ERRORS at",
            ]
        ),
        st.text(max_size=40),
    ),
    max_size=30,
).map("\n".join)


def check_log(parsed: ReconLog) -> None:
    for invocation in parsed.invocations:
        for flag in invocation:
            assert "/" not in flag and not ACCESSION.search(flag), flag
            assert FLAG.fullmatch(flag) or flag in ("-3T", "-3t"), flag
    assert all(a >= 0 and b >= 0 for a, b in parsed.truncations)


@given(text=st.one_of(any_text, log_like), partial=st.booleans())
def test_any_text_parses_without_raising(text: str, partial: bool) -> None:
    parsed = parse_recon_log(text, partial=partial)
    check_log(parsed)
    if partial:
        assert parsed.finished is None
    stamp = parse_build_stamp(text)
    assert (stamp.version is None) == (stamp.major is None)
    assert stamp.version is None or stamp.version in stamp.raw

    files = {MAIN: log_ref(MAIN), **pial_ref(text[:20], text[20:40] or None)}
    found = subject_provenance({STAMP: text, MAIN: text, LH: text, RH: text}, files, text[:30])
    assert found.flags == tuple(sorted(set(found.flags)))
    assert found.n_invocations == 3 * len(parse_recon_log(text).invocations)
    assert found.pial_variant in (None, "pial", "pial.T1", "pial.T2", "pial.FLAIR")
    assert set(found.truncations) == {"lh", "rh"}


@given(data=st.binary(max_size=300))
def test_any_bytes_decode_and_parse(data: bytes) -> None:
    text = decode_text(data)
    check_log(parse_recon_log(text))
    parse_build_stamp(text)


@given(arguments=st.lists(tokens, max_size=8), layout=st.sampled_from(["fs6", "fs7"]))
def test_no_token_with_a_slash_or_an_accession_survives_in_the_flags(
    arguments: list[str], layout: str
) -> None:
    parsed = parse_recon_log(log("-first " + " ".join(arguments), layout=layout))
    (flags,) = parsed.invocations
    check_log(parsed)
    # and the filter drops nothing else: what is flag-shaped and clean is kept, in order
    assert flags == (
        "-first",
        *(
            token
            for token in arguments
            if (FLAG.fullmatch(token) or token in ("-3T", "-3t")) and not ACCESSION.search(token)
        ),
    )
    found = subject_provenance({MAIN: log(" ".join(arguments), layout=layout)}, {}, "sub-01")
    assert all("/" not in flag and not ACCESSION.search(flag) for flag in found.flags)


# ---------------------------------------------------------------------------------------
# the whole logs (network)

LOG_NAMES = ("recon-all.log", "recon-all-lh.log", "recon-all-rh.log")


@pytest.fixture(scope="session")
def real_logs(real_subject: Path, real_subject_fs7: Path) -> dict[str, dict[str, bytes]]:
    """The six logs the excerpts are cut from: those of the two real subjects that
    ``conftest.py`` keeps in the pytest cache, with the sizes of the bucket listing."""
    subjects = {"fs6": real_subject, "fs7": real_subject_fs7}
    return {
        prefix: {name: (subject / "scripts" / name).read_bytes() for name in LOG_NAMES}
        for prefix, subject in subjects.items()
    }


@pytest.mark.network
@pytest.mark.parametrize("prefix", ["fs6", "fs7"])
def test_the_excerpts_are_what_the_script_cuts_and_parse_like_the_whole_logs(
    real_logs: dict[str, dict[str, bytes]], prefix: str
) -> None:
    spec = importlib.util.spec_from_file_location("make_excerpts", DATA / "make_excerpts.py")
    assert spec is not None and spec.loader is not None
    make_excerpts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_excerpts)
    for name, data in real_logs[prefix].items():
        text = decode_text(data)
        committed = (DATA / f"{prefix}_{name}").read_text(encoding="utf-8")
        assert make_excerpts.excerpt(text) == committed, name
        assert parse_recon_log(committed) == parse_recon_log(text), name


@pytest.mark.network
def test_provenance_of_the_two_real_subjects_from_their_whole_logs(
    real_logs: dict[str, dict[str, bytes]],
) -> None:
    roles = {"recon-all.log": MAIN, "recon-all-lh.log": LH, "recon-all-rh.log": RH}
    for prefix, n_main, n_all, truncations, holes in (
        ("fs6", 4, 8, {"lh": (39, 225), "rh": (93, 434)}, (41, 26)),
        ("fs7", 5, 9, {"lh": (22, 230), "rh": (23, 325)}, (4, 15)),
    ):
        texts = {roles[name]: decode_text(data) for name, data in real_logs[prefix].items()}
        assert len(parse_recon_log(texts[MAIN]).invocations) == n_main
        found = subject_provenance(texts, {}, "sub-01")
        assert found == subject_provenance(excerpts(prefix), {}, "sub-01")
        assert (found.n_invocations, found.truncations, found.holes) == (n_all, truncations, holes)
        assert (found.finished, found.t2pial, found.flairpial, found.edited) == (
            True,
            False,
            False,
            False,
        )
        assert all("/" not in flag and not ACCESSION.search(flag) for flag in found.flags)

        # the first 64 kB, what Tier A fetched before decision G1: two invocations, no verdict
        head = decode_text(real_logs[prefix]["recon-all.log"][:65536])
        assert len(parse_recon_log(head, partial=True).invocations) == 2
        assert parse_recon_log(head, partial=True).finished is None
