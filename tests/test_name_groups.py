"""name_groups: grouping Eedi names judged to be the same mistake. No data files needed."""

from matheasy.shortlister import name_groups
from matheasy.shortlister.name_groups import join, overlap, unchecked


def test_overlap_is_the_share_of_shared_words():
    assert overlap("Adds instead of subtracts", "Subtracts instead of adds") == 1.0
    assert overlap("Rounds up", "Rounds down") == 1 / 3
    assert overlap("", "") == 0.0


def test_pairs_judged_the_same_join_into_groups_keyed_by_the_smallest_id():
    groups = join([(1749, 1551), (2030, 1667), (1667, 9)])
    assert groups == {1749: 1551, 1551: 1551, 2030: 9, 1667: 9, 9: 9}


def test_a_chain_reports_the_members_never_judged_against_each_other():
    same = [(1, 2), (2, 3)]
    assert unchecked(join(same), same) == {1: [(1, 3)]}
    assert unchecked(join([(1, 2)]), [(1, 2)]) == {}


def test_without_the_group_file_no_name_is_grouped(tmp_path, monkeypatch):
    monkeypatch.setattr(name_groups, "GROUPS", tmp_path / "missing.json")
    name_groups.group_of.cache_clear()
    assert name_groups.group_of() == {}
    name_groups.group_of.cache_clear()
