import unittest
from types import SimpleNamespace

from analysis.entry_points import collect_unique_entry_points


class _FakeContract:
    def __init__(self, name, entry_points):
        self.name = name
        self.entry_points = list(entry_points)


def _fake_entry_point(declarer_name, signature, *, filename, start, length):
    declarer = SimpleNamespace(name=declarer_name)
    source_mapping = SimpleNamespace(
        filename=SimpleNamespace(absolute=filename, relative=filename, short=filename),
        start=start,
        length=length,
    )
    return SimpleNamespace(
        contract_declarer=declarer,
        contract=declarer,
        solidity_signature=signature,
        full_name=signature,
        name=signature.split("(", 1)[0],
        source_mapping=source_mapping,
    )


class CollectUniqueEntryPointsTests(unittest.TestCase):
    def test_dedupes_inherited_entry_points_seen_on_base_and_root(self):
        approve = _fake_entry_point(
            "ERC20",
            "approve(address,uint256)",
            filename="IpToken.sol",
            start=100,
            length=25,
        )
        base = _FakeContract("ERC20", [approve])
        root = _FakeContract("IpToken", [approve])

        result = collect_unique_entry_points(
            [base, root],
            lambda contract: contract.entry_points,
        )

        self.assertEqual([entry for _, entries in result for entry in entries], [approve])
        self.assertEqual(result[0][1], [approve])
        self.assertEqual(result[1][1], [])

    def test_preserves_distinct_overloads(self):
        approve_no_data = _fake_entry_point(
            "ERC20",
            "approve(address,uint256)",
            filename="IpToken.sol",
            start=100,
            length=25,
        )
        approve_with_data = _fake_entry_point(
            "ERC20",
            "approve(address,uint256,bytes)",
            filename="IpToken.sol",
            start=140,
            length=32,
        )
        contract = _FakeContract("ERC20", [approve_no_data, approve_with_data])

        result = collect_unique_entry_points(
            [contract],
            lambda current: current.entry_points,
        )

        self.assertEqual(result[0][1], [approve_no_data, approve_with_data])


if __name__ == "__main__":
    unittest.main()
