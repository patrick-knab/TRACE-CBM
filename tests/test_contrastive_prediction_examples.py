from paper_expos.contrastive_prediction_examples import select_examples


def test_select_examples_prioritizes_reversals_and_distinct_datasets() -> None:
    rows = [
        {"dataset": "barista", "case_index": 1, "runner_up_reversal": 0, "margin_drop": 9.0},
        {"dataset": "barista", "case_index": 2, "runner_up_reversal": 1, "margin_drop": 2.0},
        {"dataset": "breakfast_s1", "case_index": 3, "runner_up_reversal": 1, "margin_drop": 1.0},
        {"dataset": "mpii_attr", "case_index": 4, "runner_up_reversal": 0, "margin_drop": 8.0},
    ]

    selected = select_examples(rows, 2)

    assert [row["dataset"] for row in selected] == ["barista", "breakfast_s1"]
    assert all(int(row["runner_up_reversal"]) == 1 for row in selected)


def test_select_examples_preserves_explicit_semantic_selection() -> None:
    rows = [
        {"dataset": "breakfast_s1", "case_index": 48, "runner_up_reversal": 1, "margin_drop": 1.3},
        {"dataset": "mpii_attr", "case_index": 47, "runner_up_reversal": 1, "margin_drop": 2.0},
    ]

    selected = select_examples(rows, 2, (("mpii_attr", 47), ("breakfast_s1", 48)))

    assert [(row["dataset"], row["case_index"]) for row in selected] == [
        ("mpii_attr", 47),
        ("breakfast_s1", 48),
    ]
