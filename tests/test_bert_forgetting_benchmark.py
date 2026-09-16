import json

import numpy as np
import pytest

from examples.bert_forgetting_benchmark import (
    ClassificationBatcher,
    DatasetSpec,
    build_classifier_graph,
    encode_with_vocab,
    ensure_output_dir,
    make_evaluation_records,
    parse_args,
    run_sequential_benchmark,
    select_stratified_examples,
    validate_task_names,
    validate_sample_limit,
    write_benchmark_outputs,
)

VOCAB = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "a": 4, "b": 5, "c": 6}


def test_encode_with_vocab_reserves_sep_truncates_pads_and_masks():
    encoded = encode_with_vocab("a b c", VOCAB, seq_len=4)

    assert encoded["input_ids"] == [2, 4, 5, 3]
    assert encoded["attention_mask"] == [1, 1, 1, 1]

    padded = encode_with_vocab("a", VOCAB, seq_len=5)
    assert padded["input_ids"] == [2, 4, 3, 0, 0]
    assert padded["attention_mask"] == [1, 1, 1, 0, 0]


def test_encode_with_vocab_validates_special_tokens_and_seq_len():
    with pytest.raises(ValueError, match="seq_len"):
        encode_with_vocab("a", VOCAB, seq_len=1)
    with pytest.raises(ValueError, match=r"\[SEP\]"):
        encode_with_vocab("a", {"[PAD]": 0, "[CLS]": 1}, seq_len=4)


def test_classification_batcher_shapes_shuffle_and_label_validation():
    examples = [("a", 1), ("b", 0), ("c", 1)]
    first = list(
        ClassificationBatcher(
            examples, VOCAB, seq_len=4, batch_size=2, seed=7, shuffle=True
        )
    )
    second = list(
        ClassificationBatcher(
            examples, VOCAB, seq_len=4, batch_size=2, seed=7, shuffle=True
        )
    )

    assert first[0]["x"].shape == (2, 4)
    assert first[0]["attention_mask"].shape == (2, 4)
    assert first[0]["y"].shape == (2,)
    assert np.array_equal(first[0]["y"], second[0]["y"])

    with pytest.raises(ValueError, match="label"):
        list(ClassificationBatcher([("bad", 2)], VOCAB, seq_len=4, batch_size=1))


def test_validate_task_names_rejects_duplicates_and_unsupported():
    specs = {"sst2": DatasetSpec("sst2", "nyu-mll/glue", "sst2", "sentence", "label")}

    assert validate_task_names(["sst2"], specs) == ["sst2"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_task_names(["sst2", "sst2"], specs)
    with pytest.raises(ValueError, match="Unsupported"):
        validate_task_names(["imdb"], specs)


def test_ensure_output_dir_protects_existing_outputs(tmp_path):
    output_dir = tmp_path / "out"
    ensure_output_dir(output_dir, overwrite=False)
    (output_dir / "partial_stage_checkpoint.json").write_text("{}")

    with pytest.raises(FileExistsError, match="overwrite"):
        ensure_output_dir(output_dir, overwrite=False)
    ensure_output_dir(output_dir, overwrite=True)


def test_output_writes_reject_symlinks(tmp_path):
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    linked_dir = tmp_path / "linked"
    linked_dir.symlink_to(target_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="symlinks"):
        ensure_output_dir(linked_dir, overwrite=True)

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    outside = tmp_path / "outside.json"
    (output_dir / "summary.json").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        write_benchmark_outputs(
            output_dir,
            ["sst2"],
            [[0.1], [0.2]],
            make_evaluation_records(["sst2"], [[0.1], [0.2]]),
            {"tasks": ["sst2"], "ACC": 0.2},
            {},
        )


def test_validate_sample_limit_rejects_non_positive_values():
    validate_sample_limit(None, "max_eval_samples")
    validate_sample_limit(1, "max_eval_samples")

    with pytest.raises(ValueError, match="positive"):
        validate_sample_limit(0, "max_eval_samples")
    with pytest.raises(ValueError, match="positive"):
        validate_sample_limit(-1, "max_train_samples")


def test_select_stratified_examples_balances_ordered_binary_data():
    examples = [(f"negative {index}", 0) for index in range(10)] + [
        (f"positive {index}", 1) for index in range(10)
    ]

    first = select_stratified_examples(examples, max_samples=6, seed=3)
    second = select_stratified_examples(examples, max_samples=6, seed=3)
    labels = [label for _, label in first]

    assert first == second
    assert len(first) == 6
    assert labels.count(0) == 3
    assert labels.count(1) == 3
    assert first != examples[:6]


def test_write_benchmark_outputs_json_and_jsonl(tmp_path):
    matrix = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    records = make_evaluation_records(
        ["sst2", "imdb"], matrix, losses=[[1, 2], [3, 4], [5, 6]]
    )
    summary = {
        "tasks": ["sst2", "imdb"],
        "ACC": 0.55,
        "BWT": 0.2,
        "FWT": 0.2,
        "per_task": {},
    }

    write_benchmark_outputs(
        tmp_path,
        ["sst2", "imdb"],
        matrix,
        records,
        summary,
        {"seed": 1},
        loss_matrix=[[1, 2], [3, 4], [5, 6]],
    )

    assert json.loads((tmp_path / "summary.json").read_text())["ACC"] == 0.55
    matrix_json = json.loads((tmp_path / "accuracy_matrix.json").read_text())
    assert matrix_json["accuracy_matrix"] == matrix
    assert matrix_json["loss_matrix"] == [[1, 2], [3, 4], [5, 6]]
    lines = (tmp_path / "evaluations.jsonl").read_text().strip().splitlines()
    assert len(lines) == 6
    assert json.loads(lines[0])["after_task"] is None
    assert json.loads(lines[-1])["eval_task"] == "imdb"
    assert (
        json.loads((tmp_path / "experiment_record.json").read_text())["config"]["seed"]
        == 1
    )


def test_build_classifier_graph_uses_attention_mask():
    structure = build_classifier_graph(
        depth=1,
        embed_dim=8,
        num_heads=2,
        mlp_dim=16,
        seq_len=4,
        vocab_size=len(VOCAB),
        num_classes=2,
    )

    assert "attention_mask" in structure.task_map


def test_tiny_offline_sequential_benchmark_flow_writes_outputs(tmp_path):
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{}")
    calls = []

    def fake_loader(task_name, split, batch_size, seq_len, seed, max_samples, vocab):
        return [task_name, split]

    def fake_checkpoint_loader(path):
        return {
            "params": {"encoder": "pretrained"},
            "vocab": VOCAB,
            "config": {"seq_len": 4},
        }

    def fake_model_factory(**kwargs):
        return {"model_kwargs": kwargs}

    def fake_train_fn(params, task_name, train_loader, stage_index, optimizer):
        calls.append(("train", task_name, stage_index, optimizer["reset_index"]))
        return {**params, "trained": params.get("trained", []) + [task_name]}

    def fake_evaluate_fn(params, task_name, eval_loader, stage_index):
        calls.append(("eval", task_name, stage_index))
        task_index = ["sst2", "imdb", "yelp_polarity"].index(task_name)
        return {
            "accuracy": stage_index / 10 + task_index / 100,
            "loss": 10 + stage_index + task_index,
        }

    result = run_sequential_benchmark(
        checkpoint=checkpoint,
        tasks=["sst2", "imdb", "yelp_polarity"],
        output_dir=tmp_path / "outputs",
        epochs_per_task=1,
        batch_size=2,
        lr=0.01,
        weight_decay=0.0,
        warmup_steps=0,
        grad_clip=1.0,
        max_train_samples=3,
        max_eval_samples=2,
        seq_len=4,
        seed=123,
        overwrite=False,
        checkpoint_loader=fake_checkpoint_loader,
        loader_factory=fake_loader,
        model_factory=fake_model_factory,
        train_fn=fake_train_fn,
        evaluate_fn=fake_evaluate_fn,
        optimizer_factory=lambda **kwargs: {
            **kwargs,
            "reset_index": len([c for c in calls if c[0] == "train"]),
        },
    )

    assert np.asarray(result["accuracy_matrix"]).shape == (4, 3)
    assert [call for call in calls if call[0] == "train"] == [
        ("train", "sst2", 1, 0),
        ("train", "imdb", 2, 1),
        ("train", "yelp_polarity", 3, 2),
    ]
    assert len([call for call in calls if call[0] == "eval"]) == 12
    assert (tmp_path / "outputs" / "summary.json").exists()
    assert (
        json.loads((tmp_path / "outputs" / "experiment_record.json").read_text())[
            "stages"
        ][0]["stage"]
        == 0
    )


def test_missing_checkpoint_fails_before_output(tmp_path):
    output_dir = tmp_path / "outputs"

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        run_sequential_benchmark(
            checkpoint=tmp_path / "missing.json",
            tasks=["sst2"],
            output_dir=output_dir,
            epochs_per_task=1,
            batch_size=2,
            lr=0.01,
            weight_decay=0.0,
            warmup_steps=0,
            grad_clip=1.0,
            max_train_samples=1,
            max_eval_samples=1,
            seq_len=4,
            seed=1,
            overwrite=False,
        )
    assert not output_dir.exists()


def test_stage_checkpoint_rejects_symlink_target(tmp_path):
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{}")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "stage_1_sst2.json").symlink_to(tmp_path / "outside.json")

    with pytest.raises(ValueError, match="symlink"):
        run_sequential_benchmark(
            checkpoint=checkpoint,
            tasks=["sst2"],
            output_dir=output_dir,
            epochs_per_task=1,
            batch_size=2,
            lr=0.01,
            weight_decay=0.0,
            warmup_steps=0,
            grad_clip=1.0,
            max_train_samples=1,
            max_eval_samples=1,
            seq_len=4,
            seed=1,
            overwrite=True,
            save_stage_checkpoints=True,
            checkpoint_loader=lambda _path: {
                "params": {"fake": True},
                "vocab": VOCAB,
                "config": {"seq_len": 4},
            },
            loader_factory=lambda **_kwargs: [],
            model_factory=lambda **kwargs: {"model_kwargs": kwargs},
            train_fn=lambda params, *_args: params,
            evaluate_fn=lambda *_args: {"accuracy": 0.5, "loss": 1.0},
        )


def test_parse_tasks_accepts_space_separated_and_comma_separated_values():
    args = parse_args(
        [
            "--checkpoint",
            "checkpoint.json",
            "--output_dir",
            "out",
            "--tasks",
            "sst2",
            "imdb,yelp_polarity",
        ]
    )

    assert args.tasks == ["sst2", "imdb,yelp_polarity"]
