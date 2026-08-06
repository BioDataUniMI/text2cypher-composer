import json

import pandas as pd

from text2cypher_composer import (
    build_gpt_finetune_jsonl,
    build_prompt_completion_pairs,
    load_finetune_levels,
    split_finetune_dataset,
    write_local_finetune_dataset,
)


def test_build_prompt_completion_pairs():
    df = pd.DataFrame([{"question": "How many genes?", "cypher": "MATCH (g:Gene) RETURN count(g)"}])
    pairs = build_prompt_completion_pairs(df)
    assert pairs == [
        {"prompt": "How many genes?\nCypher query:", "completion": " MATCH (g:Gene) RETURN count(g)"}
    ]


def test_load_finetune_levels_tags_id_and_level(tmp_path):
    level_a = tmp_path / "nodeLevel.json"
    level_b = tmp_path / "1hop.json"
    level_a.write_text(json.dumps([{"question": "q1", "cypher": "c1"}, {"question": "q2", "cypher": "c2"}]))
    level_b.write_text(json.dumps([{"question": "q3", "cypher": "c3"}]))

    df = load_finetune_levels({"nodeLevel": str(level_a), "1hop": str(level_b)})

    assert list(df["ID"]) == ["nodeLevel/question_1", "nodeLevel/question_2", "1hop/question_1"]
    assert list(df["level"]) == ["nodeLevel", "nodeLevel", "1hop"]


def test_split_finetune_dataset_is_disjoint_and_covers_everything():
    df = pd.DataFrame(
        [{"question": f"q{i}", "cypher": f"c{i}", "level": "a" if i < 5 else "b"} for i in range(10)]
    )
    train_df, test_df = split_finetune_dataset(df, test_frac=0.4, random_state=42)
    assert len(train_df) + len(test_df) == len(df)
    assert set(train_df["question"]).isdisjoint(set(test_df["question"]))


def test_write_local_finetune_dataset(tmp_path):
    df = pd.DataFrame([{"question": "q1", "cypher": "c1", "level": "a"}])
    path = write_local_finetune_dataset(df, tmp_path / "out.json")
    loaded = json.loads(path.read_text())
    assert loaded == [{"question": "q1", "cypher": "c1"}]


def test_build_gpt_finetune_jsonl(tmp_path):
    df = pd.DataFrame([{"question": "q1", "cypher": "c1"}, {"question": "q2", "cypher": "c2"}])
    result = build_gpt_finetune_jsonl(df, tmp_path / "out.jsonl")
    assert result.n_examples == 2

    lines = [json.loads(line) for line in result.path.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["messages"][0]["role"] == "system"
    assert lines[0]["messages"][1] == {"role": "user", "content": "q1"}
    assert lines[0]["messages"][2] == {"role": "assistant", "content": "c1"}


def test_build_gpt_finetune_jsonl_without_system_message(tmp_path):
    df = pd.DataFrame([{"question": "q1", "cypher": "c1"}])
    result = build_gpt_finetune_jsonl(df, tmp_path / "out.jsonl", add_system=False)
    line = json.loads(result.path.read_text())
    assert [m["role"] for m in line["messages"]] == ["user", "assistant"]
