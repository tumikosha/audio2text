from audio2text import markdown as md


def test_markdown_paragraphs():
    doc = {"source": "https://youtu.be/x", "title": "T: a/b", "transcript": "youtube auto (ru-orig)",
           "audio_s": 3600, "speakers": {},
           "segments": [{"s": 1, "e": 2, "text": "Здравствуйте.", "turn": True},
                        {"s": 3, "e": 4, "text": "Добрый день.", "turn": True},
                        {"s": 5, "e": 6, "text": "Итак", "turn": False}]}
    assert md.to_markdown(doc) == ("# T: a/b\n\n- Источник: https://youtu.be/x\n- Транскрипт: youtube auto (ru-orig)\n"
                                   "- Длительность: 1:00:00\n\n— `00:00:01` Здравствуйте.\n\n"
                                   "— `00:00:03` Добрый день. Итак\n")


def test_markdown_speakers_and_long_monologue():
    segs = [{"s": float(t), "e": t + 9.5, "text": f"Фраза {t}.", "speaker": "SPEAKER_00"} for t in range(0, 130, 10)]
    segs[5]["text"] = "Без точки"  # at 60 s the paragraph does not end in a sentence end: it goes on to 70 s
    segs.append({"s": 131.0, "e": 133.0, "text": "Да.", "speaker": "SPEAKER_01"})
    doc = {"source": "/tmp/встреча 1.m4a", "transcript": "локально, parakeet-tdt-0.6b-v3", "audio_s": 134.4,
           "speakers": {}, "segments": segs}
    text = md.to_markdown(doc)
    head, body = text.split("\n\n", 1)[0], text.split("\n\n", 2)[2]
    assert head == "# встреча 1"
    assert "- Транскрипт: локально, parakeet-tdt-0.6b-v3\n" in text
    assert "- Длительность: 0:02:14\n" in text
    paras = body.rstrip("\n").split("\n\n")
    assert [p.split(" ", 2)[:2] for p in paras] == [["**SPEAKER_00**", "`00:00:00`"],
                                                     ["**SPEAKER_00**", "`00:01:10`"],
                                                     ["**SPEAKER_01**", "`00:02:11`"]]
    assert paras[0].endswith("Фраза 40. Без точки Фраза 60.")
    assert paras[2].endswith("Да.")


def test_markdown_paragraph_cut_at_max_without_sentence_end():
    segs = [{"s": float(t), "e": t + 9.5, "text": f"слово {t}", "speaker": "SPEAKER_00"} for t in range(0, 140, 10)]
    paras = md.to_markdown({"source": "a.wav", "speakers": {}, "segments": segs}).rstrip().split("\n\n")[2:]
    assert [p.split(" ")[1] for p in paras] == ["`00:00:00`", "`00:02:00`"]


def test_hms_and_clock():
    assert md._hms(3599.6) == "1:00:00"
    assert md._hms(None) == "?"
    assert md._clock(59.9) == "00:00:59"


def test_write_atomic_utf8(tmp_path):
    out = tmp_path / "Записи" / "встреча 1.md"
    md.write({"source": "x.wav", "title": "Ёлка", "speakers": {}, "segments": []}, out)
    assert out.read_bytes().decode("utf-8").startswith("# Ёлка\n")
    assert [p.name for p in out.parent.iterdir()] == ["встреча 1.md"]
