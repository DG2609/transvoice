import threading
import time

import numpy as np

from transvoice.engine import (Engine, Settings, _ChannelVad, ends_sentence, is_noise, open_ended,
                               split_sentences)
from transvoice.mt import Translation, TranslationCancelled


def test_is_noise_drops_fillers():
    assert is_noise("The.", "en", 0.5)
    assert is_noise("The.", "en", 1.8)  # a lone filler is dropped whatever the segment length
    assert is_noise("TÔI", "vi", 2.0)
    assert is_noise("", "vi", 3.0)
    assert not is_noise("The meeting starts now.", "en", 0.7)
    assert not is_noise("はい", "ja", 1.5)


def test_ends_sentence():
    assert ends_sentence("参加できないと思います", "ja")
    assert ends_sentence("そうですか", "ja")
    assert not ends_sentence("明日の会議には", "ja")
    assert ends_sentence("That means lots of homework.", "en")
    assert not ends_sentence("and for many students,", "en")
    assert not ends_sentence("hôm nay trời đẹp", "vi")


def test_split_sentences():
    assert split_sentences("どんなところだと思いますか？やっぱ黒髪が", "ja") == ["どんなところだと思いますか？", "やっぱ黒髪が"]
    assert split_sentences("きょうですね、日本について", "ja") == ["きょうですね、日本について"]
    assert split_sentences("you can get a ballot. Friends at the U.S. center", "en") == \
        ["you can get a ballot.", "Friends at the U.S. center"]
    assert split_sentences("It costs 3.5 dollars", "en") == ["It costs 3.5 dollars"]
    # Japanese ASR output without "。": split at polite endings followed by a new sentence
    assert split_sentences("何にでも使えるんじゃないですか例えば残さないで食べるのも思いやりの一つですよね人の気遣い", "ja") == \
        ["何にでも使えるんじゃないですか", "例えば残さないで食べるのも思いやりの一つですよね", "人の気遣い"]
    assert split_sentences("昨日行きましたが楽しかった", "ja") == ["昨日行きましたが楽しかった"]  # が continues
    assert split_sentences("そうですね、おみそ汁とか", "ja") == ["そうですね、おみそ汁とか"]  # filler, not an ending


def test_sentence_end_at_a_chunk_cut_starts_a_new_sentence():
    chunks = ["どんなところがいいところだと思いますか", "やっぱ黒髪が似合う女性が", "いっぱいいるところです"]
    e, events, finals = _run(chunks, forced=[True, True, False], gap=0.1, wait_finals=2)
    assert [f.text for f in finals] == ["どんなところがいいところだと思いますか", "やっぱ黒髪が似合う女性がいっぱいいるところです"]


def test_continuation_after_ending_stays_in_the_sentence():
    chunks = ["昨日は会議がありました", "が、すぐ終わりました"]
    e, events, finals = _run(chunks, forced=[True, False], gap=0.1)
    assert finals[0].text == "昨日は会議がありましたが、すぐ終わりました"


def test_false_period_at_forced_english_cut_is_ignored():
    chunks = ["On August.", "15, 1940, the allies invaded.", "Next sentence starts here."]
    _, _, finals = _run(chunks, forced=[True, False, False], lang="en", gap=0.1, wait_finals=2,
                        rescore_final=False)  # checks the chunk-level joining itself
    assert finals[0].text == "On August 15, 1940, the allies invaded."
    assert finals[1].text == "Next sentence starts here."


def test_chunk_spanning_two_sentences_is_split():
    chunks = ["日本のいいところは何だと思いますか。やっぱり", "食べ物がおいしいです"]
    e, events, finals = _run(chunks, forced=[True, False], gap=0.1, wait_finals=2)
    assert [f.text for f in finals] == ["日本のいいところは何だと思いますか。", "やっぱり食べ物がおいしいです"]


def _engine(**kw):
    return Engine(Settings(**kw), on_event=lambda *a: None)


class FakeLid:
    def __init__(self, answers):
        self.answers = list(answers)

    def detect(self, samples):
        return self.answers.pop(0)


def test_language_routing_with_auto_detection():
    e = _engine(my_lang="vi", their_lang="auto")
    e.lid = FakeLid(["ja", "ko", "en"])
    long_clip = [0.0] * 32000  # 2 s: enough audio to trust language ID
    assert e._language("them", long_clip) == ("ja", True)
    assert e._language("them", long_clip) == ("ja", False)  # "ko" is not allowed -> unconfirmed guess
    assert e._language("them", [0.0] * 8000) == ("ja", False)  # short clip -> no LID call, reuse last
    assert e._language("them", long_clip) == ("en", True)
    assert e._language("me", long_clip) == ("vi", True)


def test_short_first_chunk_in_wrong_language_is_relabeled():
    # The speaker switches from Vietnamese to Japanese; the first 0.8 s chunk is guessed as Vietnamese.
    events, done = [], threading.Event()

    def on_event(kind, payload):
        events.append((kind, payload))
        if kind == "translated" and payload.final:
            done.set()

    e = Engine(Settings(my_lang="en", their_lang="auto", sentence_gap_s=0.3), on_event)
    e.their_last = "vi"
    e.lid = FakeLid(["ja"])
    e.asr = {"vi": FakeAsr(["Quay được", "gì đó"]), "ja": FakeAsr(["ワイルドカードを", "購入するとお得です"])}
    e.mt = FakeMt()
    e._start_workers()
    e._enqueue("them", np.full(12800, 0, np.float32), time.time(), True)
    time.sleep(0.2)
    e._enqueue("them", np.full(24000, 1, np.float32), time.time(), False)
    assert done.wait(5)
    e.stop()
    final = [p for k, p in events if k == "translated" and p.final][0]
    assert final.lang == "ja" and final.text == "ワイルドカードを購入するとお得です"
    assert e.mt.calls[-1] == "ワイルドカードを購入するとお得です"


def test_targets():
    e = _engine(my_lang="vi", their_lang="ja")
    assert e._target("them") == "vi"
    assert e._target("me") == "ja"


# ---- chunk cutter ----

class FakeVad:
    """Always hears speech; records where the cutter closes segments."""

    def __init__(self):
        self.pos, self.cuts = 0, []

    def accept_waveform(self, w):
        self.pos += len(w)

    def is_speech_detected(self):
        return True

    def flush(self):
        self.cuts.append(self.pos / 16000)

    def empty(self):
        return True


def _speech_with_pause(total_s, pause_at_s):
    rng = np.random.default_rng(0)
    x = 0.3 * rng.standard_normal(int(total_s * 16000)).astype(np.float32)
    x[int(pause_at_s * 16000) : int((pause_at_s + 0.1) * 16000)] *= 0.01  # a short gap between words
    return x


def test_long_speech_is_cut_in_a_pause_after_soft_limit():
    fake = FakeVad()
    _ChannelVad(fake, soft_s=2.0, hard_s=4.0).accept(_speech_with_pause(4.5, pause_at_s=2.6))
    assert len(fake.cuts) == 1 and 2.6 <= fake.cuts[0] <= 2.75


def test_single_quiet_window_does_not_cut():
    fake = FakeVad()
    x = _speech_with_pause(4.5, pause_at_s=3.0)
    x[int(2.6 * 16000) : int(2.6 * 16000) + 400] *= 0.01  # 25 ms dip: inside a word, not a pause
    _ChannelVad(fake, soft_s=2.0, hard_s=4.0).accept(x)
    assert len(fake.cuts) == 1 and fake.cuts[0] >= 3.0


def test_long_speech_without_pause_is_cut_at_hard_limit():
    fake = FakeVad()
    _ChannelVad(fake, soft_s=2.0, hard_s=4.0).accept(_speech_with_pause(9.0, pause_at_s=1.0))
    assert [round(c) for c in fake.cuts] == [4, 8]


# ---- sentences, drafts and final translations (fake ASR/MT, real threads) ----

class FakeAsr:
    """Each fake chunk is a constant array holding its text index. Audio joined from several chunks (the
    whole-sentence pass) returns `whole` if given, else the chunk texts joined."""

    def __init__(self, texts, delay=0.0, whole=None, joiner=""):
        self.texts, self.delay, self.whole, self.joiner = texts, delay, whole, joiner

    def transcribe(self, samples):
        time.sleep(self.delay)
        runs = [int(v) for i, v in enumerate(samples) if i == 0 or v != samples[i - 1]]
        if len(runs) > 1 and self.whole is not None:
            return self.whole
        return self.joiner.join(self.texts[r] for r in runs)


class FakeMt:
    def __init__(self, delay=0.0):
        self.delay, self.calls, self.cancelled = delay, [], []
        self.pids = []

    def translate(self, text, src, tgt, cancel=None, on_partial=None):
        self.calls.append(text)
        full = f"<{text}>"
        deadline = time.time() + self.delay
        while time.time() < deadline:
            if cancel is not None and cancel.is_set():
                self.cancelled.append(text)
                raise TranslationCancelled()
            if on_partial is not None:  # stream the translation in growing prefixes
                done = 1 - (deadline - time.time()) / self.delay
                on_partial(full[: max(1, int(len(full) * done))])
            time.sleep(0.01)
        return Translation(full, self.delay)

    def close(self):
        pass


def _run(texts, forced, lang="ja", my_lang="vi", mt_delay=0.0, gap=0.0, wait_finals=1, asr_delay=0.0,
         ended_ago=0.0, **settings):
    events, done = [], threading.Event()

    def on_event(kind, payload):
        events.append((kind, payload))
        if sum(k == "translated" and p.final for k, p in events) >= wait_finals:
            done.set()

    e = Engine(Settings(my_lang=my_lang, their_lang=lang, sentence_gap_s=0.3, **settings), on_event)
    e.asr = {lang: FakeAsr(texts, asr_delay, joiner="" if lang == "ja" else " ")}
    e.mt = FakeMt(mt_delay)
    e._start_workers()
    for i, f in enumerate(forced):
        e._enqueue("them", np.full(24000, i, np.float32), time.time() - ended_ago, f)
        time.sleep(gap)
    assert done.wait(5), "no final translation"
    e.stop()
    finals = [p for k, p in events if k == "translated" and p.final]
    return e, events, finals


def test_final_translation_covers_the_whole_sentence():
    chunks = ["明日の会議には", "参加できないと", "思います"]
    e, events, finals = _run(chunks, forced=[True, True, False], gap=0.1)
    assert len(finals) == 1
    assert finals[0].text == "明日の会議には参加できないと思います"
    assert finals[0].translation == "<明日の会議には参加できないと思います>"
    assert e.mt.calls[-1] == "明日の会議には参加できないと思います"
    drafts = [p for k, p in events if k == "translated" and not p.final]
    assert drafts, "earlier chunks should have produced live drafts"


def test_sentence_is_not_split_while_the_speaker_keeps_talking():
    # Chunks 2 s apart: the next chunk is being spoken, so the 0.3 s pause rule must not close the sentence.
    chunks = ["削減は中国の経済産出", "量に基づいて実施されるだろう", "と述べました"]
    _, events, finals = _run(chunks, forced=[True, True, False], gap=0.7)
    assert len(finals) == 1 and finals[0].text == "".join(chunks)


def test_backlogged_chunks_stay_in_their_sentence():
    # Slow CPU: chunks queue up for ASR and their audio ended long ago; no timer may close the sentence early.
    chunks = ["合金とは基本的には2種類以上の金", "属の混合物です"]
    _, _, finals = _run(chunks, forced=[True, False], asr_delay=0.5, ended_ago=15, chunk_hard_s=1.0)
    assert len(finals) == 1 and finals[0].text == "".join(chunks)


def test_final_uses_whole_sentence_recognition():
    # Chunk-by-chunk ASR duplicated a word at the cut; the whole-sentence pass fixes the final text.
    events, done = [], threading.Event()

    def on_event(kind, payload):
        events.append((kind, payload))
        if kind == "translated" and payload.final:
            done.set()

    e = Engine(Settings(my_lang="vi", their_lang="ja", sentence_gap_s=0.3), on_event)
    e.asr = {"ja": FakeAsr(["日本は一番", "一番新鮮な魚です"], whole="日本は一番新鮮な魚です")}
    e.mt = FakeMt()
    e._start_workers()
    e._enqueue("them", np.full(24000, 0, np.float32), time.time(), True)
    time.sleep(0.1)
    e._enqueue("them", np.full(24000, 1, np.float32), time.time(), False)
    assert done.wait(5)
    e.stop()
    final = [p for k, p in events if k == "translated" and p.final][0]
    assert final.text == "日本は一番新鮮な魚です" and final.translation == "<日本は一番新鮮な魚です>"


def test_rescore_can_be_disabled():
    chunks = ["日本は一番", "一番新鮮な魚です"]
    _, _, finals = _run(chunks, forced=[True, False], gap=0.1, rescore_final=False)
    assert finals[0].text == "日本は一番一番新鮮な魚です"


def test_english_is_not_recognised_twice():
    # English finals keep the chunk text: a second pass would not change the translation, only delay it.
    chunks = ["The meeting tomorrow", "starts at nine."]
    _, _, finals = _run(chunks, forced=[True, False], lang="en", gap=0.1)
    assert finals[0].text == "The meeting tomorrow starts at nine."
    assert "rescore_ms" not in finals[0].timings


def test_stale_draft_is_cancelled_when_the_sentence_ends():
    # A slow draft of the first chunk is running when the sentence closes with more text: it must be
    # abandoned, and the final translation of the whole sentence must still arrive.
    chunks = ["明日の会議には", "参加できないと思います"]
    e, events, finals = _run(chunks, forced=[True, False], mt_delay=1.0, gap=0.2, rescore_final=False)
    assert e.mt.cancelled == ["明日の会議には"]
    assert finals[0].translation == "<明日の会議には参加できないと思います>"


def test_final_is_not_stuck_behind_the_next_sentences_draft():
    # Real speech: sentence A ends when the next chunk starts sentence B. B's (slow) draft starts while A is
    # re-recognised; A's final must not wait for that draft.
    chunks = ["明日は雨", "ですね", "それでも行きます"]
    e, events, finals = _run(chunks, forced=[True, True, True], mt_delay=1.0, asr_delay=0.3)
    a = finals[0]
    assert a.text == "明日は雨ですね"
    assert "それでも行きます" in e.mt.cancelled
    done = [(p.id, p.final) for k, p in events if k == "translated"]
    first_b = next((i for i, (sid, _) in enumerate(done) if sid != a.id), len(done))
    assert done.index((a.id, True)) < first_b


def test_partial_translation_streams_and_never_shrinks_a_draft():
    chunks = ["明日の会議には", "参加できないと思います"]
    e, events, finals = _run(chunks, forced=[True, False], mt_delay=0.6, gap=0.8, rescore_final=False)
    partials = [p for k, p in events if k == "partial"]
    assert partials, "a slow translation should stream partial text"
    for p in partials:
        assert p.translation is None or len(p.partial) >= len(p.translation)
    assert finals[0].partial is None and finals[0].translation == "<明日の会議には参加できないと思います>"
    assert "final_first_ms" in finals[0].timings


def test_slow_translation_skips_stale_drafts():
    chunks = [f"チャンク{i}、" for i in range(6)] + ["終わりです"]
    e, _, finals = _run(chunks, forced=[True] * 6 + [False], mt_delay=0.3, gap=0.02)
    assert finals[0].translation == "<" + "".join(chunks) + ">"
    assert len(e.mt.calls) < len(chunks)  # intermediate versions were coalesced


def test_pause_closes_sentence_without_punctuation():
    _, _, finals = _run(["hôm nay trời đẹp"], forced=[False], lang="vi", my_lang="ja")
    assert finals[0].text == "hôm nay trời đẹp" and finals[0].closed


def test_open_ended_text():
    assert open_ended("I went to the", "en") and open_ended("because", "en")
    assert not open_ended("I went to the store", "en") and not open_ended("another", "en")
    assert open_ended("昨日は雨が", "ja") and not open_ended("明日行きます", "ja")
    assert open_ended("tôi muốn đi đến", "vi") and not open_ended("tôi đi học rồi", "vi")


def test_hesitation_after_an_open_word_waits_longer():
    # "... to the" + pause: the speaker is looking for a word; a pause that would end a finished sentence
    # must not cut this one, so the next chunk still joins it.
    chunks = ["I would like to book a table at the", "restaurant downtown."]
    _, _, finals = _run(chunks, forced=[False, False], lang="en", gap=0.5, sentence_gap_open_s=1.5)
    assert len(finals) == 1 and finals[0].text == " ".join(chunks)


def test_sliver_segments_are_ignored():
    e = _engine(my_lang="vi", their_lang="ja")
    e.asr = {"ja": FakeAsr(["x"])}
    e._add_chunk("them", np.zeros(1600, np.float32), time.time(), True)  # 0.1 s
    assert e.open == {} and e.pending == {}


def test_same_language_is_not_translated():
    e, _, finals = _run(["hôm nay trời đẹp"], forced=[False], lang="vi", my_lang="vi")
    assert e.mt.calls == [] and finals[0].translation is None


def test_no_drafts_translates_once_per_sentence():
    chunks = ["明日の会議には", "参加できないと", "思います"]
    e, _, finals = _run(chunks, forced=[True, True, False], gap=0.1, drafts=False)
    assert e.mt.calls == ["明日の会議には参加できないと思います"]
