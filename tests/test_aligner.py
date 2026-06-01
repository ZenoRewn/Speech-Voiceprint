from merger.aligner import MIXED, UNKNOWN, align
from stt.base import STTResult, Utterance, Word
from voiceprint.base import SpeakerSegment


def make_stt(utterances):
    return STTResult(utterances=utterances, language="zh-CN", duration=10.0)


def test_word_assigned_by_max_overlap():
    # word1 0.0-0.5 fully inside Speaker_A; word2 0.5-1.0 has 0.1s in A and 0.4s in B.
    words = [Word("你好", 0.0, 0.5), Word("世界", 0.5, 1.0)]
    utt = Utterance(text="你好 世界", start=0.0, end=1.0, words=words)
    segments = [
        SpeakerSegment(0.0, 0.6, "Speaker_A"),
        SpeakerSegment(0.6, 1.0, "Speaker_B"),
    ]
    aligned = align(make_stt([utt]), segments, utterance_majority_threshold=0.5)
    assert [w.speaker for w in aligned[0].words] == ["Speaker_A", "Speaker_B"]


def test_word_falls_back_to_midpoint_when_no_overlap():
    words = [Word("孤词", 5.0, 5.4)]
    utt = Utterance(text="孤词", start=5.0, end=5.4, words=words)
    # Segment that doesn't strictly overlap (zero-overlap edge): start=5.4 end=6.0
    segments = [SpeakerSegment(0.0, 4.0, "Speaker_A"), SpeakerSegment(5.4, 6.0, "Speaker_B")]
    aligned = align(make_stt([utt]), segments)
    assert aligned[0].words[0].speaker == UNKNOWN  # midpoint 5.2 sits in the gap


def test_word_midpoint_picks_segment_in_gap():
    words = [Word("中点", 1.0, 1.2)]
    utt = Utterance(text="中点", start=1.0, end=1.2, words=words)
    # Word range fully inside Speaker_A segment but the boundary calc picks via overlap first.
    segments = [SpeakerSegment(0.5, 1.5, "Speaker_A")]
    aligned = align(make_stt([utt]), segments)
    assert aligned[0].words[0].speaker == "Speaker_A"


def test_utterance_marked_mixed_when_no_majority():
    words = [
        Word("a", 0.0, 0.3),
        Word("b", 0.3, 0.6),
        Word("c", 0.6, 0.9),
        Word("d", 0.9, 1.2),
    ]
    utt = Utterance(text="a b c d", start=0.0, end=1.2, words=words)
    segments = [
        SpeakerSegment(0.0, 0.6, "Speaker_A"),
        SpeakerSegment(0.6, 1.2, "Speaker_B"),
    ]
    aligned = align(make_stt([utt]), segments, utterance_majority_threshold=0.7)
    assert aligned[0].speaker == MIXED
    assert aligned[0].speaker_confidence == 0.5


def test_utterance_takes_majority_speaker():
    words = [
        Word("w1", 0.0, 0.2),
        Word("w2", 0.2, 0.4),
        Word("w3", 0.4, 0.6),
        Word("w4", 0.95, 1.05),
    ]
    utt = Utterance(text="w1 w2 w3 w4", start=0.0, end=1.05, words=words)
    segments = [
        SpeakerSegment(0.0, 0.9, "Speaker_A"),
        SpeakerSegment(0.9, 1.2, "Speaker_B"),
    ]
    aligned = align(make_stt([utt]), segments, utterance_majority_threshold=0.7)
    assert aligned[0].speaker == "Speaker_A"
    assert aligned[0].speaker_confidence == 0.75


def test_no_segments_returns_unknown():
    words = [Word("hi", 0.0, 0.3)]
    utt = Utterance(text="hi", start=0.0, end=0.3, words=words)
    aligned = align(make_stt([utt]), segments=[])
    assert aligned[0].speaker == UNKNOWN
    assert aligned[0].words[0].speaker == UNKNOWN


def test_word_to_dict_round_trip():
    words = [Word("x", 0.123456, 0.789012)]
    utt = Utterance(text="x", start=0.123456, end=0.789012, words=words, azure_speaker="Guest-1")
    segments = [SpeakerSegment(0.0, 1.0, "Speaker_A")]
    aligned = align(make_stt([utt]), segments)
    d = aligned[0].to_dict()
    assert d["speaker"] == "Speaker_A"
    assert d["azure_speaker"] == "Guest-1"
    assert d["words"][0]["start"] == round(0.123456, 3)
