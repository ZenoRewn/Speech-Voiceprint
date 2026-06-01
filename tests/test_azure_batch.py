import pytest

from stt.azure_batch import _parse_iso_duration, AzureBatchTranscription


@pytest.mark.parametrize(
    "value,expected",
    [
        ("PT1.5S", 1.5),
        ("PT0S", 0.0),
        ("PT0.84S", 0.84),
        ("PT1M30S", 90.0),
        ("PT2H3M4.5S", 2 * 3600 + 3 * 60 + 4.5),
        (None, 0.0),
        ("", 0.0),
        (12.5, 12.5),
        (3, 3.0),
    ],
)
def test_iso_duration_parses(value, expected):
    assert _parse_iso_duration(value) == pytest.approx(expected)


def test_merge_files_assembles_utterances():
    sample = [
        {
            "duration": "PT3.2S",
            "recognizedPhrases": [
                {
                    "offset": "PT0.1S",
                    "duration": "PT1.0S",
                    "speaker": 1,
                    "locale": "en-US",
                    "nBest": [
                        {
                            "display": "Hello world.",
                            "confidence": 0.93,
                            "displayWords": [
                                {"displayText": "Hello", "offset": "PT0.1S", "duration": "PT0.4S"},
                                {"displayText": "world.", "offset": "PT0.5S", "duration": "PT0.6S"},
                            ],
                        }
                    ],
                },
                {
                    "offset": "PT1.5S",
                    "duration": "PT1.0S",
                    "speaker": 2,
                    "locale": "en-US",
                    "nBest": [
                        {
                            "display": "Hi there.",
                            "displayWords": [
                                {"displayText": "Hi", "offset": "PT1.5S", "duration": "PT0.3S"},
                                {"displayText": "there.", "offset": "PT1.8S", "duration": "PT0.7S"},
                            ],
                        }
                    ],
                },
            ],
        }
    ]
    result = AzureBatchTranscription._merge_files(sample)
    assert len(result.utterances) == 2
    assert result.utterances[0].azure_speaker == "Guest-1"
    assert result.utterances[1].azure_speaker == "Guest-2"
    assert result.language == "en-US"
    assert result.duration == pytest.approx(3.2)
    assert result.utterances[0].words[0].text == "Hello"
    assert result.utterances[0].words[1].end == pytest.approx(1.1)
