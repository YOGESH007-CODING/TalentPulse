from src.aspect_extraction import detect_aspects
from src.data_prep import sentiment_from_rating


def test_sentiment_mapping():
    assert [sentiment_from_rating(value) for value in range(1, 6)] == ["Negative", "Negative", "Neutral", "Positive", "Positive"]


def test_multiple_aspects_are_detected():
    assert set(detect_aspects("The battery will not charge and the packaging was damaged.")) == {"battery", "delivery"}
