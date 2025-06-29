from model_utils import predict_sentiment

def test_predict_sentiment_single():
    result = predict_sentiment("I love this!")
    assert result in ["Positive", "Negative"]

def test_predict_sentiment_batch():
    results = predict_sentiment(["I love this!", "I hate this!"])
    assert isinstance(results, list)
    assert all(r in ["Positive", "Negative"] for r in results)

def test_predict_sentiment_proba():
    results = predict_sentiment(["I love this!", "I hate this!"], return_proba=True)
    assert isinstance(results, list)
    for sentiment, proba in results:
        assert sentiment in ["Positive", "Negative"]
        assert isinstance(proba, list) and len(proba) == 2
