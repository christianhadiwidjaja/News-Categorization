from summarizer import Summarizer

def bert_summarizer(raw_text):
    model = Summarizer()
    result = model(raw_text, min_length=20)
    summary = "".join(result)

    return summary