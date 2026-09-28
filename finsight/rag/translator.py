from langdetect import detect

class LanguageBridge:
    def __init__(self):
        try:
            from deep_translator import GoogleTranslator
            self.GoogleTranslator = GoogleTranslator
        except Exception:
            self.GoogleTranslator = None

    def detect(self, text):
        try:
            lang = detect(text)
        except Exception:
            lang = "en"
        return {"hi": "hi", "ta": "ta", "en": "en"}.get(lang, "en")

    def to_english(self, text, lang):
        if lang == "en" or self.GoogleTranslator is None:
            return text
        try:
            return self.GoogleTranslator(source=lang, target="en").translate(text)
        except Exception:
            return text

    def from_english(self, text, lang):
        if lang == "en" or self.GoogleTranslator is None:
            return text
        try:
            return self.GoogleTranslator(source="en", target=lang).translate(text)
        except Exception:
            return text
