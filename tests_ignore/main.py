from transliterate.discover import autodiscover
from transliterate import *
from transliterate.base import TranslitLanguagePack, registry

autodiscover()

class FrancoArabicLanguagePack(TranslitLanguagePack):
    language_code = "franco-arabic"
    language_name = "Franco Arabic"
    mapping = (
        u"abcdefghijklmnopqrstuvwxyz0123456789",
        u"abcdefghijklmnopqrstuvwxyz0123456789",
    )
    pre_processor_mapping = {
        "7": "h", "aa": "3", "8": "gh",
    }

registry.register(FrancoArabicLanguagePack)

text = "3tine hol el ghrad"
print(translit(text, 'franco-arabic'))