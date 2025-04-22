from franco_arabic_transliterator.franco_arabic_transliterator import *

str = 'hi bro kifak nchalla kello tamem, lek 3tine hayda'
transliterator = FrancoArabicTransliterator()

# Pick up one of the disambiguation methods
print(transliterator.transliterate(str, method="lexicon")) # ازيك يا حبيبي
# print(transliterator.transliterate(str, method="language-model")) # ازيك يا حبيبي

