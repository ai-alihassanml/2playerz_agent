import langid

text = "Салам"
lang, confidence = langid.classify(text)
print(lang, confidence) 
