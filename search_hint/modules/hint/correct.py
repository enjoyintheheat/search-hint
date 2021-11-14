import spacy
from spellchecker import SpellChecker
import re
import numpy as np
from dotenv import load_dotenv


SMILES = {'🧳': 'Багаж',
'🌂': 'Зонтик',
'☂️': 'Зонтик',
'🧵': 'Нитки',
'🧶': 'Пряжа',
'👓': 'Очки',
'🕶️': 'Солнцезащитные очки',
'🥽': 'Очки',
'🥼': 'Лабораторный халат',
'🦺': 'Защитный жилет',
'👔': 'Галстук',
'👕': 'Футболка',
'👖': 'Джинсы',
'🧣': 'Шарф',
'🧤': 'Перчатки',
'🧥': 'Пальто',
'🧦': 'Носки',
'👗': 'Платье',
'👘': 'Кимоно',
'🥻': 'Сари',
'🩱': 'Цельный Купальник',
'🩲': 'Трусы',
'🩳': 'Шорты',
'👙': 'Бикини',
'👚': 'Женская одежда',
'👛': 'Кошелек',
'👜': 'Сумочка',
'👝': 'Сумка-клатч',
'🎒': 'Рюкзак',
'👞': 'Мужская обувь',
'👟': 'Кроссовки',
'🥾': 'Походный ботинок',
'🥿': 'Обувь На Плоской подошве',
'👠': 'Обувь На Высоком Каблуке',
'👡': 'Женская сандалия',
'🩰': 'Балетные туфли',
'👢': 'Женский ботинок',
'👑': 'Корона',
'👒': 'Женская шляпа',
'🎩': 'Цилиндр',
'🎓': 'Выпускной колпачок',
'🧢': 'Накладная крышка',
'🪖': 'Военный шлем',
'⛑️': 'Шлем спасателя',
'💄': 'Губная помада',
'💍': 'Кольцо',
'💼': 'Портфель'}
          
          
# %%
"""
### Загрузка языковых моделей
"""

# %%

#Загрузка русской и английской языковой модели    
spell_ru = SpellChecker(language='ru') 
spell_en = SpellChecker(language='en') 

nlp_ru = spacy.load("ru_core_news_lg", disable=['parser', 'ner'])
nlp_en = spacy.load("en_core_web_lg", disable=['parser', 'ner'])
    

# %%
"""
### Main
"""

# %%


def from_ghbdtn(text):    
    
    if not (',' in text or ';' in text or "'" in text):
        return text
    
    layout = dict(zip(map(ord, '''qwertyuiop[]asdfghjkl;'zxcvbnm,./`QWERTYUIOP{}ASDFGHJKL:"ZXCVBNM<>?~'''),
                               '''йцукенгшщзхъфывапролджэячсмитьбю.ёЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ,Ё'''))
                  
    return text.translate(layout)


def check_ru(s):
    search = []
    
    for t in s.split(' '):    
        misspelled = spell_ru.unknown([t])
    
        if len(misspelled) == 0:
            search.append(t)
            
        for word in misspelled:
            search.append(spell_ru.correction(word))
    
    doc = nlp_ru(" ".join(search))
    
    for tok in doc:    
        if tok.pos_ == "NOUN" and tok.lemma_ != '':
            return " ".join(search)

        
def check_en(s):
    search = []
    
    for t in s.split(' '):    
        misspelled = spell_en.unknown([t])
    
        if len(misspelled) == 0:
            search.append(t)
            
        for word in misspelled:
            search.append(spell_en.correction(word))
    
    doc = nlp_en(" ".join(search))
    
    for tok in doc:   
        if tok.pos_ == "NOUN" and tok.lemma_ != '':
            return " ".join(search)
    

def correction(s):
    if s[0] in SMILES.keys():
        s = SMILES[s[0]]
        
    s = s.lower()
    s1 = '%s' % s
    s = re.sub(r'[^A-Za-zА-Яа-я0-9 ]', '', s)
    s = s.replace('\x22', '')
    
    lang = 'ru'
    txt = check_ru(s)
    if txt == '' or txt is None:
        txt = check_en(s)
        lang = 'en'
    
    if txt is None or lang == 'en':
        s1 = from_ghbdtn(s1)
        s1 = re.sub(r'[^A-Za-zА-Яа-я0-9 -]', '', s1)
        s1 = s1.replace('\x22', '')
        t = check_ru(s1)
        if t is None:
            txt = s1 
        else:
            txt = t
    else:
        txt = s
        
    return txt.strip()
