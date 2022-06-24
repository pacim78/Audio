from ast import Raise
from logging import warning
import sys, re, warnings, copy
import pandas as pd
import numpy

def numword(doc):
    newdoc = doc
    pattern = "\.\d+|\d+"
    df = {
        "0" : ["zero"],
        "1" : ["uno"],
        "2" : ["due"],
        "3" : ["tre"],
        "4" : ["quattro"],
        "5" : ["cinque"],
        "6" : ["sei"],
        "7" : ["sette"],
        "8" : ["otto"],
        "9" : ["nove"],
        "10" : ["dieci"],
        "11" : ["undici"],
        "12" : ["dodici"],
        "13" : ["tredici"],
        "14" : ["quattordici"],
        "15" : ["quindici"],
        "16" : ["sedici"],
        "17" : ["diciassette"],
        "18" : ["diciotto"],
        "19" : ["diciannove"],
        "20" : ["venti"],
        "30" : ["trenta"],
        "40" : ["quaranta"],
        "50" : ["cinquanta"],
        "60" : ["sessanta"],
        "70" : ["settanta"],
        "80" : ["ottanta"],
        "90" : ["novanta"],
        "100" : ["cento"],
        "1000" : ["mille","mila"],
        "1000000" : ["un milione","milioni"],
        "1000000000" : ["un miliardo","miliardi"]
    }
    lista = re.finditer(pattern,doc)
    def intnum(num):
        wrd = str(num)
        if num == 0:
            replace = ""
        elif num <= 20:
            replace = df[wrd][0]
        elif num < 100:
            if str(num)[-1] in ["1","8"]:
                dec = df[wrd[0]+"0"][0][:-1]
            else:
                dec = df[wrd[0]+"0"][0] 
            replace = " ".join(
                (
                    dec,
                    intnum(int(wrd[-1:]))
                )
            )
        elif num < 200:
            replace = " ".join(
                (
                    df["100"][0],
                    intnum(int(wrd[-2:]))
                )
            )
        elif num < 1000:
            replace = " ".join(
                (
                    df[wrd[0]][0],
                    df["100"][0],
                    intnum(int(wrd[-2:]))
                )
            )
        elif num < 2000:
            replace = " ".join(
                (
                    df["1000"][0],
                    intnum(int(wrd[-3:]))
                )
            )
        elif num < 1000000:
            replace = " ".join(
                (
                    intnum(int(wrd[0:-3])),
                    df["1000"][1],
                    intnum(int(wrd[-3:]))
                )
            )
        elif num < 2000000:
            replace = " ".join(
                (
                    df["1000000"][0],
                    intnum(int(wrd[-6:]))
                )
            )
        elif num < 1000000000:
            replace = " ".join(
                (
                    intnum(int(wrd[0:-6])),
                    df["1000000"][1],
                    intnum(int(wrd[-6:]))
                )
            )
        elif num < 2000000000:
            replace = " ".join(
                (
                    df["1000000000"][0],
                    intnum(int(wrd[-9:]))
                )
            )
        elif num < 1000000000000:
            replace = " ".join(
                (
                    intnum(int(wrd[0:-9])),
                    df["1000000000"][1],
                    intnum(int(wrd[-9:]))
                )
            )
        else:
            warnings.warn("number over one trillion", DeprecationWarning)
            replace = ""
        return replace
    def pointnum(num):
        wrd = num[1:]
        num_list = []
        for letter in wrd:
            num_list.append(intnum(int(letter)))
        replace = " ".join(num_list)
        return " punto " + replace+" "
    for i,match in enumerate(lista):
        if doc[match.span()[0]] == ".":
            replace = pointnum(str(match.group().strip()))
            newdoc = re.sub(pattern,replace,newdoc,1)
        else:
            wrd = str(match.group().strip())
            num = int(wrd)
            if num == 0:
                replace = "zero"
            else:
                replace = intnum(num)
            newdoc = re.sub(pattern,replace,newdoc,1)
    return " "+newdoc.strip()+" "

def phoneme(doc):
    path = "C:/Users/Paolo/OneDrive/Desktop/audio/preproc/phoneme_data.csv"
    df = pd.read_csv(path)
    doc = numword(doc).lower()
    doc = [x for x in doc]
    docnew = copy.deepcopy(doc)
    for row in df.iterrows():
        p = row[1].pattern
        i = row[1].indexp
        fakedoc = []
        for x in docnew:
            try:
                int(x)
                fakedoc.append("1")
            except:
                fakedoc.append(x)
        for _ in re.finditer(p,"".join(fakedoc)):
            fakedoc = []
            for x in docnew:
                try:
                    int(x)
                    fakedoc.append("1")
                except:
                    fakedoc.append(x)
            m = re.search(p,"".join(fakedoc))
            if (docnew[m.span()[0]]).isdigit():
                pass
            else:
                try:
                    gr = m.group(1)
                    docnew[m.span()[0]:m.span()[0]+len(gr)] = [(i)]
                except:
                    try:
                        docnew[m.span()[0]:m.span()[1]] = [(i)]
                    except:
                        pass
    if len(docnew) < 372:
        docnew = numpy.concatenate((docnew,[99]*(372-len(docnew))))
    else:
        print('length problem with label')
    return docnew

def ph_to_txt(phonemes):
    res = ""
    path = "C:/Users/Paolo/OneDrive/Desktop/audio/preproc/phoneme_data.csv"
    df = pd.read_csv(path)
    for ph in phonemes:
        if ph == 99:
            pass
        else:
            transcription = list(df[df.indexp==int(ph)].phoneme)[0]
            res+=transcription
    return res