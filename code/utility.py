# ref pythonworld
def isNum(stri):
    try:
        float(stri)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(stri)
        return True
    except (TypeError, ValueError):
        pass
    return False

def doEncode(data, attrs, lines, attrNum, sampleNum):
    length = len(data)
    length1 = len(lines)
    for j in range(attrNum):
        nc = 0
        n_attr = {}
        if not isNum(lines[0][j]):
            attrs.add(j)
        for i in range(sampleNum):
            if isNum(lines[i][j]):
                data[i][j] = float(lines[i][j])
            elif lines[i][j] in n_attr:
                data[i][j] = n_attr[lines[i][j]]
            else:
                n_attr[lines[i][j]] = nc
                data[i][j] = n_attr[lines[i][j]]
                nc += 1
    return data,attrs

def calculateGini(data, column, sv, category):
    sc = [0,0,0,0]
    gini = 0.0
    length = len(data)
    if not category:
        for i in range(length):
            if data[i][column] <= sv:
                sc[int(data[i][-1])] = sc[int(data[i][-1])] + 1
            else:
                sc[2+int(data[i][-1])] = sc[2+int(data[i][-1])] + 1
    else:
        for i in range(length):
            if data[i][column] == sv:
                sc[int(data[i][-1])] = sc[int(data[i][-1])] + 1
            else:
                sc[2+int(data[i][-1])] = sc[2+int(data[i][-1])] + 1
    sclen1 = len(sc)
    score = [0.0,0.0]
    if (sc[0] + sc[1]) != 0:
        for i in range(2):
            score[0] = score[0] + (sc[i]  / (sc[0] + sc[1]))**2
        gini = gini + (1.0 - score[0]) * ((sc[0] + sc[1]) / (length))
    if (sc[2] + sc[3]) != 0:
        for i in range(2):
            score[1] = score[1] + (sc[2+i] / (sc[2] + sc[3]))**2
        gini = gini + (1.0 - score[1]) * ((sc[2] + sc[3]) / (length))
    return gini

def calculateGiniRF(data, column, sv, category):
    sc = [0,0,0,0]
    gini = 0.0
    length = len(data)
    if not category:
        for i in range(length):
            if data[i][column] < sv:
                sc[int(data[i][-1])] = sc[int(data[i][-1])] + 1
            else:
                sc[2+int(data[i][-1])] = sc[2+int(data[i][-1])] + 1
    else:
        for i in range(length):
            if data[i][column] == sv:
                sc[int(data[i][-1])] = sc[int(data[i][-1])] + 1
            else:
                sc[2+int(data[i][-1])] = sc[2+int(data[i][-1])] + 1
    sclen1 = len(sc)
    score = [0.0,0.0]
    if (sc[0] + sc[1]) != 0:
        for i in range(2):
            score[0] = score[0] + (sc[i]  / (sc[0] + sc[1]))**2
        gini = gini + (1.0 - score[0]) * ((sc[0] + sc[1]) / (length))
    if (sc[2] + sc[3]) != 0:
        for i in range(2):
            score[1] = score[1] + (sc[2+i] / (sc[2] + sc[3]))**2
        gini = gini + (1.0 - score[1]) * ((sc[2] + sc[3]) / (length))
    return gini
