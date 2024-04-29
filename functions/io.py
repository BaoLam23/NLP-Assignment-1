def load_w2v(filename):
    w2v = {}
    w2vFile = open(filename, 'r', encoding='utf-8')
    w2v_size = int(w2vFile.readline())
    w2v_dim = int(w2vFile.readline())
    for i in w2vFile:
        s = i.split()
        v = [float(val) for val in s[1:]]
        w2v[s[0].strip()] = v
    w2vFile.close()
    # return w2v, w2v_size, w2v_dim
    return w2v

def write_output(filename, lines):
    file = open(filename, "a")
    file.writelines(lines)
    file.close();

def load_test_data(filename):
    X=[]
    Y=[]
    with open(filename) as f:
        f.readline()
        for l in f:
            w1, w2, label = l.strip().split()
            X.append(w1)
            Y.append(w2)
            raw_label.append(label)