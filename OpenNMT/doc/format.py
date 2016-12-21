import sys
opt = ""
desc = ""
for l in sys.stdin:
    if l.startswith("**"):
        if opt != "":
            print(""+opt[1:] + "\n:   " + desc+"\n")
        print("")
        print(l)
        print("")
        opt = ""
        desc = ""
    elif l.startswith("  -"):
        if opt != "":
            print(""+opt[1:] + "\n:   " + desc+"\n")
        opt = l.split(None)[0]
        desc = l.split(None, 1)[1].strip()
    else:
        desc += l.strip()
