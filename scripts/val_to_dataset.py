#!/usr/bin/python3


def cut_endl(s):
    if s and s[-1] == '\n':
        return s[:-1]
    return s


def dump(text, vulg, fo):
    assert len(text) == len(vulg)
    fo.write(text + '\n')
    fo.write(vulg + '\n')


def main():
    with open('supervised.txt', 'w') as fo:
        #for fname in 'val1.txt', 'val2.txt', 'val3.txt':
        for fname in 'supervised_train.txt', 'supervised_train2.txt':
            with open(fname) as fi:
                text = []
                vulg = []
                exp_vulg = False
                for raw_line in fi:
                    line = cut_endl(raw_line)
                    if not line:
                        assert not exp_vulg
                        if text:
                            dump(''.join(text), ''.join(vulg), fo)
                        text = []
                        vulg = []
                        exp_vulg = False
                    elif not exp_vulg:
                        text.extend(list(line))
                        exp_vulg = True
                    else:
                        vulg.extend(list(line))
                        exp_vulg = False


if __name__ == '__main__':
    main()
