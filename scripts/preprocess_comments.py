#!/usr/bin/python3

def remove_tags(line, opening, closing):
    stack = []
    for c in line:
        if c == closing:
            while stack and stack[-1] != opening:
                stack.pop()
            if stack:
                stack.pop()
        else:
            stack.append(c)
    return ''.join(stack)

with open('comments.csv') as f:
    with open('filtered_comments.csv', 'w') as g:
        for raw_line in f:
            line = raw_line[raw_line.index(',')+1:]
            if line[0] == '"':
                line = line[1:-2]
            line = line.strip()
            line = remove_tags(line, '<', '>')
            line = remove_tags(line, '&', ';')
            if line:
                g.write(line + '\n')
