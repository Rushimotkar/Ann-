def mcp_andnot(x1, x2):
    w1 = 1
    w2 = -1
    threshold = 1
    net = (x1 * w1) + (x2 * w2)
    if net >= threshold:
        return 1
    else:
        return 0

print("X1 X2 Output")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(x1, x2, mcp_andnot(x1, x2))