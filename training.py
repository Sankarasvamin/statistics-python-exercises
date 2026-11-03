import sys

lst = [line.rstrip("\n") for line in sys.stdin]
n = int(lst[0])
for i in range(1,int(n + 1)):
    if len(lst[i]) <= 10:
        print("lst[i]")
    else:
        print(lst[i][0] + str(int(len(lst[i]) - 2)) + lst[i][-1])
                       
                       
