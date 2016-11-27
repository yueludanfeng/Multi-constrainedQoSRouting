f = open('test01.txt', 'r')
line = f.readline().split()
print line
node_num = line[0]
edge_num = line[1]
print node_num
print edge_num
print type(node_num)
a = int(node_num)
print type(a)
print a


def test(a,b,c,d,e):
    print a,b,c,d,e

def func(fp):
    while 1:
        line = fp.readline()
        if not line:
            break
        line = line.split(',')
        print type(int(line[0]))
        test(line[0],line[1],line[2],line[3],line[4])
func(f)
