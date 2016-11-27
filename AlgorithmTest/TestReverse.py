
li = [1,2,3,4,5]


class Test:

    def printList(self, li, start):

        print li[start]
        if len(li) == start+1:
          return
        else:
            self.printList(li, start+1)

T = Test()
T.printList(li, 0)
