# coding=utf-8
# Author:   Lixinming

_g = "private global var"
g = "public global var"

class A:
    def __init__(self):
        self.public_var = "A public_var"
        self.__private_var = "A private_var"
        print "A's constructor is called!"

    def __private_func(self):
        print "A private function."

    def public_func(self):
        print "A public function."

    def pub_call_private(self):
        self.__private_func()

class B(A):
    def __init__(self):
        #在B中并不会默认调用值,除非你显式调用
        pass
        A.__init__(self)

    def __private_func(self):
        print "B private function."

    def b_pub_call_private(self):
        self.__private_func()
        B.__private_func()

