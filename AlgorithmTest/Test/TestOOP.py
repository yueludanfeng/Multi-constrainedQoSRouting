# coding=utf-8
class Person(object):

    count = 0

    def __init__(self, name, age):
        """Initialize a person"""
        # 在name前加上__即可将name变为private成员变量，没有加的则是public成员变量
        self.__name = name
        self.age = age
        Person.count += 1

    def show_myself(self):
        print(' name = %s\
        \n age = %d' % (self.__name, self.age))
        # print(' I am the %dth Person' % Person.count)

    def __del__(self):
        """destroy a person"""
        Person.count -= 1
        # print('''I am to western paradise''')


class SchoolMember(object):
    def __init__(self, name, age):
        self.__name = name
        self.age = age

    def tell(self):
        print('name=%s \nage=%d' % (self.__name, self.age))


class Teacher(SchoolMember):

    def __init__(self, name, age, salary):
        """Init a Teacher"""
        SchoolMember.__init__(self, name, age)#注意此处的self必须加上，不然的话会出错
        self.salary = salary

    def tell(self):
        SchoolMember.tell(self)
        print('salary=%s' % self.salary)


class Student(SchoolMember):
    def __init__(self, name, age, marks):
        """Init a Student"""
        SchoolMember.__init__(self, name, age)#注意此处的self必须加上，不然的话会出错
        self.marks = marks

    def tell(self):
        SchoolMember.tell(self)
        print('marks=%s' % self.marks)

p = Person('lxm', 24)
p.show_myself()

# teacher = Teacher('zhangsan', 28, 10000)
# student = Student('lxsi', 24, 100)
#
# member = [teacher, student]
# for item in member:
#     item.tell()
#
# print 'age=%d' % teacher.age


class A(object):
    def __init__(self):
        self.__private()
        self.public()

    def __private(self):
        print 'A.__private()'

    def public(self):
        print 'A.public()'


class B(A):
    def __private(self):
        print 'B.__private()'

    def public(self):
        print 'B.public()'


b = B()

b.public()
