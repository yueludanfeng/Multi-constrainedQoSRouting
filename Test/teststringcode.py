# coding=utf-8
# Author:   Lixinming
u = u'中文' #显示指定unicode类型对象u
str = u.encode('gb2312') #以gb2312编码对unicode对像进行编码
# str1 = u.encode('gbk') #以gbk编码对unicode对像进行编码
# str2 = u.encode('utf-8') #以utf-8编码对unicode对像进行编码

print str.decode('gb2312')#以gb2312编码对字符串str进行解码，以获取unicode
