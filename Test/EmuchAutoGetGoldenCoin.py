# coding=utf-8
# Author:   Lixinming
import urllib2
import urllib
import cookielib
import re


def swith(string=""):
    res = string.split('加')
    if len(res) == 1:
        pass
    else:
        ans = int(res[0]) + int(res[1])
        return ans

    res = string.split('减')
    if len(res) == 1:
        pass
    else:
        ans = int(res[0]) - int(res[1])
        return ans

    res = string.split('乘以')
    if len(res) == 1:
        pass
    else:
        ans = int(res[0]) * int(res[1])
        return ans

    res = string.split('除以')
    if len(res) == 1:
        pass
    else:
        ans = int(res[0]) / int(res[1])
        return ans


def extract_info_from_string(string=""):
    str = string
    a = str.split('等于')
    # print str
    # print a
    ans = a[0].split("问题：")
    print 'ans=',ans
    # print ans[1]
    print 'result=', swith(ans[1])

class Emuch_Logger:

    def __init__(self):
        self.url_login = "http://emuch.net/bbs/logging.php?action=login"
        """
        username = vanillasmile
        password = ******
        cookietime = 31536000
        loginsubmit = 会员登录

        """
        self.form_data_dict = {
            'username':'710506937',
            'password':'lxm123',
            'cookietime': 31536000,
            'loginsubmit':'会员登录'

        }
        self.credit_url = "http://emuch.net/bbs/logging.php?action=getcredit"


    def add_form_data(self, dict_obj=dict()):
        for k,v in dict_obj.iteritems():
            self.form_data_dict[k] = v

    @staticmethod
    def get_hash_code(tag_name, response):
        hash_regex = r'(<input.+name=")(' + tag_name + r')(" value=")([\d\w]+)(">)'
        m = re.search(hash_regex, response)
        if m:
            # retrun tag_name, hash_code
            return m.group(2), m.group(4)
        else:
            return

    @staticmethod
    def get_credit_number(response):
        # regex = r'(<u>\xbd\xf0\xb1\xd2: )(\d\d\.\d)(</u>)'
        regex_float = r'(<u>\xbd\xf0\xb1\xd2: )(\d*\.\d+)(</u>)'
        regex_int = r'(<u>\xbd\xf0\xb1\xd2: )(\d*)(</u>)'
        m_float = re.search(regex_float, response)
        m_int = re.search(regex_int, response)
        if m_float:
            credit_num_str = m_float.group(2)
        if m_int:
            credit_num_str = m_int.group(2)
        return credit_num_str

    def post_with_cookie(self):
        print '----------enter into post_with_cookie function----'
        "post data with cookie setting, return page string and cookie tuple"
        # get formhash
        url_login = self.url_login
        response = urllib2.urlopen(url_login).read()
        formhash = self.get_hash_code('formhash', response)
        # update form_data_dict
        self.add_form_data({'formhash': formhash})
        # set cookie
        cj = cookielib.CookieJar()
        form_data = self.form_data_dict
        try:
            print '------------enter into try block----------'
            opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))
            print '----------A'
            urllib2.install_opener(opener)
            print '----------B'
            req = urllib2.Request(url_login, urllib.urlencode(form_data))
            print '----------C'
            u = urllib2.urlopen(req)
            print '----------D'
            cookie_list = []
            print '-----------E'
            # for index, cookie in enumerate(cj):
            #     cookie_list.append(cookie)
            # return u.read(), tuple(cookie_list)
            return u.read()
            print '-----------F'
        except:
            print '----------enter into except block--------'
            print "Ooops! Failed to log in !>_< there may be a problem."
            return None

    def log_in(self):
        """
        Method to pass values by POST 2 times to log in emuch.net,
        return cookie tuple.
        """
        num1, num2, operation = 0, 0, ''
        qustion_regex = r'(\xce\xca\xcc\xe2\xa3\xba)(\d+)(.+)(\d+)' + \
                        r'(\xb5\xc8\xd3\xda\xb6\xe0\xc9\xd9?)'
        while not (num1 and num2 and operation):
            # response = self.post_with_cookie()[0]
            response = self.post_with_cookie()
            response = response.encode('utf-8')
            response = ''''''+response

            print 'response=',response
            # match_obj = re.search(qustion_regex, response)
            print '--------------hhhhhhhhhhhhh'
            answer = extract_info_from_string(response)
            # get question parts
        #     try:
        #         num1, num2, operation = match_obj.group(2), match_obj.group(4), \
        #                                 match_obj.group(3)
        #         print '--------operation is ----'
        #         print 'num1=',num1
        #         print 'num2=',num2
        #         print 'operation=',operation
        #
        #         # return num1, num2, operation
        #     except:
        #         print "failed to get question"
        #         # time.sleep(6)
        #         pass
        # # further log in
        # calculate verify question
        # division
        # if operation == '\xb3\xfd\xd2\xd4':
        #     answer = str(int(num1) / int(num2))
        # # multiplication
        # if operation == '\xb3\xcb\xd2\xd4':
        #     answer = str(int(num1) * int(num2))
        # get formhash value
        formhash = self.get_hash_code('formhash', response)[1]
        # get post_sec_hash value
        post_sec_hash = self.get_hash_code('post_sec_hash', response)[1]
        # update form_data_dict
        self.add_form_data({'formhash': formhash,
                            'post_sec_code': answer,
                            'post_sec_hash': post_sec_hash
                            })
        # login_response = self.post_with_cookie()

        # cookies_tup = self.post_with_cookie()[1]
        # return cookies_tup

    def get_credit(self):
        """get today's credit,
           if get, return page content, else return 'have_got' and credit_num
        """
        # get formhash value
        req_1 = urllib2.Request(self.credit_url,
                                urllib.urlencode({'getmode': '1'}))
        response_1 = urllib2.urlopen(req_1).read()
        if self.get_hash_code('formhash', response_1):
            formhash = self.get_hash_code('formhash', response_1)[1]
            # formhash = self.get_hash_code_BSoup('formhash', credit_url)
            credit_form_data = {'getmode': '1', 'creditsubmit': '领取红包'}
            credit_form_data['formhash'] = formhash
            setattr(self, 'credit_form_data', credit_form_data)
            # post values to get credit
            data = urllib.urlencode(credit_form_data)
            req_2 = urllib2.Request(self.credit_url, data)
            response_2 = urllib2.urlopen(req_2).read()
            if response_2:
                credit_num = self.get_credit_number(response_2)
                self.log(event='get_credit_succeed', credit_num=credit_num)
            return response_2
        else:
            # print 'got!'
            credit_num = self.get_credit_number(
                self.send_post(self.credit_url, self.form_data_dict))
            self.log(event='get_credit_fail', credit_num=credit_num)
            return 'have_got', credit_num



def emuch(emuch_logger):
    url = emuch_logger.url_login
    #chk internet connection
    try:
        resp = urllib2.urlopen(url)
    except:
        return 'no_internet'
    # title_1 = emuch_logger.get_page_title(url)
    # try:
    #     title_2 = emuch_logger.get_page_title('http://www.weibo.com/')
    # except:
    #     title_2 = None
    # try:
    #     title_3 = emuch_logger.get_page_title('http://www.baidu.com/')
    # except:
    #     title_3 = None
    # if title_1 == title_2 or title_1 == title_3:
    #     return 'need_login'
    if False:
        pass
    else:
        cookies = emuch_logger.log_in()
        #get credit
        response = emuch_logger.get_credit()
        #check if there is a formhash tag
        if 'have_got' in response:
            print 'You\'ve got today\'s coin~'
            print 'Current credit number : %s' % str(response[1])
            return
        else:
            credit_num = emuch_logger.get_credit_number(response)
            print "Today's credit -> get!"
            print 'Current credit number : %s' % str(credit_num)
            return

emuch_logger = Emuch_Logger()
emuch(emuch_logger)