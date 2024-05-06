import re
import os


class TxtAna:
    def __init__(self):
        # self.question = self.read_data(os.path.join(os.path.dirname(__file__), 'question.csv'))
        self.praise = self.read_data(os.path.join(os.path.dirname(__file__), 'praise.csv'))
        # self.question_en = self.read_data(os.path.join(os.path.dirname(__file__), 'question_en.csv'))
        self.praise_en = self.read_data(os.path.join(os.path.dirname(__file__), 'praise_en.csv'))
        self.talk_with_parent_list = self.read_data(os.path.join(os.path.dirname(__file__), 'talk_with_parent.txt'))
        self.talk_about_homework_list = self.read_data(os.path.join(os.path.dirname(__file__), 'talk_about_homework.txt'))
        self.talk_about_salary_list = self.read_data(os.path.join(os.path.dirname(__file__), 'talk_about_salary.txt'))


    def read_data(self, path):
        result = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip('\n')
                items = line.split(',')
                result.append(items[1])
        return result[:-1]

    # def is_question(self, text, language_type='cn'):  # 是疑问句
    #     question = self.question if language_type == "cn" else self.question_en
    #     for i in range(1, len(question)+1):
    #         if re.match(question[i], text) != None:
    #             return True
    #     return False

    def is_praise(self, text, language_type='cn'):  # 是口头表扬
        praise = self.praise if language_type == "cn" else self.praise_en
        for i in range(len(praise)):
            if re.match(praise[i], text) != None:
                return True
        return False
    
    def talk_with_parent(self,text,language='cn'):
        talk_with_parent_list=self.talk_with_parent_list if language=='cn' else []
        for i in range(len(talk_with_parent_list)):
            if re.match(talk_with_parent_list[i], text) != None:
                return True
        return False
    
    def talk_about_homework(self,text,language='cn'):
        talk_about_homework_list=self.talk_about_homework_list if language=='cn' else []
        for i in range(len(talk_about_homework_list)):
            if re.match(talk_about_homework_list[i], text) != None:
                return True
        return False
    
    def talk_about_salary(self,text,language='cn'):
        talk_about_salary_list=self.talk_about_salary_list if language=='cn' else []
        for i in range(len(talk_about_salary_list)):
            if re.match(talk_about_salary_list[i], text) != None:
                # print(text+"   "+talk_about_salary_list[i])
                return True,i
        return False,0
        


if __name__ == '__main__':
    txt_ana = TxtAna()
    print(txt_ana.is_question('你是叫彤彤吗'))
    print(txt_ana.is_praise('彤彤你真棒呀'))
    print(txt_ana.talk_about_salary('彤彤你真棒呀'))
