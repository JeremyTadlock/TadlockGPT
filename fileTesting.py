import os
from email.parser import Parser
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
directory_names = []
file_names = []
root_dir = "C:\\Users\\Admin\\PycharmProjects\\MyGPT\\wow\\maildir"

counter = 0
for i in os.listdir(root_dir):
    current_dir = root_dir+"\\"+i
    if counter <= 5:
        counter+=1
        for j in os.listdir(current_dir):
            if j == "all_documents":
                directory_names.append(current_dir+"\\"+j)
                print(current_dir+"\\"+j)
                break
    else:
        break

data_text = ""
count = 0
for dir_name in directory_names:
    # for loop to loop through all files in dir (email txt files)
    if count < 2:
        count+=1
        #print("NAME:", dir_name)
        for path in os.listdir(dir_name):
            # check if current path is a file
            if os.path.isfile(os.path.join(dir_name, path)):
                file_name = dir_name + "\\" + path
                with open(file_name, "r") as f:
                    data = f.read()
                email = Parser().parsestr(data)
                body_text = email.get_payload()
                data_text = data_text + "\n" + body_text
                #print(path)



words = word_tokenize(data_text)
cleaned_words = [word for word in words if word not in stopwords.words('English')]
print(cleaned_words)
