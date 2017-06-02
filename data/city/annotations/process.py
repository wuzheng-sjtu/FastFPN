import os

pwd = os.getcwd()
for file in os.listdir(pwd):
    ext = file.split('.',1)[1]
    if ext == 'txt.txt':
        print '!'
        os.system('mv '+file + ' '+ file.split('.',1)[0]+'.txt')

