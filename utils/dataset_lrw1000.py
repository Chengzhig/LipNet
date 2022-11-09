import numpy
from PIL import Image, ImageEnhance, ImageFilter
from torch import permute
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import glob
import numpy as np
import random

from torchvision import transforms
from turbojpeg import TJPF_GRAY, TurboJPEG

from .cvtransforms import *
import torch
from collections import defaultdict

# from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE


jpeg = TurboJPEG()

pinyinlist = ['C', 'a', 'ai', 'ai ji', 'an', 'an jian', 'an quan', 'an zhao', 'ba', 'ba li',
              'ba xi', 'bai', 'ban', 'ban dao', 'ban fa', 'bang jia', 'bao', 'bao chi', 'bao dao',
              'bao gao', 'bao hu', 'bao kuo', 'bao yu', 'bao zhang', 'bei', 'bei bu', 'bei jing',
              'bei jing shi jian', 'bei yue', 'ben', 'ben ci', 'ben yue', 'bi', 'bi jiao', 'bi ru',
              'bi xu', 'bian hua', 'biao da', 'biao shi', 'biao zhi', 'biao zhun', 'bie', 'bing',
              'bing qie', 'bo chu', 'bu', 'bu duan', 'bu fen', 'bu guo', 'bu hui', 'bu jin', 'bu men',
              'bu neng', 'bu shao', 'bu shu', 'bu tong', 'bu yao', 'bu zhang', 'cai', 'cai fang',
              'cai qu', 'can jia', 'can yu', 'ce', 'ceng jing', 'chan pin', 'chan sheng', 'chan ye',
              'chang', 'chang qi', 'chang yi', 'chao guo', 'chao xian', 'che', 'cheng', 'cheng gong', 'cheng guo',
              'cheng li', 'cheng nuo', 'cheng shi', 'cheng wei', 'chi', 'chi xu', 'chu', 'chu lai', 'chu le', 'chu li',
              'chu xi', 'chu xian', 'chuan tong', 'chuang xin', 'chun', 'ci', 'ci ci', 'ci qian', 'cong', 'cu jin',
              'cun zai', 'cuo shi', 'da', 'da cheng', 'da dao', 'da gai', 'da hui', 'da ji', 'da jia', 'da xing',
              'da xue', 'da yu', 'da yue', 'da zao', 'dai', 'dai biao', 'dai lai', 'dan', 'dan shi', 'dang', 'dang di',
              'dang qian', 'dang shi', 'dang tian', 'dang zhong', 'dao', 'dao le', 'dao zhi', 'de', 'de dao', 'de guo',
              'deng', 'deng deng', 'di', 'di da', 'di fang', 'di qu', 'di zhi', 'dian', 'dian shi', 'diao cha',
              'diao yan',
              'dong', 'dong bu', 'dong fang', 'dong li', 'dong xi', 'dou', 'dou shi', 'du', 'duan', 'duan jiao', 'dui',
              'dui hua', 'dui wai', 'dui yu', 'duo', 'duo ci', 'duo nian', 'e', 'e luo si', 'er', 'er ling yi qi',
              'er qie',
              'er shi', 'er shi liu', 'er shi qi', 'er shi san', 'er shi si', 'er shi wu', 'er shi yi', 'er tong',
              'er wei ma',
              'fa', 'fa biao', 'fa bu', 'fa bu hui', 'fa chu', 'fa guo', 'fa hui', 'fa she', 'fa sheng', 'fa xian',
              'fa yan ren',
              'fa yuan', 'fa zhan', 'fan', 'fan dui', 'fan rong', 'fan wei', 'fan zui', 'fang', 'fang an', 'fang fan',
              'fang mian',
              'fang shi', 'fang wen', 'fang xiang', 'fei', 'fei chang', 'fen', 'fen bie', 'fen qi', 'fen zhong',
              'fen zi', 'feng hui',
              'feng shuo', 'feng xian', 'fu', 'fu jian', 'fu pin', 'fu wu', 'fu ze', 'fu ze ren', 'gai', 'gai bian',
              'gai ge', 'gai shan',
              'gan', 'gan jue', 'gan shou', 'gan xie', 'gang', 'gang gang', 'gao', 'gao du', 'gao feng', 'gao ji',
              'gao wen', 'gao xiao',
              'ge', 'ge de', 'ge fang', 'ge guo', 'ge jie', 'ge ren', 'ge wei', 'gei', 'gen', 'gen ju', 'geng',
              'geng duo', 'geng hao',
              'geng jia', 'gong an', 'gong bu', 'gong cheng', 'gong gong', 'gong he guo', 'gong kai', 'gong li',
              'gong min', 'gong shi',
              'gong si', 'gong tong', 'gong xian', 'gong xiang', 'gong ye', 'gong you', 'gong zuo', 'gou tong',
              'guan jian', 'guan li',
              'guan xi', 'guan xin', 'guan yu', 'guan zhong', 'guan zhu', 'guang dong', 'guang fan', 'guang xi',
              'gui ding', 'gui fan',
              'gui zhou', 'guo', 'guo cheng', 'guo fang bu', 'guo ji', 'guo jia', 'guo lai', 'guo min dang', 'guo nei',
              'guo qu',
              'guo wu yuan', 'ha', 'hai', 'hai shang', 'hai shi', 'hai wai', 'hai you', 'hai zi', 'han', 'han guo',
              'hao', 'he',
              'he bei', 'he nan', 'he ping', 'he xin', 'he zuo', 'hen', 'hen duo', 'hou', 'hu', 'hu bei',
              'hu lian wang', 'hu nan',
              'hu xin', 'hua', 'hua bei', 'hua ti', 'huan jing', 'huan ying', 'huang', 'hui', 'hui dao', 'hui gui',
              'hui jian',
              'hui shang', 'hui tan', 'hui wu', 'hui yi', 'hui ying', 'huo ban', 'huo bi', 'huo de', 'huo dong',
              'huo li', 'huo zai',
              'huo zhe', 'ji', 'ji ben', 'ji chang', 'ji chu', 'ji di', 'ji duan', 'ji gou', 'ji guan', 'ji hua',
              'ji ji', 'ji jiang',
              'ji lu', 'ji shi', 'ji shu', 'ji tuan', 'ji xu', 'ji yu', 'ji zhe', 'ji zhi', 'ji zhong', 'jia',
              'jia bin', 'jia ge',
              'jia qiang', 'jian', 'jian chi', 'jian ding', 'jian guan', 'jian jue', 'jian kang', 'jian li', 'jian she',
              'jiang',
              'jiang hua', 'jiang hui', 'jiang su', 'jiang xi', 'jiang yu', 'jiao', 'jiao liu', 'jiao tong', 'jiao yu',
              'jie',
              'jie duan', 'jie guo', 'jie jue', 'jie mu', 'jie shao', 'jie shou', 'jie shu', 'jie xia lai', 'jie zhi',
              'jin',
              'jin nian', 'jin nian lai', 'jin qi', 'jin ri', 'jin rong', 'jin ru', 'jin tian', 'jin xing', 'jin yi bu',
              'jin zhan',
              'jin zhuan', 'jing', 'jing fang', 'jing guo', 'jing ji', 'jing shen', 'jiu', 'jiu shi', 'jiu shi shuo',
              'ju', 'ju ban',
              'ju da', 'ju jiao', 'ju li', 'ju min', 'ju shi', 'ju ti', 'ju xing', 'ju you', 'jue de', 'jue ding',
              'jun', 'jun fang',
              'jun shi', 'ka', 'ka ta er', 'kai', 'kai fa', 'kai fang', 'kai mu', 'kai mu shi', 'kai qi', 'kai shi',
              'kai zhan', 'kan',
              'kan dao', 'kan kan', 'kao', 'ke', 'ke hu duan', 'ke ji', 'ke neng', 'ke xue', 'ke yi', 'kong jian',
              'kong zhi', 'kuai',
              'la', 'lai', 'lai shuo', 'lai zi', 'lan', 'lang', 'lao', 'le', 'lei', 'li', 'li ji', 'li liang',
              'li mian', 'li shi',
              'li yi', 'li yong', 'lian', 'lian bang', 'lian he', 'lian he guo', 'lian xi', 'lian xian', 'lian xu',
              'liang', 'liang an',
              'liang hao', 'liao jie', 'lin', 'ling chen', 'ling dao', 'ling dao ren', 'ling wai', 'ling yu', 'liu',
              'long', 'lou', 'lu',
              'lu xu', 'lun', 'lun tan', 'luo', 'luo shi', 'ma', 'mao yi', 'mei', 'mei guo', 'mei nian', 'mei ti',
              'mei you', 'men', 'meng',
              'meng xiang', 'mi', 'mi qie', 'mi shu zhang', 'mian dui', 'mian lin', 'min zhong', 'ming', 'ming que',
              'ming tian', 'ming xian',
              'mo', 'mu biao', 'mu qian', 'n', 'na', 'na ge', 'na me', 'nan bu', 'nan fang', 'ne', 'nei', 'nei rong',
              'neng', 'neng gou',
              'neng li', 'neng yuan', 'ni', 'nian', 'nian qing', 'nin', 'nu li', 'ou meng', 'ou zhou', 'peng you', 'pi',
              'pian', 'pin dao',
              'ping jia', 'ping tai', 'pu bian', 'pu jing', 'qi', 'qi che', 'qi dai', 'qi dong', 'qi jian', 'qi lai',
              'qi shi', 'qi ta',
              'qi wen', 'qi ye', 'qi zhong', 'qian', 'qian shu', 'qiang', 'qiang diao', 'qiang jiang yu', 'qiang lie',
              'qiao', 'qing',
              'qing kuang', 'qing zhu', 'qu', 'qu de', 'qu nian', 'qu xiao', 'qu yu', 'quan', 'quan bu', 'quan guo',
              'quan mian', 'quan qiu',
              'quan ti', 'que', 'que bao', 'que ding', 'que ren', 'ran hou', 'rang', 're', 'ren', 'ren he', 'ren lei',
              'ren min',
              'ren min bi', 'ren shi', 'ren shu', 'ren wei', 'ren wu', 'ren yuan', 'reng ran', 'ri', 'ri ben',
              'ri qian', 'rong he',
              'ru guo', 'ru he', 'san', 'san nian', 'san shi', 'san tian', 'sao miao', 'sen', 'sha te', 'shan', 'shang',
              'shang hai',
              'shang sheng', 'shang wang', 'shang wu', 'shang ye', 'shao', 'shao hou', 'she bei', 'she hui', 'she ji',
              'she shi', 'she xian',
              'shen', 'shen fen', 'shen hua', 'shen me', 'shen ru', 'shen zhi', 'sheng', 'sheng chan', 'sheng huo',
              'sheng ji', 'sheng ming',
              'shi', 'shi ba', 'shi bu shi', 'shi chang', 'shi dai', 'shi er', 'shi gu', 'shi hou', 'shi ji',
              'shi ji shang', 'shi jian',
              'shi jie', 'shi jiu', 'shi liu', 'shi pin', 'shi qi', 'shi san', 'shi shi', 'shi si', 'shi wei', 'shi wu',
              'shi xian',
              'shi yan', 'shi ye', 'shi yong', 'shi zhong', 'shou', 'shou ci', 'shou dao', 'shou du', 'shou kan',
              'shou shang', 'shou xian',
              'shou xiang', 'shu', 'shu ji', 'shu ju', 'shuang', 'shuang fang', 'shui', 'shui ping', 'shuo', 'shuo shi',
              'si', 'si chuan',
              'si shi', 'si wang', 'sou suo', 'sui', 'sui zhe', 'suo', 'suo wei', 'suo yi', 'suo you', 'suo zai', 'ta',
              'ta men', 'tai',
              'tai wan', 'tan suo', 'tao', 'tao lun', 'te', 'te bie', 'ti', 'ti chu', 'ti gao', 'ti gong', 'ti sheng',
              'ti shi', 'ti xian',
              'ti zhi', 'tian', 'tian qi', 'tian ran qi', 'tiao', 'tiao jian', 'tiao zhan', 'tiao zheng', 'tie lu',
              'ting', 'tong',
              'tong bao', 'tong guo', 'tong ji', 'tong shi', 'tong yi', 'tou piao', 'tou zi', 'tu po', 'tuan dui',
              'tuan jie', 'tui chu',
              'tui dong', 'tui jin', 'wai', 'wai jiao', 'wai jiao bu', 'wai zhang', 'wan', 'wan cheng', 'wan quan',
              'wan shang', 'wang',
              'wang zhan', 'wei', 'wei fa', 'wei fan', 'wei hu', 'wei lai', 'wei le', 'wei sheng', 'wei xian',
              'wei xie', 'wei yu',
              'wei yuan', 'wei yuan hui', 'wei zhi', 'wen', 'wen ding', 'wen hua', 'wen ming', 'wen ti', 'wo', 'wo guo',
              'wo men', 'wu',
              'wu ren ji', 'wu shi', 'xi', 'xi bu', 'xi huan', 'xi ji', 'xi jin ping', 'xi lie', 'xi tong', 'xi wang',
              'xia', 'xia mian',
              'xia wu', 'xia zai', 'xian', 'xian chang', 'xian jin', 'xian sheng', 'xian shi', 'xian yi ren',
              'xian zai', 'xiang',
              'xiang gang', 'xiang guan', 'xiang mu', 'xiang xi', 'xiang xin', 'xiao', 'xiao shi', 'xiao xi',
              'xie shang', 'xie tiao',
              'xie yi', 'xin', 'xin wen', 'xin wen lian bo', 'xin xi', 'xin xin', 'xin xing', 'xin yi lun', 'xing',
              'xing cheng',
              'xing dong', 'xing shi', 'xing wei', 'xing zheng', 'xu li ya', 'xu yao', 'xuan bu', 'xuan ju', 'xuan ze',
              'xue', 'ya',
              'yan fa', 'yan ge', 'yan jiu', 'yan zhong', 'yang shi', 'yao', 'yao qing', 'yao qiu', 'ye', 'yi', 'yi ci',
              'yi dao',
              'yi dian', 'yi ding', 'yi dong', 'yi fa', 'yi ge', 'yi hou', 'yi hui', 'yi ji', 'yi jian', 'yi jing',
              'yi kuai', 'yi lai',
              'yi liao', 'yi lu', 'yi qi', 'yi qie', 'yi shang', 'yi shi', 'yi si lan', 'yi ti', 'yi wai', 'yi wei zhe',
              'yi xi lie',
              'yi xia', 'yi xie', 'yi yang', 'yi yi', 'yi yuan', 'yi zhi', 'yi zhong', 'yin', 'yin du', 'yin fa',
              'yin hang', 'yin qi',
              'yin wei', 'ying', 'ying dui', 'ying gai', 'ying guo', 'ying ji', 'ying lai', 'ying xiang', 'yong',
              'yong you', 'you',
              'you de', 'you guan', 'you hao', 'you qi', 'you shi', 'you suo', 'you xiao', 'you yi', 'you yu', 'yu',
              'yu hui', 'yu ji',
              'yu jing', 'yu yi', 'yuan', 'yuan yi', 'yuan yin', 'yuan ze', 'yue', 'yue lai yue', 'yun ying', 'zai',
              'zai ci', 'zai hai',
              'zai jian', 'zan men', 'zao', 'zao cheng', 'zao yu', 'ze ren', 'zen me', 'zen me yang', 'zeng jia',
              'zhan', 'zhan kai',
              'zhan lve', 'zhan shi', 'zhan zai', 'zhang', 'zhao dao', 'zhao kai', 'zhe', 'zhe ge', 'zhe jiang',
              'zhe li', 'zhe me',
              'zhe ming', 'zhe yang', 'zhe zhong', 'zhei xie', 'zhen de', 'zhen dui', 'zhen zheng', 'zheng', 'zheng ce',
              'zheng chang',
              'zheng fu', 'zheng ge', 'zheng shi', 'zheng zai', 'zheng zhi', 'zhi', 'zhi bo', 'zhi chi', 'zhi chu',
              'zhi dao', 'zhi du',
              'zhi hou', 'zhi jian', 'zhi jie', 'zhi neng', 'zhi qian', 'zhi shao', 'zhi shi', 'zhi wai', 'zhi xia',
              'zhi xing', 'zhi you',
              'zhi zao', 'zhong', 'zhong bu', 'zhong da', 'zhong dian', 'zhong e', 'zhong fang',
              'zhong gong zhong yang',
              'zhong gong zhong yang zheng zhi ju', 'zhong guo', 'zhong hua min zu', 'zhong shi', 'zhong wu',
              'zhong xin', 'zhong yang',
              'zhong yang qi xiang tai', 'zhong yao', 'zhou', 'zhou nian', 'zhu', 'zhu he', 'zhu quan', 'zhu ti',
              'zhu xi', 'zhu yao',
              'zhu yi', 'zhua', 'zhuan', 'zhuan ji', 'zhuan jia', 'zhuan xiang', 'zhuan ye', 'zhuang tai', 'zhun bei',
              'zi', 'zi ben',
              'zi ji', 'zi jin', 'zi xun', 'zi you', 'zi yuan', 'zi zhu', 'zong', 'zong he', 'zong li', 'zong shu ji',
              'zong tong', 'zou',
              'zu', 'zu guo', 'zu zhi', 'zui', 'zui gao', 'zui hou', 'zui jin', 'zui xin', 'zui zhong', 'zun zhong',
              'zuo', 'zuo chu',
              'zuo dao', 'zuo hao', 'zuo tian', 'zuo wei', 'zuo yong', 'zuo you']

# '''方法2'''
# 单韵母:a 0 o 1 e 2 i 3 u 4 v 5
# 复韵母:ai 6 ei 7 ui 8 ao 9 ou 10 iu 11 ie 12 ve 13 er 14
# 前鼻韵母:an 15 en 16 in 17 un 18 vn 19
# 后鼻韵母:ang 20 eng 21 ing 22 ong 23
# 声母:b 24 p 25 m 26 f 27 d 28 t 29 n 30 l 31 g 32 k 33 h 34 j 35 q 36 x 37
#       zh 38 ch 39 sh 40 r 41 z 42 c 43 s 44 y 45 w 46
# C 47 空 48
PhonemeList = {'C': [47], 'a': [0], 'ai': [6], 'an': [15], 'ang': [20], 'ao': [9], 'ba': [24, 0], 'bai': [24, 6],
               'ban': [24, 15], 'bang': [24, 20], 'bao': [24, 9], 'bei': [24, 7], 'ben': [24, 16], 'beng': [24, 21],
               'bi': [24, 3], 'bian': [24, 3, 15], 'biao': [24, 3, 9], 'bie': [24, 12], 'bin': [24, 17],
               'bing': [24, 22], 'bo': [24, 1], 'bu': [24, 4], 'ca': [43, 0], 'cai': [43, 6], 'can': [43, 15],
               'cang': [43, 20], 'cao': [43, 9], 'ce': [43, 2], 'cen': [43, 16], 'ceng': [43, 21], 'cha': [39, 0],
               'chai': [39, 6], 'chan': [39, 15], 'chang': [39, 20], 'chao': [39, 9], 'che': [39, 2], 'chen': [39, 16],
               'cheng': [39, 21], 'chi': [39], 'chong': [39, 23], 'chou': [39, 10], 'chu': [39, 4], 'chuai': [39, 4, 6],
               'chuan': [39, 4, 15], 'chuang': [39, 4, 20], 'chui': [39, 8], 'chun': [39, 18], 'chuo': [39, 4, 1],
               'ci': [43], 'cong': [43, 23], 'cou': [43, 10], 'cu': [43, 4], 'cuan': [43, 4, 15], 'cui': [43, 8],
               'cun': [43, 18], 'cuo': [43, 4, 1], 'da': [28, 0], 'dai': [28, 6], 'dan': [28, 15], 'dang': [28, 20],
               'dao': [28, 9], 'de': [28, 2], 'deng': [28, 21], 'di': [28, 3], 'dian': [28, 3, 15], 'diao': [28, 3, 9],
               'die': [28, 12], 'ding': [28, 22], 'diu': [28, 11], 'dong': [28, 23], 'dou': [28, 10], 'du': [28, 4],
               'duan': [28, 4, 15], 'dui': [28, 4, 7], 'dun': [28, 18], 'duo': [28, 4, 1], 'e': [2], 'en': [16],
               'er': [14], 'fa': [27, 0], 'fan': [27, 15], 'fang': [27, 20], 'fei': [27, 7], 'fen': [27, 16],
               'feng': [27, 21], 'fo': [27, 1], 'fou': [27, 10], 'fu': [27, 4], 'ga': [32, 0], 'gai': [32, 6],
               'gan': [32, 15], 'gang': [32, 20], 'gao': [32, 9], 'ge': [32, 2], 'gei': [32, 7], 'gen': [32, 16],
               'geng': [32, 21], 'gong': [32, 23], 'gou': [32, 10], 'gu': [32, 4], 'gua': [32, 4, 0],
               'guai': [32, 4, 6], 'guan': [32, 4, 15], 'guang': [32, 4, 20], 'gui': [32, 4, 7], 'gun': [32, 18],
               'guo': [32, 4, 1], 'ha': [34, 0], 'hai': [34, 6], 'han': [34, 15], 'hang': [34, 20], 'hao': [34, 9],
               'he': [34, 2], 'hei': [34, 7], 'hen': [34, 16], 'heng': [34, 21], 'hong': [34, 23], 'hou': [34, 10],
               'hu': [34, 4], 'hua': [34, 4, 0], 'huai': [34, 4, 6], 'huan': [34, 4, 15], 'huang': [34, 4, 20],
               'hui': [34, 4, 7], 'hun': [34, 18], 'huo': [34, 4, 1], 'ji': [35, 3], 'jia': [35, 3, 0],
               'jian': [35, 3, 15], 'jiang': [35, 3, 20], 'jiao': [35, 3, 9], 'jie': [35, 12], 'jin': [35, 17],
               'jing': [35, 22], 'jiu': [35, 11], 'ju': [35, 5], 'juan': [35, 5, 15], 'jue': [35, 13], 'jun': [35, 19],
               'ka': [33, 0], 'kai': [33, 6], 'kan': [33, 15], 'kang': [33, 20], 'kao': [33, 9], 'ke': [33, 2],
               'ken': [33, 16], 'keng': [33, 21], 'kong': [33, 23], 'kou': [33, 10], 'ku': [33, 4], 'kua': [33, 4, 0],
               'kuai': [33, 4, 6], 'kuan': [33, 4, 15], 'kuang': [33, 4, 20], 'kui': [33, 8], 'kun': [33, 18],
               'kuo': [33, 4, 1], 'la': [31, 0], 'lai': [31, 6], 'lan': [31, 15], 'lang': [31, 20], 'lao': [31, 9],
               'le': [31, 2], 'lei': [31, 7], 'leng': [31, 21], 'li': [31, 3], 'lia': [31, 3, 0], 'lian': [31, 3, 15],
               'liang': [31, 3, 20], 'liao': [31, 3, 9], 'lie': [31, 12], 'lin': [31, 17], 'ling': [31, 22],
               'liu': [31, 11], 'long': [31, 23], 'lou': [31, 10], 'lu': [31, 4], 'luan': [31, 4, 15], 'lun': [31, 18],
               'luo': [31, 4, 1], 'lv': [31, 5], 'lve': [31, 13], 'ma': [26, 0], 'mai': [26, 6], 'man': [26, 15],
               'mang': [26, 20], 'mao': [26, 9], 'me': [26, 2], 'mei': [26, 7], 'men': [26, 16], 'meng': [26, 21],
               'mi': [26, 3], 'mian': [26, 3, 15], 'miao': [26, 3, 9], 'mie': [26, 12], 'min': [26, 17],
               'ming': [26, 22], 'miu': [26, 11], 'mo': [26, 1], 'mou': [26, 10], 'mu': [26, 4], 'na': [30, 0],
               'nai': [30, 6], 'nan': [30, 15], 'nang': [30, 20], 'nao': [30, 9], 'ne': [30, 2], 'nei': [30, 7],
               'nen': [30, 16], 'neng': [30, 21], 'ni': [30, 3], 'nian': [30, 3, 15], 'niang': [30, 3, 20],
               'niao': [30, 3, 9], 'nie': [30, 12], 'nin': [30, 17], 'ning': [30, 22], 'niu': [30, 11],
               'nong': [30, 23], 'nu': [30, 4], 'nuan': [30, 4, 15], 'nue': [30, 13], 'nuo': [30, 4, 1], 'nv': [30, 5],
               'o': [2], 'ou': [10], 'pa': [25, 0], 'pai': [25, 6], 'pan': [25, 15], 'pang': [25, 20], 'pao': [25, 9],
               'pei': [25, 7], 'pen': [25, 16], 'peng': [25, 21], 'pi': [25, 3], 'pian': [25, 3, 15],
               'piao': [25, 3, 9], 'pin': [25, 17], 'ping': [25, 22], 'po': [25, 1], 'pou': [25, 10], 'pu': [25, 4],
               'qi': [36, 3], 'qia': [36, 3, 0], 'qian': [36, 3, 15], 'qiang': [36, 3, 20], 'qiao': [36, 3, 9],
               'qie': [36, 12], 'qin': [36, 17], 'qing': [36, 22], 'qiong': [36, 3, 23], 'qiu': [36, 11], 'qu': [36, 5],
               'quan': [36, 5, 15], 'que': [36, 13], 'qun': [36, 18], 'ran': [41, 15], 'rang': [41, 20], 'rao': [41, 9],
               're': [41, 2], 'ren': [41, 16], 'reng': [41, 21], 'ri': [41], 'rong': [41, 23], 'rou': [41, 10],
               'ru': [41, 4], 'ruan': [41, 4, 15], 'rui': [41, 8], 'run': [41, 18], 'ruo': [41, 4, 1], 'sa': [44, 0],
               'sai': [44, 6], 'san': [44, 15], 'sang': [44, 20], 'sao': [44, 9], 'se': [44, 2], 'sen': [44, 16],
               'seng': [44, 21], 'sha': [40, 0], 'shai': [40, 6], 'shan': [40, 15], 'shang': [40, 20], 'shao': [40, 9],
               'she': [40, 2], 'shen': [40, 16], 'sheng': [40, 21], 'shi': [40], 'shou': [40, 10], 'shu': [40, 4],
               'shua': [40, 4, 0], 'shuai': [40, 4, 6], 'shuan': [40, 4, 15], 'shuang': [40, 4, 20], 'shui': [40, 4, 7],
               'shun': [40, 18], 'shuo': [40, 4, 1], 'si': [44], 'song': [44, 23], 'sou': [44, 10], 'su': [44, 4],
               'suan': [44, 4, 15], 'sui': [44, 4, 7], 'sun': [44, 18], 'suo': [44, 4, 1], 'ta': [29, 0],
               'tai': [29, 6], 'tan': [29, 15], 'tang': [29, 20], 'tao': [29, 9], 'te': [29, 2], 'teng': [29, 21],
               'ti': [29, 3], 'tian': [29, 3, 15], 'tiao': [29, 3, 9], 'tie': [29, 12], 'ting': [29, 22],
               'tong': [29, 23], 'tou': [29, 10], 'tu': [29, 4], 'tuan': [29, 4, 15], 'tui': [29, 4, 7],
               'tun': [29, 18], 'tuo': [29, 4, 1], 'wa': [46, 0], 'wai': [46, 6], 'wan': [46, 15], 'wang': [46, 20],
               'wei': [46, 7], 'wen': [46, 16], 'weng': [46, 21], 'wo': [46, 1], 'wu': [46, 4], 'xi': [37, 3],
               'xia': [37, 3, 0], 'xian': [37, 3, 15], 'xiang': [37, 3, 20], 'xiao': [37, 3, 9], 'xie': [37, 12],
               'xin': [37, 17], 'xing': [37, 22], 'xiong': [37, 3, 23], 'xiu': [37, 11], 'xu': [37, 5],
               'xuan': [37, 5, 15], 'xue': [37, 13], 'xun': [37, 19], 'ya': [45, 0], 'yan': [45, 15], 'yang': [45, 20],
               'yao': [45, 9], 'ye': [12], 'yi': [45], 'yin': [45, 17], 'ying': [45, 22], 'yong': [45, 23],
               'you': [45, 10], 'yu': [5], 'yuan': [5, 15], 'yue': [13], 'yun': [45, 19], 'za': [42, 0], 'zai': [42, 6],
               'zan': [42, 15], 'zang': [42, 20], 'zao': [42, 9], 'ze': [42, 2], 'zei': [42, 7], 'zen': [42, 16],
               'zeng': [42, 21], 'zha': [38, 0], 'zhai': [38, 6], 'zhan': [38, 15], 'zhang': [38, 20], 'zhao': [38, 9],
               'zhe': [38, 2], 'zhen': [38, 16], 'zheng': [38, 21], 'zhi': [38], 'zhong': [38, 23], 'zhou': [38, 10],
               'zhu': [38, 4], 'zhua': [38, 4, 0], 'zhuan': [38, 4, 15], 'zhuang': [38, 4, 20], 'zhui': [38, 8],
               'zhun': [38, 18], 'zhuo': [38, 4, 1], 'zi': [42], 'zong': [42, 23], 'zou': [42, 10], 'zu': [42, 4],
               'zuan': [42, 4, 15], 'zui': [42, 8], 'zun': [42, 18], 'zuo': [42, 4, 1], 'zhei': [38, 7], 'n': [16],
               }

# '''方法3'''
# 单韵母:a 0 o 1 e 2 i 3 u 4 v 5
# 复韵母:ai 6 ei 7 ui 8 ao 9 ou 10 iu 11 ie 12 ve 13 er 14
# 前鼻韵母:an 15 en 16 in 17 un 18 vn 19
# 后鼻韵母:ang 20 eng 21 ing 22 ong 23
# ch 25 sh 26 zh 27 c 28 s 29 z 30 r 31 空 32
# C 24
# PhonemeList = {'C': [24], 'a': [0], 'ai': [6], 'ji': [3], 'an': [15], 'jian': [3, 15],
#                'quan': [5, 15], 'zhao': [27, 9], 'ba': [0], 'li': [3], 'xi': [3],
#                'bai': [6], 'ban': [15], 'dao': [9], 'fa': [0], 'bang': [20],
#                'jia': [3, 0], 'bao': [9], 'chi': [25], 'gao': [9], 'hu': [4],
#                'kuo': [4, 1], 'yu': [5], 'zhang': [20], 'bei': [7], 'bu': [4],
#                'jing': [22], 'shi': [26], 'yue': [13], 'ben': [16], 'ci': [28],
#                'bi': [3], 'jiao': [3, 9], 'ru': [31, 4], 'xu': [5], 'bian': [3, 15],
#                'hua': [4, 0], 'biao': [3, 9], 'da': [0], 'zhi': [27], 'zhun': [27, 18],
#                'bie': [12], 'bing': [22], 'qie': [12], 'bo': [1], 'chu': [25, 4],
#                'duan': [4, 15], 'fen': [16], 'guo': [4, 1], 'hui': [4, 7], 'jin': [17],
#                'men': [16], 'neng': [21], 'shao': [26, 9], 'shu': [26, 4], 'tong': [23],
#                'yao': [9], 'cai': [28, 6], 'fang': [20], 'qu': [5], 'can': [28, 15],
#                'ce': [28, 2], 'ceng': [28, 21], 'chan': [25, 15], 'pin': [17], 'sheng': [26, 21],
#                'ye': [12], 'chang': [25, 20], 'qi': [3], 'yi': [3], 'chao': [25, 9],
#                'xian': [3, 15], 'che': [25, 2], 'cheng': [25, 21], 'gong': [23], 'nuo': [4, 1],
#                'wei': [7], 'lai': [6], 'le': [2], 'chuan': [25, 4, 15], 'chuang': [25, 4, 20],
#                'xin': [17], 'chun': [25, 18], 'qian': [3, 15], 'cong': [28, 23], 'cu': [28, 4],
#                'cun': [28, 18], 'zai': [30, 6], 'cuo': [28, 4, 1], 'gai': [6], 'xing': [22],
#                'xue': [13], 'zao': [30, 9], 'dai': [6], 'dan': [15], 'dang': [20],
#                'di': [3], 'tian': [3, 15], 'zhong': [27, 23], 'de': [2], 'deng': [21],
#                'dian': [3, 15], 'diao': [3, 9], 'cha': [25, 0], 'yan': [15], 'dong': [23],
#                'dou': [10], 'du': [4], 'dui': [4, 7], 'wai': [6], 'duo': [4, 1],
#                'nian': [3, 15], 'e': [2], 'luo': [4, 1], 'si': [29], 'er': [14],
#                'ling': [22], 'liu': [11], 'san': [29, 15], 'wu': [4], 'ma': [0],
#                'she': [26, 2], 'ren': [31, 16], 'yuan': [5, 15], 'zhan': [27, 15], 'fan': [15],
#                'rong': [31, 23], 'zui': [30, 4, 7], 'mian': [3, 15], 'wen': [16],
#                'xiang': [3, 20], 'fei': [7], 'zi': [30], 'feng': [21], 'shuo': [26, 4, 1],
#                'fu': [4], 'ze': [30, 2], 'ge': [2], 'shan': [26, 15], 'gan': [15],
#                'jue': [13], 'shou': [26, 10], 'xie': [12], 'gang': [20], 'xiao': [3, 9],
#                'jie': [12], 'gei': [7], 'gen': [16], 'ju': [5], 'geng': [21],
#                'hao': [9], 'he': [2], 'kai': [6], 'min': [17], 'you': [10],
#                'zuo': [30, 4, 1], 'gou': [0], 'guan': [4, 15], 'zhu': [27, 4],
#                'guang': [4, 20], 'gui': [4, 7], 'ding': [22], 'zhou': [27, 10], 'nei': [7],
#                'ha': [0], 'hai': [6], 'shang': [26, 20], 'han': [15], 'nan': [15],
#                'ping': [22], 'hen': [16], 'hou': [10], 'lian': [3, 15], 'wang': [20],
#                'ti': [3], 'huan': [4, 15], 'ying': [22], 'huang': [4, 20], 'tan': [15],
#                'huo': [4, 1], 'zhe': [27, 2], 'jiang': [3, 20], 'lu': [4], 'tuan': [4, 15],
#                'bin': [17], 'qiang': [3, 20], 'kang': [20], 'su': [29, 4], 'mu': [4],
#                'xia': [3, 0], 'ri': [31], 'zhuan': [27, 4, 15], 'shen': [26, 16], 'jiu': [11],
#                'jun': [19], 'ka': [0], 'ta': [0], 'kan': [15], 'kao': [9],
#                'ke': [2], 'kong': [23], 'kuai': [4, 6], 'la': [0], 'lan': [15],
#                'lang': [20], 'lao': [9], 'lei': [7], 'liang': [3, 20], 'yong': [23],
#                'liao': [3, 9], 'lin': [17], 'chen': [25, 16], 'long': [23], 'lou': [10],
#                'lun': [18], 'mao': [9], 'mei': [7], 'meng': [21], 'mi': [3],
#                'ming': [22], 'que': [13], 'mo': [1], 'n': [16], 'na': [0], 'me': [2],
#                'ne': [2], 'ni': [3], 'qing': [22], 'nin': [17], 'nu': [4], 'ou': [10],
#                'peng': [21], 'pi': [3], 'pian': [3, 15], 'tai': [6], 'pu': [4],
#                'lie': [12], 'qiao': [3, 9], 'kuang': [4, 20], 'qiu': [11], 'ran': [31, 15],
#                'rang': [31, 20], 're': [31, 2], 'reng': [31, 21], 'sao': [29, 9], 'miao': [3, 9],
#                'sen': [29, 16], 'sha': [26, 0], 'te': [2], 'gu': [4], 'shuang': [26, 4, 20],
#                'shui': [26, 4, 7], 'sou': [29, 10], 'suo': [29, 4, 1], 'sui': [29, 4, 7], 'wan': [15],
#                'tao': [9], 'tiao': [3, 9], 'zheng': [27, 21], 'tie': [12], 'ting': [22],
#                'tou': [10], 'piao': [3, 9], 'tu': [4], 'po': [1], 'tui': [4, 7],
#                'wo': [1], 'ya': [0], 'xuan': [5, 15], 'yang': [20], 'yin': [17],
#                'hang': [20], 'yun': [19], 'zan': [30, 15], 'zen': [30, 16], 'zeng': [30, 21],
#                'lve': [13], 'zhei': [27, 2], 'zhen': [27, 16], 'zu': [30, 4], 'zhua': [27, 4, 0],
#                'zhuang': [27, 4, 20], 'xun': [19], 'zong': [30, 23], 'zou': [30, 10], 'zun': [30, 18]}


class LRW1000_Dataset(Dataset):

    def __init__(self, phase, args):

        self.args = args
        self.data = []
        self.phase = phase

        if (self.phase == 'train'):
            # local
            # self.index_root = 'E:/LRW1000_Public_pkl_jpeg/trn'
            # 3080
            self.index_root = '/home/mingwu/workspace_czg/data/LRW/LRW1000_Public_pkl_jpeg/trn'
            # 3090
            # self.index_root = '/home/czg/dataset/LRW1000_Public_pkl_jpeg/trn'
            # self.index_root = '/home/czg/dataset/LRW1000_Phome/trn'
        # elif (self.phase == 'val'):
        #     self.index_root = '/home/czg/dataset/LRW1000_Public_pkl_jpeg/val'
        else:
            # local
            # self.index_root = 'E:/LRW1000_Public_pkl_jpeg/trn'
            # 3080
            self.index_root = '/home/mingwu/workspace_czg/data/LRW/LRW1000_Public_pkl_jpeg/tst'
            # 3090
            # self.index_root = '/home/czg/dataset/LRW1000_Public_pkl_jpeg/tst'
            # self.index_root = '/home/czg/dataset/LRW1000_Phome/tst'

        self.data = glob.glob(os.path.join(self.index_root, '*.pkl'))

        self.PhonemeList = ['C', 'a', 'ai', 'ji', 'an', 'jian',
                            'quan', 'zhao', 'ba', 'li', 'xi',
                            'bai', 'ban', 'dao', 'fa', 'bang',
                            'jia', 'bao', 'chi', 'gao', 'hu',
                            'kuo', 'yu', 'zhang', 'bei', 'bu',
                            'jing', 'shi', 'yue', 'ben', 'ci',
                            'bi', 'jiao', 'ru', 'xu', 'bian',
                            'hua', 'biao', 'da', 'zhi', 'zhun',
                            'bie', 'bing', 'qie', 'bo', 'chu',
                            'duan', 'fen', 'guo', 'hui', 'jin',
                            'men', 'neng', 'shao', 'shu', 'tong',
                            'yao', 'cai', 'fang', 'qu', 'can',
                            'ce', 'ceng', 'chan', 'pin', 'sheng',
                            'ye', 'chang', 'qi', 'yi', 'chao',
                            'xian', 'che', 'cheng', 'gong', 'nuo',
                            'wei', 'lai', 'le', 'chuan', 'chuang',
                            'xin', 'chun', 'qian', 'cong', 'cu',
                            'cun', 'zai', 'cuo', 'gai', 'xing',
                            'xue', 'zao', 'dai', 'dan', 'dang',
                            'di', 'tian', 'zhong', 'de', 'deng',
                            'dian', 'diao', 'cha', 'yan', 'dong',
                            'dou', 'du', 'dui', 'wai', 'duo',
                            'nian', 'e', 'luo', 'si', 'er',
                            'ling', 'liu', 'san', 'wu', 'ma',
                            'she', 'ren', 'yuan', 'zhan', 'fan',
                            'rong', 'zui', 'mian', 'wen',
                            'xiang', 'fei', 'zi', 'feng', 'shuo',
                            'fu', 'ze', 'ge', 'shan', 'gan',
                            'jue', 'shou', 'xie', 'gang', 'xiao',
                            'jie', 'gei', 'gen', 'ju', 'geng',
                            'hao', 'he', 'kai', 'min', 'you',
                            'zuo', 'gou', 'guan', 'zhu',
                            'guang', 'gui', 'ding', 'zhou', 'nei',
                            'ha', 'hai', 'shang', 'han', 'nan',
                            'ping', 'hen', 'hou', 'lian', 'wang',
                            'ti', 'huan', 'ying', 'huang', 'tan',
                            'huo', 'zhe', 'jiang', 'lu', 'tuan',
                            'bin', 'qiang', 'kang', 'su', 'mu',
                            'xia', 'ri', 'zhuan', 'shen', 'jiu',
                            'jun', 'ka', 'ta', 'kan', 'kao',
                            'ke', 'kong', 'kuai', 'la', 'lan',
                            'lang', 'lao', 'lei', 'liang', 'yong',
                            'liao', 'lin', 'chen', 'long', 'lou',
                            'lun', 'mao', 'mei', 'meng', 'mi',
                            'ming', 'que', 'mo', 'n', 'na', 'me',
                            'ne', 'ni', 'qing', 'nin', 'nu', 'ou',
                            'peng', 'pi', 'pian', 'tai', 'pu',
                            'lie', 'qiao', 'kuang', 'qiu', 'ran',
                            'rang', 're', 'reng', 'sao', 'miao',
                            'sen', 'sha', 'te', 'gu', 'shuang',
                            'shui', 'sou', 'suo', 'sui', 'wan',
                            'tao', 'tiao', 'zheng', 'tie', 'ting',
                            'tou', 'piao', 'tu', 'po', 'tui',
                            'wo', 'ya', 'xuan', 'yang', 'yin',
                            'hang', 'yun', 'zan', 'zen', 'zeng',
                            'lve', 'zhei', 'zhen', 'zu', 'zhua',
                            'zhuang', 'xun', 'zong', 'zou', 'zun']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        pkl = torch.load(self.data[idx])
        st = -1
        ed = 39
        for i, d in enumerate(pkl['duration']):
            if st == -1 and d == 1:
                st = i
            if st != -1 and d == 0:
                ed = i
                break

        video = pkl.get('video')
        video = [jpeg.decode(img, pixel_format=TJPF_GRAY) for img in video]
        video = np.stack(video, 0)
        # video[:st, :, :, :] = np.zeros((st, 96, 96, 1)).astype(video.dtype)
        # video[ed:, :, :, :] = np.zeros((40 - ed, 96, 96, 1)).astype(video.dtype)
        # video[0:ed - st, :, :, :] = video[st:ed, :, :, :]
        # video[st:ed, :, :, :] = np.zeros((ed - st, 96, 96, 1)).astype(video.dtype)
        video = video[:, :, :, 0]
        if (self.phase == 'train'):
            video = RandomCrop(video, (88, 88))
            video = HorizontalFlip(video)
        elif self.phase == 'val' or self.phase == 'test':
            video = CenterCrop(video, (88, 88))
        # video = gaussian_noise(video, 0, 0.1)
        # pkl['video'] = torch.FloatTensor(video)[:, None, ...] / 255.0
        videoTensor = torch.FloatTensor(video)[:, None, ...] / 255.0
        # for i in range(40):
        #     mean_video = torch.mean(videoTensor[i, :, :])
        #     std_video = torch.std(videoTensor[i, :, :])
        #     if mean_video != torch.zeros(1) and std_video != torch.zeros(1):
        #         videoTensor[i, :, :] -= mean_video
        #         videoTensor[i, :, :] /= std_video
        pkl['video'] = videoTensor

        label = pkl.get('label')
        pinyin = pinyinlist[int(label)]
        pinyinlable = np.full((40), 49).astype(pkl['pinyinlable'].dtype)
        t = 0
        for i in pinyin.split(' '):
            for j in PhonemeList[i]:
                pinyinlable[t] = j
                t += 1
            pinyinlable[t] = 48
            t += 1
        pkl['target_lengths'] = torch.tensor(t - 1).numpy()
        pkl['pinyinlable'] = pinyinlable
        pkl['tgt'] = torch.tensor(ed - st)

        # t = 0
        # for item in pkl['pinyinlable']:
        #     if item == 0:
        #         break
        #     t += 1
        # pkl['target_lengths'] = torch.tensor(t).numpy()
        #
        # pinyinlable = np.full((40), 28).astype(pkl['pinyinlable'].dtype)
        #
        # try:
        #     t = pkl['pinyinlable'].shape[0]
        #     pinyinlable[:t, ...] = pkl['pinyinlable'].copy()
        # except Exception as e:  # 可以写多个捕获异常
        #     print("ValueError")
        # pkl['pinyinlable'] = pinyinlable

        return pkl


if __name__ == '__main__':
    # local
    # target_dir = f'E:/LRW1000_Public_pkl_jpeg/trn'
    # index_file = f'E:/LRW/info/trn_1000.txt'
    # 3080
    target_dir = f'/home/mingwu/workspace_czg/data/LRW/LRW1000_Public_pkl_jpeg/trn'
    index_file = f'/home/mingwu/workspace_czg/data/LRW/LRW/info/trn_1000.txt'
    # 3090
    # target_dir = f'/home/czg/dataset/LRW1000_Public_pkl_jpeg/trn'
    # index_file = f'/home/czg/dataset/LRW/info/trn_1000.txt'

    dataset = LRW1000_Dataset('train', index_file)
    print('Start running, Data Length:', len(dataset))
    loader = DataLoader(dataset,
                        batch_size=1,
                        num_workers=1,
                        shuffle=False,
                        drop_last=False)

    for i_iter, input in enumerate(loader):
        video = input.get('video')
        # video = torch.FloatTensor(video)[:, None, ...] * 255.0
        video = torch.squeeze(video, dim=0)
        video = torch.squeeze(video, dim=0)
        label = input.get('label')
        for i in range(video.shape[0]):
            image = video[i, :, :, :]
            # image = image[0, :, :, :]
            # image = image.permute(2, 0, 1)
            # image = transforms.ToPILImage()(image)
            # image.show()
            #
            # im1 = image.filter(ImageFilter.SHARPEN)
            # im1.show()
            # # 亮度增强
            # enh_bri = ImageEnhance.Brightness(image)
            # brightness = 3
            # image_brightened = enh_bri.enhance(brightness)
            # image_brightened.show()
            # im0 = image_brightened.filter(ImageFilter.EDGE_ENHANCE)
            # im0.show()
            # # 色度增强(饱和度↑)
            # enh_col = ImageEnhance.Color(image)
            # color = 2
            # image_colored = enh_col.enhance(color)
            # image_colored.show()
            # # 对比度增强
            # enh_con = ImageEnhance.Contrast(image)
            # contrast = 5
            # image_contrasted = enh_con.enhance(contrast)
            # image_contrasted.show()
            # # 锐度增强
            # enh_sha = ImageEnhance.Sharpness(image)
            # sharpness = 4.0
            # image_sharped = enh_sha.enhance(sharpness)
            # image_sharped.show()
            # break
