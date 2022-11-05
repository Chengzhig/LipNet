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

# '''方法3'''
# 单韵母:a 0 o 1 e 2 i 3 u 4 v 5
# 复韵母:ai 6 ei 7 ui 8 ao 9 ou 10 iu 11 ie 12 ve 13 er 14
# 前鼻韵母:an 15 en 16 in 17 un 18 vn 19
# 后鼻韵母:ang 20 eng 21 ing 22 ong 23
# ch 25 sh 26 zh 27 c 28 s 29 z 30 r 31
# C 24
PhonemeList = {'C': [24], 'a': [0], 'ai': [6], 'ji': [3], 'an': [15], 'jian': [3, 15],
               'quan': [5, 15], 'zhao': [27, 9], 'ba': [0], 'li': [3], 'xi': [3],
               'bai': [6], 'ban': [15], 'dao': [9], 'fa': [0], 'bang': [20],
               'jia': [3, 0], 'bao': [9], 'chi': [25], 'gao': [9], 'hu': [4],
               'kuo': [4, 1], 'yu': [5], 'zhang': [20], 'bei': [7], 'bu': [4],
               'jing': [22], 'shi': [26], 'yue': [13], 'ben': [16], 'ci': [28],
               'bi': [3], 'jiao': [3, 9], 'ru': [31, 4], 'xu': [5], 'bian': [3, 15],
               'hua': [4, 0], 'biao': [3, 9], 'da': [0], 'zhi': [27], 'zhun': [27, 18],
               'bie': [12], 'bing': [22], 'qie': [12], 'bo': [1], 'chu': [25, 4],
               'duan': [4, 15], 'fen': [16], 'guo': [4, 1], 'hui': [4, 7], 'jin': [17],
               'men': [16], 'neng': [21], 'shao': [26, 9], 'shu': [26, 4], 'tong': [23],
               'yao': [9], 'cai': [28, 6], 'fang': [20], 'qu': [5], 'can': [28, 15],
               'ce': [28, 2], 'ceng': [28, 21], 'chan': [25, 15], 'pin': [17], 'sheng': [26, 21],
               'ye': [12], 'chang': [25, 20], 'qi': [3], 'yi': [3], 'chao': [25, 9],
               'xian': [3, 15], 'che': [25, 2], 'cheng': [25, 21], 'gong': [23], 'nuo': [4, 1],
               'wei': [7], 'lai': [6], 'le': [2], 'chuan': [25, 4, 15], 'chuang': [25, 4, 20],
               'xin': [17], 'chun': [25, 18], 'qian': [3, 15], 'cong': [28, 23], 'cu': [28, 4],
               'cun': [28, 18], 'zai': [30, 6], 'cuo': [28, 4, 1], 'gai': [6], 'xing': [22],
               'xue': [13], 'zao': [30, 9], 'dai': [6], 'dan': [15], 'dang': [20],
               'di': [3], 'tian': [3, 15], 'zhong': [27, 23], 'de': [2], 'deng': [21],
               'dian': [3, 15], 'diao': [3, 9], 'cha': [25, 0], 'yan': [15], 'dong': [23],
               'dou': [10], 'du': [4], 'dui': [4, 7], 'wai': [6], 'duo': [4, 1],
               'nian': [3, 15], 'e': [2], 'luo': [4, 1], 'si': [29], 'er': [14],
               'ling': [22], 'liu': [11], 'san': [29, 15], 'wu': [4], 'ma': [0],
               'she': [26, 2], 'ren': [31, 16], 'yuan': [5, 15], 'zhan': [27, 15], 'fan': [15],
               'rong': [31, 23], 'zui': [30, 4, 7], 'mian': [3, 15], 'wen': [16],
               'xiang': [3, 20], 'fei': [7], 'zi': [30], 'feng': [21], 'shuo': [26, 4, 1],
               'fu': [4], 'ze': [30, 2], 'ge': [2], 'shan': [26, 15], 'gan': [15],
               'jue': [13], 'shou': [26, 10], 'xie': [12], 'gang': [20], 'xiao': [3, 9],
               'jie': [12], 'gei': [7], 'gen': [16], 'ju': [5], 'geng': [21],
               'hao': [9], 'he': [2], 'kai': [6], 'min': [17], 'you': [10],
               'zuo': [30, 4, 1], 'gou': [0], 'guan': [4, 15], 'zhu': [27, 4],
               'guang': [4, 20], 'gui': [4, 7], 'ding': [22], 'zhou': [27, 10], 'nei': [7],
               'ha': [0], 'hai': [6], 'shang': [26, 20], 'han': [15], 'nan': [15],
               'ping': [22], 'hen': [16], 'hou': [10], 'lian': [3, 15], 'wang': [20],
               'ti': [3], 'huan': [4, 15], 'ying': [22], 'huang': [4, 20], 'tan': [15],
               'huo': [4, 1], 'zhe': [27, 2], 'jiang': [3, 20], 'lu': [4], 'tuan': [4, 15],
               'bin': [17], 'qiang': [3, 20], 'kang': [20], 'su': [29, 4], 'mu': [4],
               'xia': [3, 0], 'ri': [31], 'zhuan': [27, 4, 15], 'shen': [26, 16], 'jiu': [11],
               'jun': [19], 'ka': [0], 'ta': [0], 'kan': [15], 'kao': [9],
               'ke': [2], 'kong': [23], 'kuai': [4, 6], 'la': [0], 'lan': [15],
               'lang': [20], 'lao': [9], 'lei': [7], 'liang': [3, 20], 'yong': [23],
               'liao': [3, 9], 'lin': [17], 'chen': [25, 16], 'long': [23], 'lou': [10],
               'lun': [18], 'mao': [9], 'mei': [7], 'meng': [21], 'mi': [3],
               'ming': [22], 'que': [13], 'mo': [1], 'n': [16], 'na': [0], 'me': [2],
               'ne': [2], 'ni': [3], 'qing': [22], 'nin': [17], 'nu': [4], 'ou': [10],
               'peng': [21], 'pi': [3], 'pian': [3, 15], 'tai': [6], 'pu': [4],
               'lie': [12], 'qiao': [3, 9], 'kuang': [4, 20], 'qiu': [11], 'ran': [31, 15],
               'rang': [31, 20], 're': [31, 2], 'reng': [31, 21], 'sao': [29, 9], 'miao': [3, 9],
               'sen': [29, 16], 'sha': [26, 0], 'te': [2], 'gu': [4], 'shuang': [26, 4, 20],
               'shui': [26, 4, 7], 'sou': [29, 10], 'suo': [29, 4, 1], 'sui': [29, 4, 7], 'wan': [15],
               'tao': [9], 'tiao': [3, 9], 'zheng': [27, 21], 'tie': [12], 'ting': [22],
               'tou': [10], 'piao': [3, 9], 'tu': [4], 'po': [1], 'tui': [4, 7],
               'wo': [1], 'ya': [0], 'xuan': [5, 15], 'yang': [20], 'yin': [17],
               'hang': [20], 'yun': [19], 'zan': [30, 15], 'zen': [30, 16], 'zeng': [30, 21],
               'lve': [13], 'zhei': [27, 2], 'zhen': [27, 16], 'zu': [30, 4], 'zhua': [27, 4, 0],
               'zhuang': [27, 4, 20], 'xun': [19], 'zong': [30, 23], 'zou': [30, 10], 'zun': [30, 18]}


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
        video = video[:, :, :, 0]
        if (self.phase == 'train'):
            video = RandomCrop(video, (88, 88))
            video = HorizontalFlip(video)
        elif self.phase == 'val' or self.phase == 'test':
            video = CenterCrop(video, (88, 88))
        video = gaussian_noise(video, 0, 0.1)
        pkl['video'] = torch.FloatTensor(video)[:, None, ...] / 255.0

        label = pkl.get('label')
        pinyin = pinyinlist[int(label)]
        pinyinlable = np.full((40), 32).astype(pkl['pinyinlable'].dtype)
        t = 0
        for i in pinyin.split(' '):
            for j in PhonemeList[i]:
                pinyinlable[t] = j
                t += 1
        pkl['target_lengths'] = torch.tensor(t).numpy()
        pkl['pinyinlable'] = pinyinlable
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
