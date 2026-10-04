import jieba
import numpy as np

#windows必须从训练开始到推理一直是个定值
#函数部分的token相关的变量全部为向量或者向量列表(词嵌入，没有词),与att相关的全部是自然数或自然数列表！！！
#tokenlist=tokenwindows
#默认在一个token中最后一维为出现训练过程中的出现频次
#embedded_table即嵌入表，在此算法中指ka或kb
#words_attlist_words即一个完整text的att列表
#没有要求长记忆力,或者提升提取关键信息能力的情况下其实我觉得aattlist足够充当words_attlist_words了，理论上对注意力使用b_attlist迭代次数越多越好吧

def randtoken(dim):
    return np.concatenate([np.random.randn(dim),[1.0]])

def windows_support_token(word,embedded_table,chance,g,windows,textwindows,stability,dim):#给出嵌入表，需要更改的token，windows就可以得到更改后的token
    try:
        for k in textwindows:
            if k not in embedded_table:
                embedded_table[k]=np.concatenate([np.random.rand(dim),[1.0]])
        if word not in embedded_table:
            embedded_table[word]=g([embedded_table[k] for k in textwindows],windows)
        if word in embedded_table:
            embedded_table[word]=chance(embedded_table[word],g([embedded_table[k] for k in textwindows],windows),stability)
    except:
        embedded_table[word]=np.concatenate([np.random.rand(dim),[1.0]])
    return(embedded_table[word])

def embedded_table_support_attlist(base,words_attlist_words,embedded_table,words,ka,windows,dim):#没有要求长记忆力,或者提升提取关键信息能力的情况下其实我觉得aattlist足够充当words_attlist_words了
    prontoken={}
    b_attlist=[]
    for i,att in enumerate(words_attlist_words):
        if words[i] not in embedded_table:
            embedded_table[words[i]]=ka[words[i]]
        if i<windows:
            textwindows=words[i+1:windows-1]
        if i>=windows:
            textwindows=words[i-windows:i]
        prontoken[words[i]]=windows_context_suport_token(words[i],embedded_table,ka,words,words_attlist_words,windows,textwindows,dim,base)
        b_attlist.append(cosdis(prontoken[words[i]],embedded_table[words[i]]))
    return(b_attlist)
        

def windows_context_suport_token(word,embedded_table_b,embedded_table_a,words,words_attlist_words,windows,textwindows,dim,base,ka):
    for i,word_words in enumerate(words):
        if words[i] not in embedded_table_b:
            embedded_table_b[words[i]]=embedded_table_a[words[i]]
    b=np.zeros(np.shape(embedded_table_b[words[0]]))
    try:
        a=g([embedded_table_a[e] for e in textwindows],windows)
        for i,att in enumerate(words_attlist_words):
            if words_attlist_words[i]>base:
                b=embedded_table_b[words[i]]*att+b
        word=np.concatenate((a[:-1]+b[:-1]),embedded_table_b[-1]) #同样是窗口与上文注意力token结合
        return(word)
    except:
        print("'windows_support_token'发生错误，检测是否因kb(word)=None导致")
        if words[i] not in embedded_table_b:
            embedded_table_b[words[i]]=ka[words[i]]

def chance(oldtoken,newtoken,stability):
    if oldtoken[-1]<stability:
        endtoken=(oldtoken[-1]*oldtoken[:-1]+newtoken[:-1])/oldtoken[-1]
    else:
        endtoken=(oldtoken[:-1]*stability+newtoken[:-1])/(stability+1)
    endtoken=np.concatenate(endtoken,[oldtoken[-1]+1])
    return(endtoken)

def cosdis(token,tokenpr):
    token=token[:-1]
    tokenpr=tokenpr[:-1]
    dis=1-np.dot(token,tokenpr)/(np.linalg.norm(token)*np.linalg.norm(tokenpr))
    return(dis)

def g(tokenlist,windows):
    tl=tokenlist;w=windows
    end=zeros=np.zeros(np.shape(tl[0]))
    for a in tl:
        end=a+end
    return end

def h(tokenlist,windows,tokenlist_needatt,base,att_list,g):#tokenlist 窗口内上下文token 、tokenlist_needatt 需要注意的token， att_list att列表
    tl=tokenlist;w=windows;b=base;tn=tokenlist_needatt;al=att_list
    end=np.zeros(np.shape(tl[0]))
    for i,att in enumerate(att_list):
        if att>=b:
            tokenatt=tn[i]*att
            end=tokenatt+end
    e=end+g(tl,w)#窗口与上文注意力token结合
    return(e)

def a_attlist(text_tokenlist,g,windows):
    aattlist=[0.0]*len(text_tokenlist)
    for a,token in enumerate(text_tokenlist):
        if a<windows:
            continue
        tokenlist=text_tokenlist[a-windows:a]
        att=cosdis(g(tokenlist,windows),token)
        aattlist[a]=att
    for i,a in enumerate(aattlist[0:windows]):
        aattlist[i]=1
    return(aattlist)


def get_ka(dim,text,g,chance,windows,stability,iteration_wordbase,need_iteration_frequency):
    ka={}
    iteration_frequency=0
    words=jieba.lcut(text,cut_all=False)
    for i,word in enumerate(words):
        if i<windows:
            offo_textwindows=words[i+1:windows-1]
            windows_support_token(word,ka,chance,g,windows,offo_textwindows,stability,dim)
        if i>=windows:
            textwindows=words[i-windows:i]
            windows_support_token(word,ka,chance,g,windows,textwindows,stability,dim)
        if len(ka)>iteration_wordbase and iteration_frequency<need_iteration_frequency:
            iteration_frequency=iteration_frequency+1
            for a in ka:
                ka[a][-1]=1
        if iteration_frequency>=need_iteration_frequency:
            return(ka)

def get_kb(words_attlist_words,embedded_table_a,dim,text,h,chance,windows,stability,iteration_wordbase,need_iteration_frequency,base):
    kb={}
    iteration_frequency=0
    words=jieba.lcut(text,cut_all=False)
    for i,word in enumerate(words):
        if i<windows:
            offo_textwindows=words[i+1:windows-1]
            kb[word]=windows_context_suport_token(word,kb,embedded_table_a,words,words_attlist_words,windows,offo_textwindows,dim,base)
        if i>=windows:
            textwindows=words[i-windows:i]
            kb[word]=windows_context_suport_token(word,kb,embedded_table_a,words,words_attlist_words,windows,textwindows,dim,base)
        if len(kb)>iteration_wordbase and iteration_frequency<need_iteration_frequency:
            iteration_frequency=iteration_frequency+1
            for a in kb:
                kb[a][-1]=1
        if iteration_frequency>=need_iteration_frequency:
            return(kb)

                        
class pron:
    windows=5
    stability=50
    dim=256
    base=0.7
    need_iteration_frequency=10
    iteration_wordbase=1000
p=pron()

with open("nn.txt","r",encoding="utf-8") as f:
    text=f.read()
