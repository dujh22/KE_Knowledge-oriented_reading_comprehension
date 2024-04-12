import json,re
import spacy
import en_core_web_sm

nlp = spacy.load('en_core_web_sm')

#sent = "Which time zones is the country lied in?"
sent = "Which work does the religious leader appear in"
doc=nlp(sent)
root_index = None
start = 0
for tok in doc:
  print(tok,tok.dep_)
#   if tok.dep_=="ROOT":
#     root_index = start
#     print(tok)
#     break
#   start+=1

exit()


with open('./valid.json') as f:
    txt_list=json.load(f)
questions=[]
inputs=[]
title=0
passage_id=0

replace_num = [0,0] #：H,L
no_box = [0,0] #no_box,Hq==Tq
empty_box = [0,0] # empty box in T,;empty box in T, but box in passage

temp1 = ["ROOT","prep","prep"]
temp1 = ["ROOT","poss"]

# def process_ptag(q,ent,count_ind):


def replace_pattern(string,q,ent,count_ind, debug=False):
    flag = 0
    mat_str = string
    if mat_str=="":
        mat_str = ""
        doc=nlp(q)
        deps = []
        for tok in doc:
            deps.append(tok.dep_)
        if debug:
            print(deps)
        if "ROOT" in deps:
            rind = deps.index("ROOT")
            if debug: print(rind)
            if "prep" in deps[rind+1:]:
                pind = deps.index("prep")
                if debug: print(pind)
                for i in range(pind+1,len(doc)):
                    if deps[i]=="prep" or deps[i]=="punct":
                        break
                    if mat_str!="":
                        mat_str+=" "
                    mat_str+=str(doc[i])
                    if debug: print(mat_str)
            elif "poss" in deps[rind+1:]:
                pind = deps.index("poss")
                mat_str = str(doc[pind])
        elif "aux" in deps:
            if "nsubj" in deps:
                pind = deps.index("nsubj")
                mat_str = str(doc[pind])
        
        if mat_str!="":
            q_new = q.replace(mat_str,ent)
            if debug: print(mat_str in q, q_new)
            flag=1
            replace_num[count_ind]+=1
    else:
        if mat_str in q: #完全匹配
            q_new = q.replace(mat_str,ent)
            replace_num[count_ind]+=1
            flag=1
        else:
            indices = [i.start() for i in re.finditer(" ", string)] #以中心词作为末尾词，从前到后依次去掉词形成新向量做匹配
            doc=nlp(string)
            root_index = None
            start = 0
            for tok in doc:
                start+=len(tok)
                if tok.dep_=="ROOT":
                    root_index = start
                    break
                start+=1
            if debug:
                print(indices)
            for ind in indices:
                if ind+1>root_index:
                    break
                mat_str = string[ind+1:root_index]
                if debug:
                    print(mat_str,ind+1,root_index)
                if mat_str in q:
                    q_new = q.replace(mat_str,ent)
                    replace_num[count_ind]+=1
                    flag=1
                    break
            
            if flag==0:
                # root后介词prep后面直到下一个prep 没有prep时poss所有格
                # 无root有aux，后的nsubj
                mat_str = ""
                doc=nlp(q)
                deps = []
                for tok in doc:
                    deps.append(tok.dep_)
                if debug:
                    print(deps)
                if "ROOT" in deps:
                    rind = deps.index("ROOT")
                    if debug: print(rind)
                    if "prep" in deps[rind+1:]:
                        pind = deps.index("prep")
                        if debug: print(pind)
                        for i in range(pind+1,len(doc)):
                            if deps[i]=="prep" or deps[i]=="punct":
                                break
                            if mat_str!="":
                                mat_str+=" "
                            mat_str+=str(doc[i])
                            if debug: print(mat_str)
                    elif "poss" in deps[rind+1:]:
                        pind = deps.index("poss")
                        mat_str = str(doc[pind])
                elif "aux" in deps:
                    if "nsubj" in deps:
                        pind = deps.index("nsubj")
                        mat_str = str(doc[pind])
                
                if mat_str!="":
                    q_new = q.replace(mat_str,ent)
                    if debug: print(mat_str in q, q_new)
                    flag=1
                    replace_num[count_ind]+=1
            

    if flag==0:
        return q
    else:
        return q_new

count = 0
Tq_news = ""
Hq_news = ""
Lq_news = ""
for txt in txt_list:
    
    Tq = txt["question"]["KoRC-T"]
    Hq = txt["question"]["KoRC-H"]
    Lq = txt["question"]["KoRC-L"]

    #print(Tq)
    #if "[]" not in Tq:
    if "[" in txt["passage"]:
        #print("-------",count)
        matchobj = re.search(r'.*?\[(.*?)\].*', txt["passage"])
        try:
            string = matchobj.group(1)
        except:
            print("part box: ",txt["passage"],Tq)
            Hq_new = Hq
            Lq_new = Lq
            if "[]" in Tq:
                Tq_new = Tq.replace("[]",txt["question_entity"])
            else:
                Tq_new = Tq 
        else:
            Hq_new = replace_pattern(string,Hq,txt["question_entity"],0)
            Lq_new = replace_pattern(string,Lq,txt["question_entity"],1)
            if "[]" in Tq:
                Tq_new = Tq.replace("[]",txt["question_entity"])
            else:
                matchobj = re.search(r'.*?\[(.*?)\].*', Tq)
                try:
                    string = matchobj.group(1)
                except:
                    Tq_new = Tq
                else:
                    Tq_new = Tq.replace("[" + string + "]",txt["question_entity"])
        
        #if Hq_new==Hq or Lq_new==Lq:
        # if count in [0,9,11,12,19,25,34,35,39,59,64,68,71,75,76,77,78,92,95,96,81]:
        #     print(count,Tq)
        #     print(Hq_new==Hq,Hq,Hq_new)
        #     print(Lq_new==Lq,Lq,Lq_new)
        #     Hq_new = replace_pattern(string,Hq,txt["question_entity"],0,debug=True)
        #     Lq_new = replace_pattern(string,Lq,txt["question_entity"],1,debug=True)
        #     print()
        # if replace_num[0] != replace_num[1]:
        #     print(count,Tq,Hq,Lq)
        #     print(txt["question_entity"],Hq_new,Lq_new)
        #     exit()
        # except:
        #     print(Tq)

    else:
        no_box[0]+=1 #no box中70% H和T问题一样，其余很多T都是Which work did Cameron produce?这个问题
        # print("no box: ",txt["passage"])
        # print(Tq)
        # print()
        if Tq==Hq:
            no_box[1]+=1
        Hq_new = Hq
        Lq_new = Lq
        Tq_new = Tq
    
    #print(Tq,Tq_new)


    Hq_news+=Hq_new.strip('\n')
    Hq_news+="|||"
    Tq_news+=Tq_new.strip('\n')
    Tq_news+="|||"
    Lq_news+=Lq_new.strip('\n')
    Lq_news+="|||"

    
    count+=1
    if count%100==0:
        print(count)

# with open("./KoRC-H_valid.txt","w") as f:
#     f.write(Hq_news)
with open("./KoRC-T_valid.txt","w") as f:
    f.write(Tq_news)
# with open("./KoRC-L_valid.txt","w") as f:
#     f.write(Lq_news)


print("Total: {}, no box:{}, H=T in no_box:{}, empty_box: {}, not empty in passage:{}, \n replace H: {}, replace L: {},".format(len(txt_list),
                    no_box[0],no_box[1],empty_box[0],empty_box[1],
                    replace_num[0]/len(txt_list),replace_num[1]/len(txt_list)))
    

