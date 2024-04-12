# Load the model in fairseq
from fairseq.models.roberta import RobertaModel

# import torch
# lz = torch.load('./roberta.base/model1.pt')
# for param_tensor in lz['model'].keys():
#     print(param_tensor)

# exit()

# for parameter in lz.parameters():
#     print(parameter)

# https://cloud.tsinghua.edu.cn/f/e03f7a904526498c81a4/?dl=1
roberta = RobertaModel.from_pretrained('./roberta.large', checkpoint_file='model.pt')
roberta.eval()  # disable dropout (or leave in train mode to finetune)

#with open('learning_to_retrieve_reasoning_paths/unfamiliar_questions.txt') as f:
#    for line in f.readlines():
#passage = "Waterloo, New York (village):Waterloo is a village in Seneca , <mask> ( State ) , United States .The population was 5,171 at the 2010 census and is now the most populated village in Seneca .The village is named after the Waterloo in Belgium , where Napoleon was defeated .It is the primary county seat of Seneca , with the other being Ovid as part of a two - shire system established in 1822 .Most of the county administrative offices are located in the village .Therefore , many political sources only list Waterloo as the county seat .The Village of Waterloo is mostly in the Town of Waterloo , but the part south of the Cayuga - Seneca Canal of the village is in the Town of Fayette and a small area in the southeast of the village is in Town of Seneca Falls .Waterloo is east of Geneva and is located in between the two main Finger Lakes , Seneca Lake and Cayuga Lake ."
complement=[]
with open('learning_to_retrieve_reasoning_paths/unfamiliar_question.txt') as fp:
    for line in fp.readlines():
#passage = "Grave Dwellers , is a term about a particular form of homelessness in <mask> , in which unstable or poor people are intending to get rid of the cold , so they sleep inside the graves not in use yet , rather than cardboard boxes in public areas .This phenomenon was first noted in 2016 with the report of Shahrvand Newspaper , mentioning about 50 grave dwellers in the cemetery of Nasirabad Shahriar , Tehran , and caused many reactions in the social networks and among the social celebrities .Some sources , after the publication of the report about the grave dwellers in Nasirabad , reported that the cemetery was cleared by beating and pounding them out of there .In a letter to President Rouhani , the Oscar - winning director Asghar Farhadi expressed shame and regret about the condition of those men , women , and children who spend their cold nights in a graveyard."
        try:
            mask_index = line.split(' ').index("<mask>")
        except:
            print(line)
            exit()
        a = roberta.fill_mask(line, topk=1)[0][0]
        print(a.split(' ')[mask_index])
        complement.append(a.split(' ')[mask_index]+'\n')
with open('learning_to_retrieve_reasoning_paths/preprocessed_questions.txt','a') as fp_new:
    fp_new.writelines(complement)  

#'
# [('The first Star wars movie came out in 1977', 0.9504712224006653), ('The first Star wars movie came out in 1978', 0.009986752644181252), ('The first Star wars movie came out in 1979', 0.00957468245178461)]