### 1. 配置环境
python 3.8

### 2. 安装百度飞桨深度学习框架
参考 https://www.paddlepaddle.org.cn/

conda install paddlepaddle-gpu==2.4.2 cudatoolkit=11.7 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge

### 3. 相关说明

1. 受内存限制，对大部分文件内容进行了清楚，但保留文件进行占位，可以后续将对应文件替换进来。
2. 主目录下main为模型全流程算法实现
3. 主目录下data_prepare用于将数据处理为百度飞桨可操作数据对象
4. 主目录下quesion结尾的文件用于识别问题核心词后将核心词替换进去（详细逻辑见论文/报告）
5. 主目录下cgm为结合大模型ChatGLM的代码
6. 主目录下cgt为结合大模型ChatGPT的代码
7. utils为工具包函数
8. ke为相关数据源文件和中间文件
9. checkpoint为模型输出

### 4.数据说明

```
数据格式说明
[
  {
    "id": "f54e9fa55c8180d387dc337ce22b52bcfcf32f2025540e651131ac547e47e3a6",       // 每一条数据（一个文章 + 一个问题为一条数据）的唯一 ID 编号
    "title": "Wayne State University College of Engineering",                       // 文章标题
    "passage": "The Wayne is responsible for all engineering related programs at Wayne .With alumni of the college totaling over 25,000 , it is one of the premier engineering colleges in [State A] along with being in the top 30 % of the country .Founded in 1933 , the College of Engineering has grown to include a variety of programs ranging from civil engineering , biomedical engineering , and many others .It is one of only 24 PACE partner labs in the country as well as being a leader in biomedical engineering .The College of Engineering is located in the Wayne State campus in Detroit .It is located in the College of Engineering building which is shared with the Danto Engineering Development Center .The current Dean of Engineering is Dr. Farshad Fotouhi .",    // 文章本身
    "question": {    // 文章对应的问题
      "KoRC-H": "Which region of the country does State A belong to?",      // 人类标注构建的问题
      "KoRC-T": "What is the part of [State A]?",                           // 使用模板生成的问题
      "KoRC-L": " What subdivisions are included in State A?",              // 使用 GPT：text-davinci-003 生成的问题中，质量最高的一个
      "KoRC-L-candidate": [                     // 使用 GPT: text-davinci-003 生成的问题
        "",
        "1. Which areas are part of State A?",
        "2. What divisions make up State A?",
        "3. What are the components of State A?",
        "4. What sections compose State A?",
        "5. What subdivisions are included in State A?"
      ]
    },
    "answers": [                                // 问题答案列表
      "Upper Peninsula of Michigan",
      "Lower Peninsula of Michigan"
    ],
    "answer_ids": [                             // 问题答案对应的 Wikidata 中的 ID
      "Q1338",
      "Q3596"
    ],
    "question_entity": "Michigan",              // 匿名化的实体原本的名字
    "question_entity_id": "Q1166",              // 匿名化的实体原本的 Wikidata ID
    "reasoning_paths": {                        // 回答问题的参考推理路径
      "Upper Peninsula of Michigan": [
        "Michigan (Q1166) <- located in the administrative territorial entity (P131) <- Wayne State University College of Engineering (Q16971485) -> country (P17) -> United States of America (Q30) <- country (P17) <- Upper Peninsula of Michigan (Q1338)",
        "Michigan (Q1166) <- located in the administrative territorial entity (P131) <- Wayne State University (Q349055) -> country (P17) -> United States of America (Q30) <- country (P17) <- Upper Peninsula of Michigan (Q1338)"
      ],
      "Lower Peninsula of Michigan": [
        "Michigan (Q1166) <- located in the administrative territorial entity (P131) <- Wayne State University College of Engineering (Q16971485) -> country (P17) -> United States of America (Q30) <- country (P17) <- Lower Peninsula of Michigan (Q3596)",
        "Michigan (Q1166) <- located in the administrative territorial entity (P131) <- Wayne State University (Q349055) -> country (P17) -> United States of America (Q30) <- country (P17) <- Lower Peninsula of Michigan (Q3596)"
      ]
    }
  }
]

```

