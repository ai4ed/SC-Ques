import sys
import pickle
import json
sys.path.append("../predicts/")
import os
import pandas as pd

with open("../datasets/SC-Ques/test.jsons", "r") as f:
    processed_data = []
    for line in f.readlines()[0:100]:
        processed_data.append(json.loads(line))

if sys.argv[1] == "bart":
    from ModelBart import AutoFillBlank
elif sys.argv[1] == "bert":
    from ModelBert import AutoFillBlank
elif sys.argv[1] == "electra":
    from ModelElectra import AutoFillBlank
elif sys.argv[1] == "xlnet":
    from ModelXlnet import AutoFillBlank
elif sys.argv[1] == "roberta":
    from ModelRoberta import AutoFillBlank
else:
    raise ValueError

file_path = sys.argv[2]
model_path = sys.argv[3]

"""
model.predict
input: {'choice_dict': {'A': "We go to school from Monday to Friday. And we don't go to school on the weekend.",
   'B': "We go to school on Monday on Friday. And we don't go to school on the weekend.",
   'C': "We go to school from Monday to Friday. And we don't go to school from the weekend."}}
output: "ans", prob_dict: {"A":0.5, "B":0.3, "C":0.2}
"""

def process_predict(processed_data, model_name):
    predict_file = file_path
    '''
    if os.path.exists(predict_file):
        predict_data = pickle.load(open(predict_file, "rb"))
        return predict_data
    '''
    model = AutoFillBlank(model_path)
    predict_data = []
    for i in range(len(processed_data)):
        json_line = processed_data[i]
        model_out = model.predict(json_line["choice_dict"])[1]
        json_line["pred"] = model_out["ans"]
        json_line["prob_dict"] = model_out["prob_dict"]
        json_line["prob_dict"] = model_out["prob_dict"]
        predict_data.append(json_line)
#         break
        if i % 100 == 0:
            print(str(i) + "|", end = "\t")
    with open(predict_file, "wb") as fw:
        pickle.dump(predict_data, fw)
    return predict_data

def get_class_acc(predict_data, threshold=0.0):
    all_num = 0; right_num = 0
    print(len(predict_data), len(processed_data))
    assert len(predict_data) == len(processed_data)
    i = 0; out_lines = []
    for line in predict_data:
        out_lines.append(line)
        
        prob_lsts = [x[1] for x in line["prob_dict"].items()]
        prob = max(prob_lsts)
        if prob >= threshold:
            if line["answer"] == line["pred"]:
                right_num += 1
            all_num += 1
    df = pd.DataFrame(out_lines)
    df.to_excel("./pred_result"+sys.argv[1]+".xlsx", index=False)
    return str(right_num / all_num), str(all_num)


predict_data = process_predict(processed_data, sys.argv[1])

print("model: ", sys.argv[1], sys.argv[2], sys.argv[3])
print("total", get_class_acc(predict_data))
