import pickle

with open("/workspace/quantization/src/code_search/test_files/question_code_dataset/merged_file_1.txt", "rb") as fp: 
    s = pickle.load(fp)

print(s[:2])
# for i in s:
#     if i[2] == 0:
#         print(i)
#         break