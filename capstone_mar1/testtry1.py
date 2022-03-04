import numpy as np

label_dict = {'category1':0, 'category2':1}

dfindexvalues_list = []
for i in range(89+64):
    dfindexvalues_list.append(i)

dfindexvalues = np.array(dfindexvalues_list)
print(dfindexvalues)
