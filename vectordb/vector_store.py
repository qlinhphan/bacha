import faiss
from pprint import pprint
import numpy as np

def vector_stores(mycol):
    data_db = list(mycol.find())
    get_vector = np.array([d['vector'] for d in data_db], dtype=np.float32)
    faiss.normalize_L2(get_vector)


    ind = faiss.IndexFlatIP(len(data_db[0]['vector']))
    ind.add(get_vector)

    faiss.write_index(ind, "faiss.index")
    # print(ind)

