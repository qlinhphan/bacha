from chunking.chunks import load_data
from dotenv import load_dotenv
load_dotenv()
import os
import faiss
from vectordb.mongodb import save_data_mongodb
from vectordb.mongodb import mongodb
from embedding.embedder import get_embedding
from vectordb.vector_store import vector_stores
from rechieval.rechieval import rechieval_data
from prompts.prompt_temp import prompt_temp
from llm.llm_client import llms
from llm.llama_client import llama_clients
from prompts.prompt_temp import prompt_temp

def main():
    # connect mg
    mycol = mongodb(address=os.getenv("MONGODB_URI"))

    # read_file and split chunks
    file_in_docs = os.listdir('ingestion/docs')
    list_path_file = [os.path.join('ingestion/docs', fid) for fid in file_in_docs]

    all_data_in_db = list(mycol.find())
    if len(all_data_in_db) == 0:
        save_data_mongodb(list_path_file=list_path_file, load_data = load_data, mycol = mycol, get_embedding = get_embedding)
    else:
        print("didn't save knowledge into db")
    

    # build vector store file: name_file.index
    if not os.path.exists("faiss.index"):
        vector_stores(mycol=mycol)
        print("created vector_stores")
    else:
        print("didn't create vector_store")

    # search
    file_index = faiss.read_index("faiss.index")   

    context = []
    while True:
        
        q = input("Ban: ")
        q_vector = get_embedding(q)
        response = rechieval_data(
            question_vector=q_vector,
            index_file=file_index,
            top_k=10,
            mycol=mycol,   
            min_score=0.55,
        )

        res = llama_clients(response, context, q)
        print(res)
        context.append(res)
        print("====================================================================")


if __name__ == "__main__":
    main()
