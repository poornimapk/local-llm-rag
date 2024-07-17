from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.llamafile import Llamafile
from openai import OpenAI


def main():
    # local_llm = OpenAI(
    #     base_url="http://localhost:8080/v1",
    #     api_key="sk-no-key-required")
    #
    # completion = local_llm.chat.completions.create(
    #     model="LlaMa_CPP",
    #     messages=[
    #         {"role": "system", "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user "
    #                                       "fulfillment via helping them with their requests."},
    #         {"role": "user", "content": "Write a limerick about python exceptions"}
    #     ]
    # )
    # print(completion.choices[0].message)
    # local_llm = Llamafile(base_url="http://localhost:8080", temperature=0, seed=0)
    local_llm = Llamafile(temperature=0, seed=0, request_timeout=60.0)
    # resp = local_llm.complete("who is joe biden?")
    # print(resp)
    print("local llm loaded!")
    # Settings.llm = local_llm
    # model_kwargs = {'trust_remote_code': True}
    # model_path = "C:\\Users\\confl\\gpt4all\\resources\\nomic-embed-text-v1.5.f16.gguf"
    # embed_model = HuggingFaceEmbeddings(model_name=model_path,
    #                                     model_kwargs=model_kwargs)
    # Settings.embed_model = embed_model
    #
    # local_llm = GPT4All(model_name="Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf",
    #                     model_path="C:\\Users\\confl\\gpt4all\\resources\\")

    document_reader = SimpleDirectoryReader("./data/")
    documents = document_reader.load_data()

    vector_store = MilvusVectorStore(collection_name="paulg", dim=1536, overwrite=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    print("Doc loaded to Milvus")

    index = VectorStoreIndex.from_documents(documents, storage_context)

    # Settings.llm = local_llm
    query_engine = index.as_chat_engine(llm=local_llm)

    query = "Who is Paul Graham?"
    print("Query created, next step: ask user question to local LLM")
    response = query_engine.chat(query)
    print(response)


if __name__ == "__main__":
    main()
