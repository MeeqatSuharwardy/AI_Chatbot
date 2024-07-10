from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import nltk
import torch
import torch.nn as nn
import sacrebleu

nltk.download("punkt")
DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "results\q.bin",
    )
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    criterion = nn.CrossEntropyLoss()
    outputs = llm({'context': query, 'question': target_answer})
    loss = criterion(outputs['logits'], outputs['labels'])
    perplexity = torch.exp(loss)
    return qa, loss.item(), perplexity.item()

#output function
def final_result(query):
    qa_result, , loss, perplexity = qa_bot()
    response = qa_result({'query': query})
    # Calculate ROUGE score
    rouge = sacrebleu.sentence_rouge(response, [target_answer]).score
    return response

#chainlit code
# @cl.on_chat_start
# async def start():
#     chain = qa_bot()
#     msg = cl.Message(content="Starting the bot...")
#     await msg.send()
#     msg.content = "Hi, Welcome to Medical Bot. What is your query?"
#     await msg.update()

#     cl.user_session.set("chain", chain)

# @cl.on_message
# async def main(message: cl.Message):
#     chain = cl.user_session.get("chain") 
#     cb = cl.AsyncLangchainCallbackHandler(
#         stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
#     )
#     cb.answer_reached = True
#     res = await chain.acall(message.content, callbacks=[cb])
#     answer = res["result"]
#     sources = res["source_documents"]

#     if sources:
#         answer += f"\nSources:" + str(sources)
#     else:
#         answer += "\nNo sources found"

#     await cl.Message(content=answer).send()


def main():
    print("Starting the bot...")
    qa_chain = qa_bot()

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        result = qa_chain({'query': user_input})
        answer = result["result"]
        sources = result["source_documents"]

        if sources:
            answer += f"\nSources: {sources}"
        else:
            answer += "\nNo sources found"

        print("Bot:", answer)
        print("Loss:", loss)
        print("Perplexity:", perplexity)
        print("ROUGE Score:", rouge)


main()
