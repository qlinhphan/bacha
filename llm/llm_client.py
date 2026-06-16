from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
import os

# 1. Initialize the ChatOpenAI model
def llms():
    model = ChatOpenAI(model=os.getenv("MODEL_CHAT"), api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("BASE_URL"))
    return model

# # 2. Define your template with roles and placeholders
# chat_template = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful travel assistant that plans trips for {budget} budgets."),
#     ("user", "What are 3 things I should do in {city}?")
# ])

# 3. Combine them into an executable chain using the pipe (|) operator
# chain = chat_template | model

# # 4. Invoke the chain with your dictionary variables
# response = chain.invoke({
#     "budget": "low",
#     "city": "Tokyo"
# })

# # Print the text response cleanly
# print(response.content)