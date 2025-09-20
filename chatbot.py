import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.runnables import RunnableLambda
import requests
import os

# 🔐 Load API key
OPENROUTER_API_KEY = st.secrets.get("api_key") or os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    st.error("❌ OpenRouter API key not found. Please set it in Streamlit secrets or environment variables.")
    st.stop()

# 🌍 Language selector
language = st.selectbox("Choose your language", ["English", "Hindi", "Spanish", "French", "German"])
language_prompts = {
    "English": "You are a helpful assistant. Respond in English.",
    "Hindi": "आप एक सहायक सहायक हैं। कृपया हिंदी में उत्तर दें।",
    "Spanish": "Eres un asistente útil. Responde en español.",
    "French": "Vous êtes un assistant utile. Répondez en français.",
    "German": "Du bist ein hilfreicher Assistent. Antworte auf Deutsch."
}

# 🧠 Session state
if "system_message" not in st.session_state:
    st.session_state.system_message = {"role": "system", "content": language_prompts[language]}

if "buffer_memory" not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

if "messages" not in st.session_state:
    st.session_state.messages = [st.session_state.system_message,
                                 {"role": "assistant", "content": "How can I help you today?"}]

# ✅ OpenRouter call wrapped in RunnableLambda
def call_openrouter(inputs, **kwargs):
    # Get last 3 messages from memory
    memory_messages = st.session_state.buffer_memory.chat_memory.messages[-3:]
    messages = [st.session_state.system_message] + [
        {"role": msg.type, "content": msg.content} for msg in memory_messages
    ] + [{"role": "user", "content": inputs.to_string()}]

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "anthropic/claude-3.7-sonnet:thinking",
        "messages": messages,
        "max_tokens": 512
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return content

llm = RunnableLambda(call_openrouter)

# 🔧 Create conversation chain
conversation = ConversationChain(memory=st.session_state.buffer_memory, llm=llm)

# 🎨 UI
st.title("🗣️ Conversational Chatbot")
st.subheader("㈻ Simple Chat Interface for LLMs")

prompt = st.chat_input("Your question")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # 🚫 Block certain topics
                blocked_keywords = [
    "obama", "trump", "modi", "putin", "biden", "president", "politics","ukraine", "russia", "war", "conflict", "nato", "geopolitics", "election"]

                if any(word.lower() in prompt.lower() for word in blocked_keywords):
                    response = "I'm here to help with learning and general topics, but I avoid political or public figure discussions."
                else:
                    response = conversation.predict(input=prompt)

                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"❌ Error generating response: {e}")
