import streamlit as st
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from pathlib import Path

# 모델과 토크나이저 로드
model_name = "huggyllama/llama-7b"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# GPU 사용 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 스트림릿 앱 설정
st.title("LLaMA QLoRA Chatbot")
st.markdown("파인튜닝된 LLaMA 모델을 사용하는 챗봇입니다. 질문을 입력하세요!")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

user_input = st.text_input("You: ", key="input")

if user_input:
    response = generate_response(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(response)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"])):
        st.write(f"You: {st.session_state['past'][i]}")
        st.write(f"Bot: {st.session_state['generated'][i]}")
