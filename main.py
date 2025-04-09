import streamlit as st
import tempfile
import os

from rag import Rag

def display_messages():
  for message in st.session_state.messages:
    with st.chat_message(message['role']):
      st.markdown(message['content'])

def process_file():
  st.session_state["assistant"].clear()
  st.session_state.messages = []

  for file in st.session_state["file_uploader"]:
    with tempfile.NamedTemporaryFile(delete = False) as tf:
      tf.write(file.getbuffer())
      file_path = tf.name

    with st.session_state["feeder_spinner"], st.spinner("Uploading the file..."):
      st.session_state["assistant"].feed(file_path)
    os.remove(file_path)

def process_input():
  if prompt := st.chat_input("Ask me anything"):
    with st.chat_message("user"):
      st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    response = st.session_state["assistant"].ask(prompt)
    with st.chat_message("assistant"):
      st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

def main():
  st.title("Docu-Bot")

  if len(st.session_state) == 0:
    st.session_state["assistant"] = Rag()
    st.session_state.messages = []

  st.file_uploader(
      "Upload the document",
      type = ["pdf"],
      key = "file_uploader",
      on_change = process_file,
      label_visibility = "collapsed",
      accept_multiple_files = True,
    )

  st.session_state["feeder_spinner"] = st.empty()

  display_messages()
  process_input()

if __name__ == "__main__":
  main()