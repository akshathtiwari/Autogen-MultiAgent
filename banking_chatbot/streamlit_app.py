import streamlit as st
import nest_asyncio
nest_asyncio.apply()  

import asyncio
from autogen_core import SingleThreadedAgentRuntime, TopicId


from app.runner.main import create_runtime
from app.messages.message_types import UserLogin, MyMessageType
from app.agents.user_agent import RESPONSE_STORE


def create_runtime():    
    runtime = SingleThreadedAgentRuntime()    
    return runtime

def init_runtime():
    
    if "runtime" not in st.session_state:
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)        
        runtime = create_runtime()        
        runtime.start()        
        st.session_state.runtime = runtime
        st.session_state.loop = loop

def main():
    st.title("AutoGen Multi-Agent Demo")

    
    init_runtime()
    runtime = st.session_state.runtime
    loop = st.session_state.loop

    
    if "session_id" not in st.session_state:
        st.session_state.session_id = "akshath"
    session_id = st.session_state.session_id

    
    if "did_login" not in st.session_state:
        user_login = UserLogin(username=session_id)
        loop.run_until_complete(
            runtime.publish_message(user_login, TopicId("User", source=session_id))
        )
        st.session_state.did_login = True

    
    conversation_data = RESPONSE_STORE.get(session_id, {})
    messages = conversation_data.get("messages", [])
    for msg in messages:
        if msg["role"] == "user":
            st.write(f"**You**: {msg['content']}")
        else:
            st.write(f"**Agent**: {msg['content']}")

    
    user_query = st.text_input("Your query:", key="user_query", placeholder="e.g. 'payment mismatch'")

    if st.button("Send"):
        if user_query.strip():
            
            user_msg = MyMessageType(content=user_query)
            loop.run_until_complete(
                runtime.publish_message(user_msg, TopicId("User", source=session_id))
            )
            
            loop.run_until_complete(asyncio.sleep(0.1))

            
            st.session_state.user_query = ""
            st.experimental_rerun()

if __name__ == "__main__":
    main()
