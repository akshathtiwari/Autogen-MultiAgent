from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from app.runtime.runtime_manager import RuntimeManager
from app.messages.message_types import UserCredentials

runtime_manager = RuntimeManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await runtime_manager.start_runtime()
    yield
    await runtime_manager.stop_runtime()

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    conversation_state = "await_username"
    session_id = None
    try:
        while True:
            if conversation_state == "await_username":
                await websocket.send_text("Enter your username:")
                username = await websocket.receive_text()
                session_id = username.strip()
                runtime_manager.register_websocket(session_id, websocket)
                conversation_state = "await_password"
            elif conversation_state == "await_password":
                await websocket.send_text("Enter your password:")
                password = await websocket.receive_text()
                await runtime_manager.publish_credentials(
                    UserCredentials(username=session_id, password=password),
                    session_id
                )
                conversation_state = "await_name"
            elif conversation_state == "await_name":
                await websocket.send_text("May I know your name?")
                name = await websocket.receive_text()
                await websocket.send_text(f"Hello, {name}!")
                await websocket.send_text("Please describe your banking issue or question:")
                conversation_state = "await_query"
            elif conversation_state == "await_query":
                query = await websocket.receive_text()
                await runtime_manager.publish_user_message(query, session_id)
                conversation_state = "in_conversation"
            elif conversation_state == "in_conversation":
                user_msg = await websocket.receive_text()
                await runtime_manager.publish_user_message(user_msg, session_id)
    except WebSocketDisconnect:
        runtime_manager.unregister_websocket(session_id)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
