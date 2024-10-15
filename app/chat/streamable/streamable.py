import queue
import time
from threading import Thread
from flask import current_app
from app.chat.callbacks.customCallback import MyCustomHandler
from langchain_core.messages import AIMessage, HumanMessage

class StreamableChat:
    def __init__(self, chat, input_data, config, conversation):
        self.chat = chat
        self.input_data = input_data
        self.config = config
        self.conversation = conversation

    def stream(self):
        token_queue = queue.Queue()
        handler = MyCustomHandler(token_queue)

        def task(app_context):
            app_context.push()
            try:
                for chunk in self.chat.stream(self.input_data, config=self.config):
                    answer_token = chunk.get('model', {}).get('answer', '')
                    for token in answer_token:
                        handler.on_llm_new_token(token)
                handler.on_llm_end(None, None)
                token_queue.put(None)  # Đảm bảo vòng lặp kết thúc
            except Exception as e:
                handler.on_llm_error(e)

        Thread(target=task, args=[current_app.app_context()]).start()

        tokens = []
        while True: 
            time.sleep(0.005)  # Giảm độ trễ xuống 0,001s cho mỗi token
            token = token_queue.get()
            if token is None:
                break
            tokens.append(token)
            yield token

        # Kết hợp các tokens thành một message hoàn chỉnh
        ai_message_content = ''.join(tokens)

        # Thêm các message vào conversation
        human_message_content = self.input_data.get('input', '') if isinstance(self.input_data, dict) else self.input_data
        self.conversation.add_message(HumanMessage(content=human_message_content))
        self.conversation.add_message(AIMessage(content=ai_message_content))