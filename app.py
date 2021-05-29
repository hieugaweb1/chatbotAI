import random
import Chatbot_Diagnosing
from flask import Flask, request
from pymessenger.bot import Bot

app = Flask(__name__)
ACCESS_TOKEN = 'EAAD8XQoFts4BAA8A8OXjPUT0zvHRIyUlPu65laxZCP0TuwiEMaI3J2KTE7p4w9ZCwfZBoxRZAVcCRlgcjyfFLNIPhfPcjt6Llb3fJLmstTIZAy3enPZBKT2udJjBcDjNYUOJ55F0J1yk7JMwGulgRcnGA0dJ90xeLzg3RBz3ZBqzNTeWKBOzZAUu5v4u0eoD1voZD'
VERIFY_TOKEN = 'EAAD8XQoFts4BAA8A8OXjPUT'
bot = Bot(ACCESS_TOKEN)

@app.route("/", methods=['GET', 'POST'])
def receive_message():
    # nếu method là GET, trả về token để xác nhận ứng dụng
    if request.method == 'GET':
        token_sent = request.args.get("hub.verify_token")
        return verify_fb_token(token_sent)

    # ngược lại, tiến hành lấy tin nhắn và phản hồi với user
    else:
    # lấy tin nhắn của user
        output = request.get_json()
        for event in output['entry']:
            messaging = event['messaging']
            for message in messaging:
                if message.get('message'):
                    # gửi = text
                    recipient_id = message['sender']['id']
                    if message['message'].get('text'):
                        response_sent_text = get_message(message['message'].get('text'))    
                        send_message(recipient_id, response_sent_text)

                    # nếu người dùng gửi tin nhắn k phải text
                    if message['message'].get('attachments'):
                        response_sent_nontext = "Please send text ^^"   # xử lí khác
                        send_message(recipient_id, response_sent_nontext)
    return "Message Processed"


def verify_fb_token(token_sent):
    if token_sent == VERIFY_TOKEN:
        return request.args.get("hub.challenge")
    return 'Invalid verification token'


# xử lí tin nhắn bot gửi đi
def get_message(userMess):  
    return Chatbot_Diagnosing.chat(userMess)

def send_message(recipient_id, response):
    bot.send_text_message(recipient_id, response)
    return "success"

if __name__ == "__main__":
    app.run()






