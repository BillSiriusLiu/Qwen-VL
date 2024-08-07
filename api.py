import io
import base64
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# 第一步，加载大模型
model_dir = '../Qwen-VL-Chat-7B'
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda", trust_remote_code=True).eval()
 
# 第二步，创建FastAPI应用实例
app = FastAPI()
 
def save_base64_image(base64_string, save_path):
    image_data = base64.b64decode(base64_string)
    with open(save_path, 'wb') as f:
        f.write(image_data)

# 第三步，定义请求类型，与OpenAI API兼容
class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    max_tokens: int = 1024
    temperature: float = 0.1
 
 
# 第四步，定义交互函数
def chat_handle(messages: list, max_tokens: int, temperature: float):
    contents = messages[0]["content"]
    query = ""
    for c in contents:
        if c["type"] == "text":
            query += c["text"]
        else:
            tmp_path = "tmp/0.jpg"
            save_base64_image(c["image"], tmp_path)
            query += f'<img>{tmp_path}</img>'


    response, history = model.chat(
        tokenizer,
        query=query,
        history=None,
        top_p=1,
        max_new_tokens=max_tokens,
        temperature=temperature)
    print(response)
    return response
 
 
# 第五步，定义路由和处理函数，与OpenAI的API兼容
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    # 调用自定义的文本生成函数
    response = chat_handle(request.messages, request.max_tokens, request.temperature)
    return {
        "choices": [
            {
                "message": {
                    "content": response
                }
            }
        ],
        "model": request.model
    }
 
 
# 第六步，启动FastAPI应用，默认端口为8000
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9999)
