from modelscope import (
    snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
)
import torch
model_id = 'qwen/Qwen-VL-Chat'
revision = 'v1.0.0'

model_dir = "../7Bqw_vl_chat"
torch.manual_seed(1234)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
if not hasattr(tokenizer, 'model_dir'):
    tokenizer.model_dir = model_dir

# use bf16
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device, trust_remote_code=True, fp16=True).eval()

model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device, trust_remote_code=True).eval()



# 1st dialogue turn
# Either a local path or an url between <img></img> tags.
image1_path = 'img/cafe/cafe1.jpg'
image2_path = 'img/cafe/cafe2.jpg'
init_prompt_path = 'prompt/init.txt'
task_prompt_path = 'prompt/task.txt'
init, task = "", ""
with open(init_prompt_path, 'r') as f:
    for line in f:
        init += line
with open(task_prompt_path, 'r') as f:
    for line in f:
        task += line

query = init
init_idx = task.find("[Initial Environment Image]")
current_idx = task.find("[Environment Image after Executing Some Steps]")
query += task[:init_idx+27]
query += f'<img>{image1_path}</img>'
query += task[init_idx+27:current_idx+46]
query += f'<img>{image2_path}</img>'
query += task[current_idx+46:]

response, history = model.chat(tokenizer, query=query, history=None)
print(response)


# 2nd dialogue turn
#response, history = model.chat(tokenizer, '冰箱是什么颜色的', history=history)
#print(response)

#image = tokenizer.draw_bbox_on_latest_picture(response, history)
#if image:
#  image.save('output_chat.jpg')
#else:
#  print("no box")
