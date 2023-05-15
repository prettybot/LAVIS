from fastapi import FastAPI, Request
import uvicorn, json, datetime
from lavis.models import load_model_and_preprocess
import torch

app = FastAPI()

@app.post("/")
async def generate_response(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    image = json_post_list.get("image")
    prompt = json_post_list.get("prompt")
    min_len = json_post_list.get("min_len", 1)
    max_len = json_post_list.get("max_len", 250)
    decoding_method = json_post_list.get("decoding_method", "Beam search")
    top_p = json_post_list.get("top_p", 0.9)
    beam_size = json_post_list.get("beam_size", 5)
    len_penalty = json_post_list.get("len_penalty", 1)
    repetition_penalty = json_post_list.get("repetition_penalty", 1)
    model_type = json_post_list.get("model_type", "vicuna7b")
    if model_type not in ["vicuna7b", "vicuna13b"]:
        return {"error": "model_type must be vicuna7b or vicuna13b"}
    if decoding_method not in ["Beam search", "Nucleus sampling"]:
        return {"error": "decoding_method must be 'Beam search' or 'Nucleus sampling'"}


    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_vicuna_instruct",
        model_type="vicuna7b",
        is_eval=True,
        device=device,
    )
    image_processed = vis_processors["eval"](image).unsqueeze(0).to(device)
    samples = {
        "image": image_processed,
        "prompt": prompt,
    }
    output = model.generate(
            samples,
            length_penalty=float(len_penalty),
            repetition_penalty=float(repetition_penalty),
            num_beams=beam_size,
            max_length=max_len,
            min_length=min_len,
            top_p=top_p,
            use_nucleus_sampling=decoding_method == "Nucleus sampling",
        )

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    response = {
        "output": output[0],
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'

    print(log)
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
