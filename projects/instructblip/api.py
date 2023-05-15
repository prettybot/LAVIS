from fastapi import FastAPI, File, UploadFile, Form
from typing import Optional
import uvicorn, json, datetime
from lavis.models import load_model_and_preprocess
import torch
from io import BytesIO
from PIL import Image

app = FastAPI()

@app.post("/")
async def process_data(
    image: UploadFile = File(...), 
    prompt: str = Form(...),
    min_len: Optional[int] = Form(1),
    max_len: Optional[int] = Form(250),
    decoding_method: Optional[str] = Form("Beam search"),
    top_p: Optional[float] = Form(0.9),
    beam_size: Optional[int] = Form(5),
    len_penalty: Optional[int] = Form(1),
    repetition_penalty: Optional[int] = Form(1),
    model_type: Optional[str] = Form("vicuna7b")
):
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
    img = Image.open(BytesIO(image.read())).convert('RGB')
    image_processed = vis_processors["eval"](img).unsqueeze(0).to(device)
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
