import torch
@torch.no_grad()
def chatbot(model, tokenizer, max_len, device):
    model.eval()

    while True:
        text = input("user > ").strip()
        if text == "quit":
            break

        text = "<q>" + text + "</s><a>"
        x = tokenizer.encode(text, return_tensors="pt").to(device)
        result = model.generate(x,
                                max_length=max_len,
                                use_cache=True,
                                repetition_penalty=2.0,
                                do_sample=True, # 확률적 샘플링
                                top_k = 50, # 상위확률 k개의 토큰들 중에서 확률적 샘플링
                                temperature = 1.5, # 소프트 맥스 온도
                                # top_p = 0.9  # 0~1 사이 값을 전달, 상위확률 n개의 토큰들의 누적확률을 이용한 확률적 샘플링
                                )

        q_len = len(text) + 1
        result = tokenizer.decode(result[0])
        print("bot > " , result[q_len:-4])