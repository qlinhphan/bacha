# import requests
import requests
import re
import json
def llama_clients(knowledge, context, q):
    prompt = f"""
        Bạn là trợ lý AI trong y tế, chuyên đưa ra các hướng điều trị cho bác sĩ dựa vào kiến thức được cung cấp. TUYỆT ĐỐI KHÔNG TRẢ LỜI CÁC CÂU HỎI NGOÀI PHẠM VI Y TẾ
        Kiến thức: {knowledge} 
        Lịch sử hội thoại: {context}
        Câu hỏi: {q}
        - Quy tắc trả lời:
        1. Nói chuyện hoà đồng, Khi bác sĩ nói về tình trạng bệnh nhân THÌ PHẢI DỰA VÀO KIẾN THỨC ĐƯỢC CUNG CẤP ĐỂ TRẢ LỜI, Không trả lời các câu hỏi ngoài phạm vi y tế
        2. Nếu không tìm thấy hướng điều trị phù hợp, hãy trả lời "Xin lỗi, tôi không tìm thấy hướng điều trị phù hợp dựa trên kiến thức hiện có."
        3 Cuối cùng KHI ĐÃ TÌM ĐƯỢC HƯỚNG ĐIỀU TRỊ hãy trả lời theo 3 ý: PHƯƠNG PHÁP, GHI CHÚ VÀ LƯU Ý từ kiến thức bạn có
    """
    response = requests.post(
        'http://10.10.61.29:11434/api/generate',
        json={
            'model': 'gemma2:9b',
            'prompt': prompt,
            'stream': False
        }
    )
    return response.json()['response']

def extract_json(text):
    # 1. Lấy phần nằm trong ```json ... ``` nếu có
    match = re.search(r"```json(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1)

    # 2. fallback: lấy phần {...}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    # 3. parse JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "score": None,
            "reason": "Invalid JSON",
            "raw": text
        }

def llama_test_semantic(answer, ground_truth):
    prompt = f"""
        Bạn là hệ thống đánh giá chất lượng câu trả lời.

        Nhiệm vụ: so sánh câu trả lời (A) với đáp án chuẩn (B) theo nghĩa (semantic meaning), không quan tâm cách diễn đạt.

        Câu trả lời A: {answer}

        Đáp án chuẩn B: {ground_truth}

        Hãy chấm điểm từ 0 đến 10 theo tiêu chí:

        - 10: đúng hoàn toàn về ý nghĩa, không thiếu thông tin quan trọng
        - 7-9: đúng ý chính, thiếu ít chi tiết nhỏ
        - 4-6: đúng một phần, còn thiếu hoặc sai một phần quan trọng
        - 1-3: liên quan rất ít
        - 0: sai hoàn toàn

        Chỉ trả về JSON:
        {{
        "score": "",
        "reason": ""
        }}
    """
    response = requests.post(
        'http://10.10.61.29:11434/api/generate',
        json={
            'model': 'gemma2:9b',
            'prompt': prompt,

            'stream': False
        }
    )
    return extract_json(response.json()['response'])

if __name__ == "__main__":

    
    # context = []
    # while True:
    #     q = input("Ban: ")
    #     res = llama_clients(knowledge, context, q)
    #     print(res)
    #     context.append(res)
    #     print("====================================================================")
    answer = "tôi nghĩ là con chó"
    ground_truth = "con chó"
    res = llama_test_semantic(answer, ground_truth)
    print(res)