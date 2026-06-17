# import requests
import requests
import re
import json
def llama_clients(knowledge, context, q):
    prompt = f"""
        Bạn là trợ lý AI trong lĩnh vực y tế, chuyên hỗ trợ bác sĩ đề xuất hướng điều trị dựa trên kiến thức được cung cấp.
        Tuyệt đối không trả lời các câu hỏi ngoài phạm vi y tế.

        Kiến thức được cung cấp:
        {knowledge}

        Lịch sử hội thoại:
        {context}

        Câu hỏi của bác sĩ:
        {q}

        Quy tắc trả lời:
        1. Chỉ sử dụng thông tin trong phần "Kiến thức được cung cấp" làm căn cứ chính để trả lời.
        2. Không bịa thông tin, không đưa ra kết luận trái với kiến thức được cung cấp, không trả lời câu hỏi ngoài phạm vi y tế.
        3. Nếu trong kiến thức có thông tin liên quan trực tiếp hoặc gần đúng với ca bệnh, hãy tổng hợp và đề xuất hướng điều trị dựa trên những thông tin đó; không được từ chối chỉ vì kiến thức không trùng khớp 100%.
        4. Nếu kiến thức chưa đủ để kết luận chắc chắn, vẫn phải trả lời dựa trên phần thông tin gần nhất tìm thấy và nêu rõ điểm còn thiếu hoặc điều cần đánh giá thêm.
        5. Chỉ trả lời đúng nguyên văn câu sau khi phần "Kiến thức được cung cấp" hoàn toàn không có thông tin liên quan đến câu hỏi:
           "Xin lỗi, tôi không tìm thấy hướng điều trị phù hợp dựa trên kiến thức hiện có."
        6. Khi có thông tin phù hợp, hãy trả lời ngắn gọn, rõ ràng, đúng chuyên môn và đúng 3 mục sau:
           - PHƯƠNG PHÁP:
           - GHI CHÚ:
           - LƯU Ý:
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
    # 1. Láº¥y pháº§n náº±m trong ```json ... ``` náº¿u cÃ³
    match = re.search(r"```json(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1)

    # 2. fallback: láº¥y pháº§n {...}
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

        Nhiệm vụ: so sánh câu trả lời A với đáp án chuẩn B theo mức độ tương đồng về nghĩa, không chấm theo cách diễn đạt hay từ ngữ bề mặt.

        Câu trả lời A:
        {answer}

        Đáp án chuẩn B:
        {ground_truth}

        Hãy chấm điểm từ 0 đến 10 theo tiêu chí:
        - 10: Trùng khớp hoàn toàn về ý nghĩa, không thiếu thông tin quan trọng.
        - 7-9: Đúng ý chính, chỉ thiếu hoặc khác một vài chi tiết nhỏ.
        - 4-6: Đúng một phần, còn thiếu hoặc sai một phần quan trọng.
        - 1-3: Chỉ liên quan rất ít đến đáp án chuẩn.
        - 0: Sai hoàn toàn hoặc không liên quan.

        Yêu cầu đầu ra:
        - Chỉ trả về duy nhất một JSON hợp lệ.
        - Không thêm giải thích ngoài JSON.
        - Trường "score" là số từ 0 đến 10.
        - Trường "reason" giải thích ngắn gọn lý do chấm điểm.

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
    answer = "tÃ´i nghÄ© lÃ  con chÃ³"
    ground_truth = "con chÃ³"
    res = llama_test_semantic(answer, ground_truth)
    print(res)
