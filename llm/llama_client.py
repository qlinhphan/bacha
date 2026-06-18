# import requests
import requests
import re
import json

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

        [QUY TẮC ĐỊNH DẠNG ĐẦU RA - BẮT BUỘC ĐỌC KỸ]:
        Bạn phải phân loại câu trả lời và xuất ra đúng định dạng tương ứng:
        - TUYỆT ĐỐI KHÔNG DÙNG TIẾNG TRUNG VÀ CHỈ DÙNG TIẾNG VIỆT
        - ĐẢM BẢO VIẾT ĐÚNG CHÍNH TẢ TIẾNG VIỆT, CÓ DẤU CÁCH RÕ RÀNG GIỮA CÁC TỪ
        - Nếu hỏi danh tính ("Bạn là ai?"): "Tôi là trợ lý AI y tế hỗ trợ trích xuất phác đồ điều trị."
        - Nếu người dùng nói về bản thân họ thì hãy vui vẻ đón nhận và trả lời
        - Nếu câu hỏi hỏi về những gì đã trao đổi trước đó: Dựa vào {context} để tóm tắt hoặc nhắc lại lịch sử hội thoại bằng văn bản thông thường (Không dùng JSON).
        - Nếu câu hỏi là bệnh lý nhưng {knowledge} trống: "Xin lỗi, tôi không tìm thấy hướng điều trị phù hợp dựa trên kiến thức hiện có."
        - TRONG TẤT CẢ CÁC TRƯỜNG HỢP CÒN LẠI (Khi bạn dùng {knowledge} để đưa ra đáp án, đề xuất, hoặc giải thích về y tế): BẠN BẮT BUỘC CHỈ ĐƯỢC XUẤT RA 1 KHỐI JSON DUY NHẤT
        {{
        "Phương pháp": "...",
        "Ghi chú": "...",
        "Lưu ý": "..."
        }}
    """
    response = requests.post(
        'http://10.10.61.29:11434/api/generate',
        json={
            'model': 'qwen2.5:14b',
            'prompt': prompt,
            'stream': False
        }
    )
    res  = response.json()['response']

    # print("=================================================================")
    # print("HISTORY: ", context)
    # print("=================================================================")
    
    if '{' in res and '}' in res:
        return extract_json(res)
    else:
        return res


# tóm tắt cuộc hội thoại trong 1 phiên
def llama_summary_conversation(context):
    prompt = f"""
        Bạn là trợ lý AI trong lĩnh vực y tế, Chuyên làm nhiệm vụ tóm tắt phiên hội thoại dựa vào context được cung cấp.

        Lịch sử hội thoại:
        {context}

        [QUY TẮC ĐỊNH DẠNG ĐẦU RA - BẮT BUỘC ĐỌC KỸ]:
        1. Không sử dụng chủ ngữ, vị ngữ, ngôi.
        2. Chỉ tóm tắt đầy đủ ý nghĩa của triệu chứng bác sĩ đưa ra và câu trả lời
        3. Cuối cùng xuất ra dạng text
        
    """
    response = requests.post(
        'http://10.10.61.29:11434/api/generate',
        json={
            'model': 'qwen2.5:14b',
            'prompt': prompt,
            'stream': False  
        }
    )
    res  = response.json()['response']
    return res

    # print("=================================================================")
    # print("HISTORY: ", context)
    # print("=================================================================")
    
    # if '{' in res and '}' in res:
    #     return extract_json(res)
    # else:
    #     return res





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
            'model': 'qwen2.5:14b',
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