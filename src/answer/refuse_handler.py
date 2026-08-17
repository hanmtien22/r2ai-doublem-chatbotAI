import logging

logger = logging.getLogger(__name__)

class RefuseHandler:
    """
    Xử lý từ chối và phản hồi khéo léo khi hệ thống không thể tìm ra kết quả
    hoặc khi quá trình tính toán bị lỗi hoàn toàn.
    """
    def __init__(self):
        pass

    def handle_refuse(self, reason: str = "unknown") -> str:
        """
        Tạo câu trả lời từ chối tùy theo nguyên nhân.
        """
        if reason == "no_hits":
            return "Xin lỗi, tôi không tìm thấy tài liệu hay số liệu nào liên quan đến câu hỏi của bạn trong cơ sở dữ liệu hiện tại."
        elif reason == "compute_failed":
            return "Xin lỗi, mặc dù đã tìm thấy dữ liệu liên quan nhưng tôi không thể thực hiện phép tính chính xác cho câu hỏi này. Bạn có thể thử diễn đạt lại câu hỏi không?"
        elif reason == "low_confidence":
            return "Dữ liệu tìm được có độ tin cậy thấp nên tôi không chắc chắn để đưa ra câu trả lời. Bạn có thể cung cấp thêm thông tin chi tiết hơn không?"
        else:
            return "Xin lỗi, đã xảy ra lỗi không xác định khi xử lý câu hỏi của bạn. Vui lòng thử lại sau."
