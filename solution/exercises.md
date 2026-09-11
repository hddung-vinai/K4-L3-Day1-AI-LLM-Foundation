# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
```
Ở temperature 0.0, các phản hồi gần như giống hệt nhau mỗi lần gọi (mô hình luôn chọn từ có xác suất cao nhất) và nội dung khá giống nhau, khuôn mẫu. Khi tăng dần lên 0.5, 1.0 rồi 1.5, câu trả lời trở nên đa dạng hơn về cách diễn đạt và có thể đưa ra những sự thật khác nhau giữa các lần gọi; đến 1.5, văn phong có thể trở nên lộn xộn, kém mạch lạc hoặc thậm chí sai lệch thông tin do mô hình đã chọn các từ ít khả năng hơn.
```

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
```
Nên đặt temperature thấp, khoảng 0.0–0.3, vì chatbot hỗ trợ khách hàng cần trả lời chính xác, nhất quán và đáng tin cậy (ví dụ: chính sách đổi trả, thông tin sản phẩm), chứ không cần sáng tạo hay đa dạng ngôn ngữ — temperature cao dễ dẫn đến câu trả lời không nhất quán hoặc bịa thông tin, gây mất uy tín và trải nghiệm xấu cho khách hàng
```

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
```
Ước tính chi phí (chỉ tính output, vì đề chỉ cho token đầu ra):
Tổng lượt gọi/ngày = 10.000 người × 3 lần = 30.000 lượt
Tổng token đầu ra = 30.000 × 350 = 10.500.000 token
GPT-4o: 10.500 × $0.010 = $105/ngày
GPT-4o-mini: 10.500 × $0.0006 = $6.3/ngày

→ GPT-4o đắt hơn khoảng 16,7 lần so với GPT-4o-mini cho cùng workload (đúng bằng tỷ lệ đơn giá output: 0.010/0.0006 ≈ 16.67).

Khi nào đáng dùng GPT-4o (đắt hơn):
Khi tác vụ đòi hỏi suy luận phức tạp, độ chính xác cao và ảnh hưởng trực tiếp đến uy tín hoặc doanh thu — ví dụ: soạn thảo hợp đồng pháp lý, phân tích tài chính, hoặc trả lời các câu hỏi kỹ thuật chuyên sâu mà sai sót có thể gây tổn thất lớn. Ở đây chi phí thêm là hợp lý để đổi lấy chất lượng và độ tin cậy.

Khi nào nên dùng GPT-4o-mini:
Khi tác vụ đơn giản, lặp lại với khối lượng lớn — ví dụ: chatbot trả lời câu hỏi thường gặp (FAQ), phân loại yêu cầu khách hàng, tóm tắt ngắn, hoặc các thao tác không cần suy luận sâu. Với 30.000 lượt gọi/ngày, chênh lệch $105 so với $6.3 mỗi ngày (~$3 so với ~$189/tháng) là rất đáng kể, nên mini là lựa chọn kinh tế hơn nhiều mà vẫn đáp ứng đủ nhu cầu.
```
---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
```
Hai phản hồi có độ dài từ gần bằng nhau (179 vs 183 từ) nhưng phản hồi 2 bị cắt cụt giữa chừng (do đạt giới hạn max_tokens=256), cho thấy văn phong chuyên sâu tốn nhiều token hơn để diễn đạt cùng một lượng ý. Về từ vựng, phản hồi 1 dùng ngôn ngữ đời thường, ví von cụ thể ("cuốn sổ ghi giao dịch mua kẹo", "khóa trang lại") phù hợp với trẻ 8 tuổi; phản hồi 2 dùng thuật ngữ kỹ thuật/tài chính chuyên ngành ("sổ cái phân tán", "hàm băm", "phi tập trung") kèm cấu trúc liệt kê rõ ràng, mạch lạc như văn bản chuyên môn. Điều này cho thấy system prompt định hình rất mạnh persona, giọng điệu, mức độ trừu tượng và cách tổ chức nội dung của model — dù cùng một câu hỏi và cùng nội dung cốt lõi (khái niệm blockchain), nhưng cách diễn đạt kiến thức đó sang tiếng Việt hoàn toàn khác nhau tùy vào vai trò được gán.
```
### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
```
Với đoạn văn 100 từ, số token thực tế theo tiktoken là 134, trong khi ước lượng theo công thức "số từ / 0.75" cho ra 133,3 — hai con số gần như trùng khớp, chỉ chênh lệch 0,5%. Điều này cho thấy công thức ước lượng "1 từ ≈ 1,33 token" (tương đương chia cho 0.75) tình cờ khá sát với thực tế đo được ở đoạn văn này, dù bản thân công thức này ban đầu được xây dựng dựa trên đặc điểm tokenization của tiếng Anh chứ không phải tiếng Việt.

Tiếng Việt vẫn thường tốn nhiều token hơn tiếng Anh ở cùng độ dài văn bản (134 token cho 100 từ, tức xấp xỉ 1,34 token/từ, cao hơn hẳn tiếng Anh — nơi 1 từ thường ứng với khoảng 0,75 token) vì các lý do sau:

Dấu thanh và ký tự Unicode phức tạp: Các nguyên âm có dấu (ă, â, ê, ô, ơ, ư) và 5 dấu thanh (sắc, huyền, hỏi, ngã, nặng) không nằm trong bảng mã ASCII cơ bản, khiến tokenizer (được huấn luyện chủ yếu trên tiếng Anh) phải mã hóa chúng bằng nhiều byte/token con thay vì gộp gọn thành 1 token như chữ cái Latin thường.
Từ đơn âm tiết bị tách nhỏ: Tiếng Việt là ngôn ngữ đơn lập, mỗi "từ" có thể là một hoặc nhiều âm tiết cách nhau bằng khoảng trắng (ví dụ "đất nước" = 2 âm tiết), và mỗi âm tiết lại có thể bị tách thành nhiều token, trong khi một từ tiếng Anh tương đương thường chỉ là 1 token.
Tokenizer thiên lệch về tiếng Anh: Do dữ liệu huấn luyện của BPE tokenizer chủ yếu là tiếng Anh, các từ/cụm từ tiếng Anh phổ biến được gộp thành các token dài (nguyên từ), còn tiếng Việt xuất hiện ít hơn trong dữ liệu huấn luyện nên bị tách vụn ở mức ký tự hoặc âm tiết.

→ Hệ quả: với cùng nội dung, văn bản tiếng Việt thường cần nhiều token hơn tiếng Anh, dẫn đến chi phí API cao hơn khi xử lý cùng một khối lượng thông tin.
```
---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
```
Streaming quan trọng nhất trong các ứng dụng tương tác thời gian thực nơi người dùng đang chờ trực tiếp trước màn hình, ví dụ như chatbot, trợ lý ảo, hay các công cụ soạn thảo có AI hỗ trợ — vì nó giúp giảm cảm giác chờ đợi bằng cách hiển thị từng phần câu trả lời ngay khi model sinh ra, thay vì bắt người dùng nhìn màn hình trống cho đến khi toàn bộ phản hồi (có thể mất vài giây đến hàng chục giây) hoàn tất. Ngược lại, non-streaming phù hợp hơn khi kết quả cần được xử lý tiếp trước khi hiển thị hoặc trả về — ví dụ như khi output là JSON có cấu trúc để hệ thống backend parse, khi cần kiểm duyệt/lọc nội dung trước khi cho người dùng thấy, khi chạy các tác vụ nền (batch processing) không có người dùng trực tiếp chờ, hoặc khi ứng dụng cần toàn bộ phản hồi để tính toán logic tiếp theo (ví dụ gọi thêm một API khác dựa trên toàn bộ kết quả).
```

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
```
So với delay cố định, exponential backoff có lợi thế là giãn cách thời gian chờ giữa các lần retry ngày càng dài (ví dụ 1s, 2s, 4s, 8s...), giúp giảm áp lực lên server đang quá tải một cách tăng dần thay vì giữ nguyên một mức tải lặp lại liên tục, đồng thời cho server nhiều thời gian hơn để phục hồi nếu tình trạng quá tải kéo dài. Nếu hàng nghìn client cùng retry với delay cố định giống nhau (ví dụ luôn chờ đúng 1 giây), tất cả các request sẽ đồng bộ lại và dồn về cùng một thời điểm mỗi lần retry — tạo thành các đợt tải lặp đi lặp lại, khiến server đã quá tải càng thêm quá tải và có thể không bao giờ phục hồi được; đây là lý do exponential backoff thường được kết hợp thêm với "jitter" (độ trễ ngẫu nhiên nhỏ) để phá vỡ sự đồng bộ này giữa các client.
```
---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
```
Tôi chọn persona: "Trợ lý lập trình kỹ thuật, thẳng thắn và chính xác, ưu tiên giải thích rõ bản chất vấn đề hơn là chỉ đưa ra code."
System prompt:
Bạn là một kỹ sư phần mềm senior, hỗ trợ tôi giải đáp thắc mắc khi lập trình.
Khi trả lời, hãy:
- Giải thích ngắn gọn NGUYÊN NHÂN gốc rễ của vấn đề trước khi đưa ra giải pháp,không chỉ dán code mà không giải thích.
- Nếu có nhiều cách giải quyết, nêu 2-3 lựa chọn kèm ưu/nhược điểm, để tôi tự quyết định thay vì chỉ đưa ra một đáp án duy nhất.
- Nếu code tôi đưa ra có lỗi tiềm ẩn, bug, hoặc vi phạm best practice, hãy chỉ ra thẳng thắn dù tôi không hỏi, không cần giữ thể diện.
- Dùng thuật ngữ kỹ thuật chính xác, không cần đơn giản hóa quá mức vì tôi đã có nền tảng lập trình.
- Nếu không chắc chắn (ví dụ hành vi phụ thuộc phiên bản thư viện, hệ điều hành), hãy nói rõ giả định của bạn thay vì khẳng định chắc nịch.
Trả lời bằng tiếng Việt, có thể xen tiếng Anh cho thuật ngữ kỹ thuật, code snippet nếu cần.
Giải thích 2 lựa chọn từ ngữ quan trọng:
- "Giải thích NGUYÊN NHÂN gốc rễ trước khi đưa ra giải pháp": Với lập trình cá nhân, mục tiêu không chỉ là "fix cho xong" mà là hiểu bản chất để tự xử lý các trường hợp tương tự sau này — nếu chỉ nhận code mà không hiểu tại sao, bạn sẽ phụ thuộc vào AI mãi mãi và không học được gì. Yêu cầu này ép model phải dạy thay vì chỉ làm hộ.
- "Nếu code có lỗi tiềm ẩn... hãy chỉ ra thẳng thắn dù tôi không hỏi": Đây là lựa chọn quan trọng để tránh model chỉ trả lời đúng câu hỏi hẹp mà bỏ qua vấn đề lớn hơn (ví dụ khi tôi hỏi "sửa lỗi syntax này" nhưng code có lỗ hổng bảo mật nghiêm trọng thì phải thông báo ngay cho tôi).
```
### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
```
Với persona trợ lý lập trình cá nhân, hạn chế lớn nhất là history chỉ giữ 3 lượt gần nhất.

Đề xuất cải thiện: Áp dụng chiến lược "sliding window + rolling summary" thay vì cắt cứng ở 3 lượt.

Cách triển khai ngắn gọn:

Giữ nguyên toàn bộ N lượt gần nhất (ví dụ 6-8 lượt) trong messages để đảm bảo độ chính xác của ngữ cảnh tức thời.
Khi vượt quá N lượt, không xóa thẳng các lượt cũ mà gọi thêm một lần API riêng để tóm tắt các lượt bị đẩy ra (ví dụ: "Đã thống nhất dùng FastAPI, database PostgreSQL, đang debug lỗi connection pool ở file db.py") thành 2-3 câu ngắn, rồi chèn tóm tắt này vào đầu system prompt như một đoạn "bối cảnh dự án" cố định.
Mỗi khi có thêm lượt mới đẩy cũ ra khỏi window, cập nhật lại đoạn tóm tắt (merge tóm tắt cũ + nội dung mới bị đẩy ra) để bối cảnh dài hạn không bị mất hoàn toàn dù vẫn tiết kiệm token so với giữ nguyên toàn bộ lịch sử.
Ưu điểm: vừa giữ được chi tiết chính xác ở các lượt gần (nơi cần độ chính xác code cao), vừa không mất hoàn toàn ngữ cảnh dài hạn (nơi chỉ cần biết "đã quyết định gì" chứ không cần nguyên văn).
```
---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
