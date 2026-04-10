# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Phung Huu Phu
**Nhóm:** []
**Ngày:** 2026-04-10

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là hai vector embedding có hướng gần giống nhau, tức là hai câu có ngữ nghĩa tương đồng. Giá trị càng gần 1 thì mức tương đồng ngữ nghĩa càng cao.

**Ví dụ HIGH similarity:**
- Sentence A: "How can I reset my account password?"
- Sentence B: "What are the steps to change my password?"
- Tại sao tương đồng: Cả hai cùng hỏi về cùng một tác vụ trong account settings, khác nhau chủ yếu ở cách diễn đạt.

**Ví dụ LOW similarity:**
- Sentence A: "Boil water for 8 minutes before adding noodles."
- Sentence B: "Neural networks use gradient descent for optimization."
- Tại sao khác: Hai câu thuộc hai domain hoàn toàn khác nhau (nấu ăn vs machine learning), không có ý nghĩa chung.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity tập trung vào hướng của vector (ngữ nghĩa), ít bị ảnh hưởng bởi độ lớn vector. Với text embeddings, hướng thường quan trọng hơn khoảng cách tuyệt đối theo tọa độ, nên cosine phù hợp hơn Euclidean distance.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Trình bày phép tính: `num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))`
> `= ceil((10000 - 50) / (500 - 50))`
> `= ceil(9950 / 450) = ceil(22.111...)`
> Đáp án: **23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Với overlap=100: `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = 25` chunks, tức là tăng từ 23 lên 25. Overlap lớn hơn giúp giữ ngữ cảnh liên tục giữa các chunk và giảm rủi ro mất ý ở ranh giới chunk.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Vietnamese legal/policy documents (thủ tục hành chính, tài chính, tư pháp)

**Tại sao nhóm chọn domain này?**
> Nhóm chọn domain pháp lý vì dữ liệu có cấu trúc điều/khoản rõ ràng và nhiều chi tiết định nghĩa phạm vi áp dụng, rất phù hợp để đánh giá chunking + retrieval. Domain này cũng có nhu cầu thực tế cao (tra cứu quy định, đối chiếu nguồn), nên benchmark query dễ xác thực bằng chứng từ văn bản.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | 66_7_2025_NQ-CP_681135.md | law_data/processed | 13520 | source_doc_id, source_path, chunk_index |
| 2 | 167_2012_TT-BTC_149305.md | law_data/processed | 8218 | source_doc_id, source_path, chunk_index |
| 3 | 15_2026_TT-BTC_696722.md | law_data/processed | 7351 | source_doc_id, source_path, chunk_index |
| 4 | 19_2014_TT-BTP_249771.md | law_data/processed | 22143 | source_doc_id, source_path, chunk_index |
| 5 | 25_2014_TT-BTP_262503.md | law_data/processed | 22549 | source_doc_id, source_path, chunk_index |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| source_doc_id | string | 66_7_2025_NQ-CP_681135 | Filter đúng văn bản mục tiêu khi query nêu rõ số hiệu nghị quyết/thông tư |
| source_path | string | processed/66_7_2025_NQ-CP_681135.md | Truy vết nguồn và hỗ trợ kiểm tra grounding |
| chunk_index | integer | 12 | Xác định vị trí chunk trong văn bản để đối chiếu điều/khoản |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| `data/rag_system_design.md` | FixedSizeChunker (`fixed_size`) | 14 | 189.36 | Trung bình, có cắt giữa ý ở vài đoạn |
| `data/rag_system_design.md` | SentenceChunker (`by_sentences`) | 5 | 476.00 | Tốt về giữ ngữ nghĩa câu, nhưng chunk khá dài |
| `data/rag_system_design.md` | RecursiveChunker (`recursive`) | 20 | 117.65 | Tốt, cân bằng giữa độ ngắn và giữ mạch nội dung |

### Strategy Của Tôi

**Loại:** RecursiveChunker (tuned từ built-in strategy)

**Mô tả cách hoạt động:**
> Strategy này chia văn bản theo thứ tự separator ưu tiên: `\n\n` -> `\n` -> `. ` -> ` ` -> cắt cứng theo ký tự. Khi một đoạn vẫn vượt `chunk_size`, thuật toán đệ quy với separator mức chi tiết hơn. Cách này ưu tiên giữ cấu trúc tự nhiên của tài liệu (đoạn, dòng, câu) trước khi phải cắt thô. Vì vậy chunk thường vừa đủ ngắn để retrieve tốt, nhưng không làm mất quá nhiều ngữ cảnh.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Các tài liệu trong lab có cấu trúc markdown và đoạn văn rõ ràng, nên recursive chunking tận dụng tốt ranh giới tự nhiên của văn bản. So với fixed-size, strategy này giảm hiện tượng cắt giữa câu; so với sentence-only, nó tránh tạo chunk quá dài khi nhiều câu dính liền trong một đoạn.

**Code snippet (nếu custom):**
```python
# Paste implementation here
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| `data/rag_system_design.md` | best baseline (SentenceChunker) | 5 | 476.00 | Context tốt nhưng chunk lớn, độ chính xác top-1 có thể giảm |
| `data/rag_system_design.md` | **của tôi** (RecursiveChunker) | 20 | 117.65 | Cân bằng tốt giữa precision và coherence trong retrieval |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | RecursiveChunker | 10/10 (top-3 recall: 5/5) | Chunk tự nhiên, dễ trace nguồn | Có thể tạo nhiều chunk nhỏ nếu dữ liệu nhiễu |
| Thông | SentenceChunker | 10/10 (top-3 recall: 5/5) | Giữ mạch câu tốt, dễ đọc | Chunk dài nên đôi lúc giảm precision top-1 |
| Hùng | FixedSizeChunker | 10/10 (top-3 recall: 5/5) | Đơn giản, nhanh, ổn định | Dễ cắt ngang ý pháp lý ở ranh giới chunk |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Sau benchmark 5 câu hỏi, strategy RecursiveChunker cho kết quả tốt và ổn định (top-3 recall 5/5, các truy vấn đều lấy được chunk liên quan). Strategy này phù hợp với dữ liệu văn bản pháp lý vì tận dụng ranh giới đoạn/câu tốt hơn so với cắt cứng fixed-size.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Tôi dùng regex `(?<=[.!?])\s+` để tách câu dựa trên dấu kết thúc câu và khoảng trắng phía sau. Sau đó `strip()` và loại bỏ câu rỗng để tránh chunk rác. Cuối cùng gom các câu theo `max_sentences_per_chunk` để tạo chunk ổn định.

**`RecursiveChunker.chunk` / `_split`** — approach:
> `chunk()` kiểm tra text rỗng và chuẩn hóa separator list trước khi gọi `_split()`. `_split()` có base case: text rỗng, text đã <= `chunk_size`, hoặc hết separator thì fallback cắt cứng fixed-size. Ở mỗi tầng, thuật toán thử separator hiện tại để merge các phần vừa kích thước; phần quá dài sẽ đệ quy xuống separator chi tiết hơn.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Tôi chuẩn hóa mỗi tài liệu thành record gồm `id`, `content`, `metadata`, `embedding`, trong đó `metadata` luôn có thêm `doc_id` để quản lý xóa/lọc. Nếu có ChromaDB thì add/query trên collection; nếu không thì dùng in-memory list. `search()` embed câu hỏi rồi xếp hạng theo dot product (với mock embedding đã normalize thì tương đương cosine ranking).

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter()` thực hiện metadata pre-filter trước, rồi mới tính similarity trên tập đã lọc để tăng precision cho các query có ngữ cảnh cụ thể. `delete_document()` xóa toàn bộ records có `metadata['doc_id'] == doc_id`; trả về boolean cho biết có xóa được hay không.

### KnowledgeBaseAgent

**`answer`** — approach:
> Tôi dùng RAG flow chuẩn: retrieve top-k chunks -> ghép context -> gọi `llm_fn(prompt)`. Prompt yêu cầu model chỉ trả lời dựa trên context và nói rõ khi context thiếu. Context được đánh số theo từng chunk kèm score để dễ trace nguồn khi đánh giá grounding.

### Test Results

```
# Paste output of: pytest tests/ -v
================================ test session starts ================================
collected 42 items

tests/test_solution.py ..........................................           [100%]

============================= 42 passed in 0.39s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Python is great for data science. | Python works well for data analysis. | high | -0.0119 | Sai |
| 2 | Reset your password in account settings. | How to change account password? | high | 0.0964 | Đúng |
| 3 | The cat sleeps on the sofa. | Stock markets closed higher today. | low | -0.0725 | Đúng |
| 4 | Machine learning models need training data. | Neural networks learn patterns from datasets. | high | 0.0779 | Đúng |
| 5 | Boil water for 8 minutes. | Quantum entanglement is non-local. | low | -0.0240 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Pair 1 bất ngờ nhất vì hai câu có nghĩa gần nhau nhưng score hơi âm. Điều này cho thấy backend mock embedding deterministic không phản ánh semantics mạnh như model embedding thật; vì vậy score chỉ phù hợp để test logic pipeline, không dùng để kết luận chất lượng ngữ nghĩa thực tế.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Nghị quyết về cắt giảm thủ tục hành chính dựa trên dữ liệu quy định phạm vi điều chỉnh gì? | Nghị quyết quy định việc thay thế hoặc cắt giảm thành phần hồ sơ thủ tục hành chính bằng khai thác/sử dụng thông tin từ cơ sở dữ liệu quốc gia và cơ sở dữ liệu chuyên ngành. |
| 2 | Theo Nghị quyết 66, thông tin trong Cơ sở dữ liệu quốc gia về dân cư dùng để thay thế những giấy tờ nào? | Thay thế CMND/CCCD/Thẻ Căn cước/Giấy chứng nhận căn cước và thông tin về cư trú. |
| 3 | Thông tư 167/2012/TT-BTC quy định kinh phí bảo đảm kiểm soát thủ tục hành chính tại cấp tỉnh do nguồn nào bảo đảm? | Do ngân sách địa phương bảo đảm. |
| 4 | Thông tư 15/2026/TT-BTC có áp dụng để xác định nghĩa vụ thuế không? | Không áp dụng để xác định nghĩa vụ thuế với ngân sách nhà nước. |
| 5 | Tài sản mã hóa của khách hàng lưu ký có được ghi nhận là tài sản của tổ chức cung cấp dịch vụ không? | Không. Tài sản mã hóa lưu ký của khách hàng không được ghi nhận là tài sản của tổ chức cung cấp dịch vụ. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Nghị quyết về cắt giảm thủ tục hành chính dựa trên dữ liệu quy định phạm vi điều chỉnh gì? | NGHỊ QUYẾT quy định cắt giảm, đơn giản hóa thủ tục hành chính dựa trên dữ liệu; nêu căn cứ pháp lý và phạm vi áp dụng | 0.7153 | Yes | Trả lời đúng trọng tâm về phạm vi điều chỉnh: thay thế/cắt giảm thành phần hồ sơ bằng khai thác dữ liệu |
| 2 | Theo Nghị quyết 66, thông tin trong Cơ sở dữ liệu quốc gia về dân cư dùng để thay thế những giấy tờ nào? | Đoạn quy định thông tin trên các CSDL chuyên ngành được khai thác để thay thế giấy tờ tương ứng trong hồ sơ | 0.6268 | Yes | Trả lời đúng nhóm giấy tờ được thay thế (CMND/CCCD/Thẻ căn cước và thông tin cư trú) |
| 3 | Thông tư 167/2012/TT-BTC quy định kinh phí bảo đảm kiểm soát thủ tục hành chính tại cấp tỉnh do nguồn nào bảo đảm? | Điều 2 của Thông tư 167 nêu nguồn kinh phí thực hiện cho trung ương và địa phương | 0.5899 | Yes | Kết luận đúng: kinh phí do ngân sách địa phương bảo đảm |
| 4 | Thông tư 15/2026/TT-BTC có áp dụng để xác định nghĩa vụ thuế không? | Điều 1 (phạm vi điều chỉnh) của Thông tư 15/2026/TT-BTC về nguyên tắc kế toán trong thí điểm thị trường tài sản mã hóa | 0.5903 | Yes | Trả lời đúng: không dùng để xác định nghĩa vụ thuế với NSNN |
| 5 | Tài sản mã hóa của khách hàng lưu ký có được ghi nhận là tài sản của tổ chức cung cấp dịch vụ không? | Điểm b nêu rõ tài sản mã hóa lưu ký của khách hàng không được ghi nhận là tài sản của tổ chức cung cấp dịch vụ | 0.6928 | Yes | Trả lời đúng: tài sản lưu ký của khách hàng không ghi nhận là tài sản của tổ chức cung cấp dịch vụ |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tôi học được rằng metadata filtering theo `source_doc_id` cải thiện rõ độ chính xác ở các câu hỏi pháp lý cụ thể. Khi query có nêu số nghị quyết/thông tư, việc lọc trước giúp tránh lấy nhầm văn bản có chủ đề tương tự.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Nhóm khác thiết kế benchmark theo nhiều kiểu câu hỏi (fact, phạm vi áp dụng, ngoại lệ), nên đánh giá retrieval toàn diện hơn. Cách gắn gold answer với evidence cụ thể giúp so sánh strategy khách quan và thuyết phục hơn.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ chuẩn hóa metadata kỹ hơn ngay từ đầu (ví dụ `department`, `language`, `doc_type`, `updated_at`) để hỗ trợ filtering tốt hơn. Ngoài ra, tôi muốn benchmark song song cả OpenAI embeddings và local embeddings trên cùng query set để kiểm tra độ ổn định retrieval.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 14 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **88 / 100 (tự đánh giá tạm thời)** |
