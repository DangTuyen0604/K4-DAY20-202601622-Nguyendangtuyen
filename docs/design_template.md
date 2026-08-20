# Design Template

## Problem

Hệ thống nhận một câu hỏi nghiên cứu dài, truy xuất bằng chứng từ corpus offline, phân tích
độ tin cậy và tổng hợp câu trả lời có citation. Cùng query được chạy qua single-agent baseline
và multi-agent workflow để so sánh chất lượng, latency và cost.

## Why multi-agent?

Research report gồm các công việc khác nhau: retrieval, evidence assessment và writing.
Tách role giúp giữ handoff rõ, kiểm tra citation trước khi xuất bản và quan sát failure theo
từng bước. Multi-agent không mặc định tốt hơn; benchmark đo coordination overhead để quyết định.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Chọn bước tiếp theo và dừng an toàn | Shared state | Route | Loop hoặc route sai |
| Researcher | Truy xuất corpus và tạo notes có citation | Query | Sources + research notes | Không có nguồn/provider lỗi |
| Analyst | Đánh giá claim, nguồn, conflict và gaps | Sources + research notes | Analysis notes | Nhầm synthetic thành nguồn thật |
| Writer | Viết câu trả lời cuối và validate citation | Research + analysis | Final answer | Citation drift/hallucination |

## Shared state

- `request`: query, audience và giới hạn nguồn.
- `sources`: evidence và citation catalog dùng chung.
- `research_notes`: handoff Researcher → Analyst.
- `analysis_notes`: handoff Analyst → Writer.
- `final_answer`: output cuối.
- `route_history`, `iteration`: debug routing và chặn loop.
- `agent_results`: token/cost/output theo từng agent.
- `trace`, `errors`: observability và partial fallback.

## Routing policy

```text
START → Supervisor ─┬→ Researcher ─┐
                    ├→ Analyst ────┤
                    ├→ Writer ─────┤
                    └→ DONE        │
                         Supervisor←┘
```

Policy: thiếu research → Researcher; có research nhưng thiếu analysis → Analyst; có analysis
nhưng thiếu final answer → Writer; có final answer → done.

## Guardrails

- Max iterations: 6 mặc định, Supervisor buộc route `done` khi chạm giới hạn.
- Timeout: 60 giây cho workflow và từng API client.
- Retry: tối đa 3 lần với timeout, connection error và rate limit.
- Fallback: offline retrieval và trả partial analysis/research notes nếu workflow fail.
- Validation: Pydantic input/state; Writer từ chối citation không thuộc source catalog.

## Benchmark plan

Ba query trong `configs/lab_default.yaml` được chạy cho cả baseline và multi-agent. Metrics:
latency, provider cost, quality 0-10, citation coverage và failure rate. Kỳ vọng multi-agent
có grounding/citation tốt hơn nhưng latency và cost cao hơn. Số liệu thực nằm trong
`reports/benchmark_report.md`.
