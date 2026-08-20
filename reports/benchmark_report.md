# Benchmark Report

| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| baseline-1 | 10.78 | 0.0004 | 7.5 |  | 0% | query=Research GraphRAG state-of-the-art and write a 500-word summary; words=545; errors=0 |
| multi-agent-1 | 32.64 | 0.0017 | 10.0 | 100% | 0% | query=Research GraphRAG state-of-the-art and write a 500-word summary; words=421; errors=0 |
| baseline-2 | 11.67 | 0.0005 | 7.5 |  | 0% | query=Compare single-agent and multi-agent workflows for customer support; words=582; errors=0 |
| multi-agent-2 | 27.55 | 0.0015 | 10.0 | 100% | 0% | query=Compare single-agent and multi-agent workflows for customer support; words=368; errors=0 |
| baseline-3 | 14.33 | 0.0004 | 7.5 |  | 0% | query=Summarize production guardrails for LLM agents; words=442; errors=0 |
| multi-agent-3 | 26.01 | 0.0014 | 10.0 | 100% | 0% | query=Summarize production guardrails for LLM agents; words=377; errors=0 |

## Methodology

Both systems receive the same queries. Latency is wall-clock time and cost comes 
from provider usage metadata. Quality uses a deterministic 0-10 rubric covering 
completeness, structure, limitations, recommendations, errors, and citations. 
Citation coverage is cited corpus labels divided by retrieved labels.

## Aggregate comparison

- Baseline: average latency 12.26s, average cost $0.000431, average quality 7.5/10.
- Multi-agent: average latency 28.73s, average cost $0.001549, average quality 10.0/10.

## Interpretation

Multi-agent latency was 2.34x baseline and cost was 3.59x baseline. The quality/citation gain therefore comes with measurable coordination overhead.
The quality score is an automated reproducible heuristic, not a substitute for 
the peer-review rubric or a human factuality review.

## Trace

https://smith.langchain.com/o/d41447c4-7530-4d7e-8e09-00fb46594227/projects/p/b3e6b54c-b88f-4711-9861-48a62695495c/trace/01a01e7b-d5a9-75a1-9f28-644dfe1c888d/run/01a01e7b-d5a9-75a1-9f28-644dfe1c888d?poll=true&start_time=2026-08-20T09%3A23%3A47.241671

## Failure mode analysis

The main observed risk is coordination overhead: multi-agent execution makes three 
LLM calls and passes growing context between roles, increasing latency and cost. 
Citation drift is another risk; Writer validates every bracketed citation against 
the retrieved source catalog. Transient provider failures are retried centrally, 
offline retrieval avoids search outages, and workflow timeout/max-iteration guards 
return the best available partial result instead of looping indefinitely.
