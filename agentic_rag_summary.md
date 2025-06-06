
# 🧠 Agentic RAG 개요 및 구현 정리

## 1. 개념 요약

**Agentic RAG**는 전통적인 RAG(Retrieval-Augmented Generation)에 **에이전트 기능**을 결합하여,  
문제를 능동적으로 해결하고 반복적으로 작업을 수행할 수 있도록 확장한 구조입니다.

---

## 2. 전통 RAG vs Agentic RAG

| 항목             | 기존 RAG                             | Agentic RAG                                    |
|------------------|--------------------------------------|------------------------------------------------|
| 처리 방식        | 질문 → 검색 → 응답                   | 질문 → 계획 → 검색/도구 실행 → 응답 반복       |
| 능동성           | 없음                                 | 있음 (에이전트가 스스로 계획하고 실행함)       |
| 반복 가능성      | 단일 스텝                            | 다중 스텝 / 반복적 실행                        |
| 활용 시나리오    | Q&A, FAQ                             | 문서 요약, 데이터 처리, 툴 조합 문제 해결 등   |

---

## 3. 구성 요소

- `Agent` : 문제를 계획하고 툴을 선택하여 작업을 실행
- `Tool` : PDF 로더, Vector DB 검색기, 요약기 등
- `Retriever` : RAG 검색 수행
- `LLM` : 텍스트 생성 (요약, 응답 등)
- `Memory` : 대화 이력 또는 문맥 유지 (선택)

---

## 4. 워크플로우 다이어그램

```mermaid
flowchart TD
    U[사용자 질문: "이 PDF에서 핵심 내용을 요약해줘"]
    A[Agent: 작업 계획 세우기]
    T1[도구1: PDF Loader → 텍스트 추출]
    T2[도구2: Vector DB에 인덱싱]
    T3[도구3: RAG 검색 → 관련 문단 선택]
    T4[도구4: 요약 생성 LLM 호출]
    A2[Agent: 요약 결과 정리]
    Out[결과 응답]

    U --> A --> T1 --> T2 --> T3 --> T4 --> A2 --> Out
```

---

## 5. 코드 예시 (LangChain)

```python
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# 1. PDF 문서 로드
loader = PyPDFLoader("example.pdf")
docs = loader.load_and_split()

# 2. 벡터스토어 생성
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# 3. RAG 체인 정의
rag_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    retriever=retriever
)

# 4. 도구 정의
tools = [
    Tool(
        name="PDF Summary Tool",
        func=rag_chain.run,
        description="PDF 문서로부터 질문에 대한 요약 정보를 제공함"
    )
]

# 5. Agent 초기화
agent = initialize_agent(
    tools=tools,
    llm=ChatOpenAI(temperature=0),
    agent="zero-shot-react-description",
    verbose=True
)

# 6. 실행
result = agent.run("이 문서에서 핵심 요점을 요약해줘.")
print(result)
```

---

## 6. 확장 가능 예

- ✅ 웹 검색 도구와 연동하여 최신 정보 수집
- ✅ 코드 실행기 도구 추가로 데이터 분석 자동화
- ✅ 대화형 UI (Streamlit, Dash)와 연결

---

## 7. 참고 기술

- LangChain: https://python.langchain.com/
- FAISS: Facebook AI Similarity Search
- OpenAI Embeddings & Chat Models
- PyPDFLoader: LangChain용 PDF 로더

---
