import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from pydantic import BaseModel, Field
from pypdf import PdfReader


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
SAMPLE_RESUME = DATA_DIR / "sample_resume.pdf"
SAMPLE_JD = DATA_DIR / "sample_jd.pdf"

load_dotenv(BASE_DIR / ".env")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEMO_MODE = os.getenv("INTERVIEWER_DEMO_MODE", "true").lower() == "true"

SESSIONS: dict[str, dict[str, Any]] = {}

app = FastAPI(
    title="AI Interviewer Agent",
    description="LangGraph-based AI interviewer with resume parsing, JD analysis, memory, scoring, and follow-up questions.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class InterviewState(TypedDict, total=False):
    session_id: str
    resume_text: str
    jd_text: str
    resume_profile: dict[str, Any]
    jd_profile: dict[str, Any]
    questions: list[dict[str, Any]]
    current_question: dict[str, Any]
    answer: str
    qa_history: list[dict[str, Any]]
    score: dict[str, Any]
    next_question: dict[str, Any] | None
    report: str
    tool_calls: list[dict[str, Any]]


class StartInterviewRequest(BaseModel):
    resume_text: str = Field(..., description="Plain text resume content.")
    jd_text: str = Field(..., description="Plain text job description content.")
    session_id: str | None = None


class AnswerRequest(BaseModel):
    session_id: str
    answer: str


class ReportRequest(BaseModel):
    session_id: str


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    extracted_chars: int


class StartInterviewResponse(BaseModel):
    session_id: str
    question: dict[str, Any]
    questions: list[dict[str, Any]]
    resume_profile: dict[str, Any]
    jd_profile: dict[str, Any]
    workflow: list[str]


class AnswerResponse(BaseModel):
    session_id: str
    question: dict[str, Any]
    answer: str
    score: dict[str, Any]
    next_question: dict[str, Any] | None
    memory: list[dict[str, Any]]
    finished: bool


class ReportResponse(BaseModel):
    session_id: str
    report: str
    average_score: float
    rounds: int


def read_pdf_text(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())
    return "\n\n".join(pages)


async def read_upload_text(file: UploadFile) -> str:
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    content = await file.read()

    if suffix == ".pdf":
        target_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{filename}"
        target_path.write_bytes(content)
        return read_pdf_text(target_path)

    if suffix in {".txt", ".md"}:
        return content.decode("utf-8", errors="ignore")

    raise HTTPException(status_code=400, detail="Only PDF, TXT, and MD files are supported.")


def ensure_session(session_id: str | None = None) -> tuple[str, dict[str, Any]]:
    resolved_id = session_id or uuid.uuid4().hex[:12]
    if resolved_id not in SESSIONS:
        SESSIONS[resolved_id] = {
            "session_id": resolved_id,
            "created_at": datetime.utcnow().isoformat(),
            "resume_text": "",
            "jd_text": "",
            "resume_profile": {},
            "jd_profile": {},
            "questions": [],
            "qa_history": [],
            "question_cursor": 0,
            "current_question": None,
            "tool_calls": [],
        }
    return resolved_id, SESSIONS[resolved_id]


def keywords_from_text(text: str, candidates: list[str]) -> list[str]:
    lowered = text.lower()
    return [keyword for keyword in candidates if keyword.lower() in lowered]


def compact_lines(text: str, limit: int = 8) -> list[str]:
    lines = [line.strip(" -\t") for line in text.splitlines() if line.strip()]
    return lines[:limit]


def openai_json(system_prompt: str, user_prompt: str, fallback: dict[str, Any]) -> dict[str, Any]:
    if DEMO_MODE or not os.getenv("OPENAI_API_KEY"):
        return fallback

    client = OpenAI()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    try:
        return json.loads(response.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return fallback


@tool
def parse_resume_tool(resume_text: str) -> dict[str, Any]:
    """Parse resume text into structured candidate profile."""
    skill_candidates = [
        "Python",
        "FastAPI",
        "LangGraph",
        "LangChain",
        "OpenAI",
        "RAG",
        "ChromaDB",
        "SQL",
        "JavaScript",
        "Docker",
        "Git",
        "Agent",
        "Prompt Engineering",
    ]
    skills = keywords_from_text(resume_text, skill_candidates)
    year_match = re.search(r"(\d+)\s*(?:年|years?)", resume_text, flags=re.IGNORECASE)
    project_lines = [
        line
        for line in compact_lines(resume_text, limit=20)
        if any(word.lower() in line.lower() for word in ["项目", "project", "rag", "agent", "fastapi"])
    ][:5]
    name = compact_lines(resume_text, limit=1)[0] if compact_lines(resume_text, limit=1) else "Candidate"

    fallback = {
        "name": name,
        "skills": skills or ["Python", "FastAPI", "OpenAI API"],
        "experience_years": int(year_match.group(1)) if year_match else None,
        "projects": project_lines,
        "summary": "Candidate profile extracted from resume text.",
    }
    return openai_json(
        "Extract a resume profile as JSON with keys: name, skills, experience_years, projects, summary.",
        resume_text,
        fallback,
    )


@tool
def analyze_jd_tool(jd_text: str) -> dict[str, Any]:
    """Analyze job description text into structured role requirements."""
    skill_candidates = [
        "Python",
        "FastAPI",
        "LangGraph",
        "LangChain",
        "OpenAI",
        "RAG",
        "Agent",
        "Tool Calling",
        "Memory",
        "Workflow",
        "Prompt Engineering",
        "Docker",
        "SQL",
    ]
    skills = keywords_from_text(jd_text, skill_candidates)
    lines = compact_lines(jd_text, limit=12)
    responsibilities = [
        line for line in lines if any(word in line.lower() for word in ["负责", "构建", "build", "develop", "设计"])
    ][:5]
    requirements = [
        line for line in lines if any(word in line.lower() for word in ["要求", "熟悉", "experience", "掌握", "了解"])
    ][:6]

    fallback = {
        "role": lines[0] if lines else "AI Agent Engineer",
        "skills": skills or ["Python", "FastAPI", "LangGraph", "OpenAI"],
        "responsibilities": responsibilities,
        "requirements": requirements or lines[:5],
        "summary": "JD profile extracted from job description text.",
    }
    return openai_json(
        "Extract a job description profile as JSON with keys: role, skills, responsibilities, requirements, summary.",
        jd_text,
        fallback,
    )


@tool
def score_answer_tool(question: str, answer: str, jd_skills: list[str], resume_skills: list[str]) -> dict[str, Any]:
    """Score an interview answer and return practical feedback."""
    answer_lower = answer.lower()
    matched_skills = [skill for skill in jd_skills if skill.lower() in answer_lower]
    resume_hits = [skill for skill in resume_skills if skill.lower() in answer_lower]
    length_score = min(len(answer.strip()) / 260, 1.0) * 35
    skill_score = (len(matched_skills) / max(len(jd_skills), 1)) * 35
    evidence_score = 20 if any(word in answer for word in ["项目", "实现", "负责", "优化", "上线", "指标"]) else 8
    clarity_score = 10 if len(answer.splitlines()) <= 8 else 6
    total = round(min(length_score + skill_score + evidence_score + clarity_score, 100), 1)

    strengths = []
    if matched_skills:
        strengths.append(f"提到了岗位相关技能：{', '.join(matched_skills[:4])}")
    if resume_hits:
        strengths.append(f"能和简历技能关联：{', '.join(resume_hits[:4])}")
    if evidence_score >= 20:
        strengths.append("回答里有项目或行动细节")
    if not strengths:
        strengths.append("回答方向基本相关，但信息还偏少")

    improvements = []
    if len(answer.strip()) < 120:
        improvements.append("回答太短，可以补充项目背景、你的动作和结果")
    if not matched_skills:
        improvements.append("还没有明显回应 JD 里的关键技能")
    if evidence_score < 20:
        improvements.append("建议加一个具体例子，说明你怎么做、结果是什么")
    if not improvements:
        improvements.append("可以继续补充量化结果，比如耗时、准确率、召回效果或用户反馈")

    return {
        "score": total,
        "matched_skills": matched_skills,
        "strengths": strengths,
        "improvements": improvements,
    }


def resume_parser_node(state: InterviewState) -> InterviewState:
    profile = parse_resume_tool.invoke({"resume_text": state["resume_text"]})
    return {
        "resume_profile": profile,
        "tool_calls": state.get("tool_calls", []) + [{"tool": "parse_resume_tool", "status": "ok"}],
    }


def jd_analyzer_node(state: InterviewState) -> InterviewState:
    profile = analyze_jd_tool.invoke({"jd_text": state["jd_text"]})
    return {
        "jd_profile": profile,
        "tool_calls": state.get("tool_calls", []) + [{"tool": "analyze_jd_tool", "status": "ok"}],
    }


def generate_questions_node(state: InterviewState) -> InterviewState:
    resume_profile = state["resume_profile"]
    jd_profile = state["jd_profile"]
    resume_skills = resume_profile.get("skills", [])
    jd_skills = jd_profile.get("skills", [])
    shared_skills = [skill for skill in jd_skills if skill in resume_skills]
    focus_skills = shared_skills or jd_skills or resume_skills or ["项目经验"]

    fallback_questions = [
        {
            "id": "q1",
            "type": "base",
            "focus": "匹配度",
            "question": "先用 2 分钟介绍一下你自己，并重点说说你和这个岗位最匹配的经历。",
        },
        {
            "id": "q2",
            "type": "base",
            "focus": focus_skills[0],
            "question": f"你简历里和 {focus_skills[0]} 相关的项目，具体是怎么做的？",
        },
        {
            "id": "q3",
            "type": "base",
            "focus": "工程落地",
            "question": "如果让你把这个能力做成一个可运行服务，你会怎么设计接口、状态和异常处理？",
        },
        {
            "id": "q4",
            "type": "base",
            "focus": "复盘能力",
            "question": "你做过的项目里，哪一部分最容易出问题？你后来怎么改的？",
        },
        {
            "id": "q5",
            "type": "base",
            "focus": "岗位理解",
            "question": "结合 JD，你觉得这个岗位最需要你证明的能力是什么？",
        },
    ]

    llm_result = openai_json(
        "Generate 5 interview questions as JSON: {\"questions\": [{\"id\", \"type\", \"focus\", \"question\"}]}",
        json.dumps({"resume": resume_profile, "jd": jd_profile}, ensure_ascii=False),
        {"questions": fallback_questions},
    )
    questions = llm_result.get("questions") or fallback_questions
    current_question = questions[0]
    return {"questions": questions, "current_question": current_question}


def user_answer_node(state: InterviewState) -> InterviewState:
    return {"answer": state["answer"].strip()}


def scoring_node(state: InterviewState) -> InterviewState:
    score = score_answer_tool.invoke(
        {
            "question": state["current_question"]["question"],
            "answer": state["answer"],
            "jd_skills": state["jd_profile"].get("skills", []),
            "resume_skills": state["resume_profile"].get("skills", []),
        }
    )
    return {
        "score": score,
        "tool_calls": state.get("tool_calls", []) + [{"tool": "score_answer_tool", "status": "ok"}],
    }


def followup_node(state: InterviewState) -> InterviewState:
    score = state["score"]["score"]
    current = state["current_question"]
    questions = state["questions"]
    history = state.get("qa_history", [])
    answered_base_count = sum(1 for item in history if item["question"].get("type") == "base")
    answered_current_followup = current.get("type") == "followup"

    if score < 75 and not answered_current_followup:
        next_question = {
            "id": f"followup-{len(history) + 1}",
            "type": "followup",
            "focus": current.get("focus", "细节追问"),
            "question": f"这个回答还可以再具体一点。请你围绕「{current.get('focus', '项目细节')}」补充一个真实项目例子：背景是什么，你做了什么，结果怎么衡量？",
        }
        return {"next_question": next_question}

    next_index = answered_base_count + 1 if not answered_current_followup else answered_base_count
    if next_index < len(questions):
        return {"next_question": questions[next_index]}

    return {"next_question": None}


def build_setup_graph():
    graph = StateGraph(InterviewState)
    graph.add_node("resume_parser", resume_parser_node)
    graph.add_node("jd_analyzer", jd_analyzer_node)
    graph.add_node("question_generator", generate_questions_node)
    graph.add_edge(START, "resume_parser")
    graph.add_edge("resume_parser", "jd_analyzer")
    graph.add_edge("jd_analyzer", "question_generator")
    graph.add_edge("question_generator", END)
    return graph.compile()


def build_answer_graph():
    graph = StateGraph(InterviewState)
    graph.add_node("user_answer", user_answer_node)
    graph.add_node("scoring", scoring_node)
    graph.add_node("followup", followup_node)
    graph.add_edge(START, "user_answer")
    graph.add_edge("user_answer", "scoring")
    graph.add_edge("scoring", "followup")
    graph.add_edge("followup", END)
    return graph.compile()


SETUP_GRAPH = build_setup_graph()
ANSWER_GRAPH = build_answer_graph()


def generate_report(session: dict[str, Any]) -> tuple[str, float]:
    history = session.get("qa_history", [])
    if not history:
        return "还没有面试记录，先完成至少一轮问答。", 0.0

    scores = [item["score"]["score"] for item in history]
    average = round(sum(scores) / len(scores), 1)
    strengths = []
    improvements = []
    for item in history:
        strengths.extend(item["score"].get("strengths", []))
        improvements.extend(item["score"].get("improvements", []))

    unique_strengths = list(dict.fromkeys(strengths))[:4]
    unique_improvements = list(dict.fromkeys(improvements))[:4]
    role = session.get("jd_profile", {}).get("role", "目标岗位")

    report = [
        f"面试复盘报告：{role}",
        "",
        f"完成轮次：{len(history)}",
        f"平均分：{average}",
        "",
        "表现比较好的地方：",
        *[f"- {item}" for item in unique_strengths],
        "",
        "下一步建议：",
        *[f"- {item}" for item in unique_improvements],
        "",
        "我会优先建议你把回答改成 STAR 结构：场景、任务、行动、结果。尤其是项目经历，不要只说用了什么技术，要说你解决了什么问题。",
    ]
    return "\n".join(report), average


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/sample")
def sample():
    return {
        "resume_text": read_pdf_text(SAMPLE_RESUME),
        "jd_text": read_pdf_text(SAMPLE_JD),
    }


@app.get("/workflow")
def workflow():
    return {
        "nodes": [
            "resume_parser",
            "jd_analyzer",
            "question_generator",
            "user_answer",
            "scoring",
            "followup",
            "report",
        ],
        "edges": [
            ["START", "resume_parser"],
            ["resume_parser", "jd_analyzer"],
            ["jd_analyzer", "question_generator"],
            ["question_generator", "question"],
            ["question", "user_answer"],
            ["user_answer", "scoring"],
            ["scoring", "followup"],
            ["followup", "question_or_report"],
        ],
    }


@app.post("/upload_resume", response_model=UploadResponse)
async def upload_resume(file: UploadFile = File(...), session_id: str | None = Form(default=None)):
    resolved_id, session = ensure_session(session_id)
    text = await read_upload_text(file)
    session["resume_text"] = text
    return UploadResponse(session_id=resolved_id, filename=file.filename or "resume", extracted_chars=len(text))


@app.post("/upload_jd", response_model=UploadResponse)
async def upload_jd(file: UploadFile = File(...), session_id: str | None = Form(default=None)):
    resolved_id, session = ensure_session(session_id)
    text = await read_upload_text(file)
    session["jd_text"] = text
    return UploadResponse(session_id=resolved_id, filename=file.filename or "jd", extracted_chars=len(text))


@app.post("/start_interview", response_model=StartInterviewResponse)
def start_interview(request: StartInterviewRequest):
    if not request.resume_text.strip() or not request.jd_text.strip():
        raise HTTPException(status_code=400, detail="Resume text and JD text are required.")

    session_id, session = ensure_session(request.session_id)
    initial_state: InterviewState = {
        "session_id": session_id,
        "resume_text": request.resume_text.strip(),
        "jd_text": request.jd_text.strip(),
        "tool_calls": [],
    }
    result = SETUP_GRAPH.invoke(initial_state)

    session.update(
        {
            "resume_text": request.resume_text.strip(),
            "jd_text": request.jd_text.strip(),
            "resume_profile": result["resume_profile"],
            "jd_profile": result["jd_profile"],
            "questions": result["questions"],
            "current_question": result["current_question"],
            "qa_history": [],
            "tool_calls": result.get("tool_calls", []),
        }
    )

    return StartInterviewResponse(
        session_id=session_id,
        question=result["current_question"],
        questions=result["questions"],
        resume_profile=result["resume_profile"],
        jd_profile=result["jd_profile"],
        workflow=["resume_parser", "jd_analyzer", "question_generator"],
    )


@app.post("/answer", response_model=AnswerResponse)
def answer(request: AnswerRequest):
    session = SESSIONS.get(request.session_id)
    if not session or not session.get("current_question"):
        raise HTTPException(status_code=404, detail="Interview session not found. Start an interview first.")
    if not request.answer.strip():
        raise HTTPException(status_code=400, detail="Answer is required.")

    state: InterviewState = {
        "session_id": request.session_id,
        "resume_profile": session["resume_profile"],
        "jd_profile": session["jd_profile"],
        "questions": session["questions"],
        "current_question": session["current_question"],
        "qa_history": session["qa_history"],
        "answer": request.answer.strip(),
        "tool_calls": session.get("tool_calls", []),
    }
    result = ANSWER_GRAPH.invoke(state)

    memory_item = {
        "question": session["current_question"],
        "answer": request.answer.strip(),
        "score": result["score"],
        "created_at": datetime.utcnow().isoformat(),
    }
    session["qa_history"].append(memory_item)
    session["tool_calls"] = result.get("tool_calls", session.get("tool_calls", []))
    session["current_question"] = result["next_question"]

    return AnswerResponse(
        session_id=request.session_id,
        question=memory_item["question"],
        answer=memory_item["answer"],
        score=result["score"],
        next_question=result["next_question"],
        memory=session["qa_history"],
        finished=result["next_question"] is None,
    )


@app.post("/report", response_model=ReportResponse)
def report(request: ReportRequest):
    session = SESSIONS.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Interview session not found.")

    text, average = generate_report(session)
    return ReportResponse(
        session_id=request.session_id,
        report=text,
        average_score=average,
        rounds=len(session.get("qa_history", [])),
    )
