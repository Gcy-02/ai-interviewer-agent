const sampleBtn = document.querySelector("#sample-btn");
const startBtn = document.querySelector("#start-btn");
const answerBtn = document.querySelector("#answer-btn");
const demoAnswerBtn = document.querySelector("#demo-answer-btn");
const reportBtn = document.querySelector("#report-btn");
const resumeText = document.querySelector("#resume-text");
const jdText = document.querySelector("#jd-text");
const answerText = document.querySelector("#answer-text");
const setupStatus = document.querySelector("#setup-status");
const sessionIdBox = document.querySelector("#session-id");
const questionBox = document.querySelector("#question");
const scoreBox = document.querySelector("#score-box");
const memoryList = document.querySelector("#memory-list");
const reportOutput = document.querySelector("#report-output");

let sessionId = null;
const demoAnswer =
  "我做过一个企业知识库 RAG 项目，主要负责 FastAPI 接口、PDF 上传、Chunk 切分、ChromaDB 检索、Top-K 召回展示和引用来源返回。项目最后可以在 GitHub 上展示完整页面截图，也能用 Swagger 验证接口。";

function renderMemory(memory) {
  memoryList.innerHTML = "";
  memory.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = `${item.question.focus} · ${item.score.score} 分`;
    memoryList.appendChild(li);
  });
}

function renderScore(score) {
  const strengths = score.strengths.map((item) => `- ${item}`).join("\n");
  const improvements = score.improvements.map((item) => `- ${item}`).join("\n");
  scoreBox.textContent = `得分：${score.score}\n\n亮点：\n${strengths}\n\n建议：\n${improvements}`;
}

sampleBtn.addEventListener("click", async () => {
  setupStatus.textContent = "正在加载示例简历和 JD...";
  const response = await fetch("/sample");
  const data = await response.json();
  resumeText.value = data.resume_text;
  jdText.value = data.jd_text;
  setupStatus.textContent = "示例材料已加载，可以开始面试。";
});

startBtn.addEventListener("click", async () => {
  setupStatus.textContent = "正在运行 LangGraph 工作流...";
  const response = await fetch("/start_interview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      resume_text: resumeText.value,
      jd_text: jdText.value,
      session_id: sessionId,
    }),
  });
  const data = await response.json();
  if (!response.ok) {
    setupStatus.textContent = data.detail || "启动失败。";
    return;
  }

  sessionId = data.session_id;
  sessionIdBox.textContent = sessionId;
  questionBox.textContent = data.question.question;
  scoreBox.textContent = "等待第一轮回答。";
  memoryList.innerHTML = "";
  reportOutput.textContent = "完成几轮回答后，可以生成复盘报告。";
  setupStatus.textContent = `已生成 ${data.questions.length} 个问题。`;
});

answerBtn.addEventListener("click", async () => {
  if (!sessionId) {
    scoreBox.textContent = "请先开始面试。";
    return;
  }

  scoreBox.textContent = "正在评分并生成追问...";
  const response = await fetch("/answer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      answer: answerText.value,
    }),
  });
  const data = await response.json();
  if (!response.ok) {
    scoreBox.textContent = data.detail || "提交失败。";
    return;
  }

  renderScore(data.score);
  renderMemory(data.memory);
  answerText.value = "";
  questionBox.textContent = data.next_question
    ? data.next_question.question
    : "这一轮面试问题已经结束，可以生成复盘报告。";
});

demoAnswerBtn.addEventListener("click", () => {
  answerText.value = demoAnswer;
});

reportBtn.addEventListener("click", async () => {
  if (!sessionId) {
    reportOutput.textContent = "请先开始面试。";
    return;
  }

  const response = await fetch("/report", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  const data = await response.json();
  if (!response.ok) {
    reportOutput.textContent = data.detail || "生成报告失败。";
    return;
  }

  reportOutput.textContent = data.report;
});
