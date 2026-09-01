"""
Streamlit UI for the Investment Trends Scanner.

One page: a question box, an "Ask" button, and a read-only answer box.
Uses the same build_agent()/ask() API as run.py -- this is just a
different front end on top of the same agent.

Run from the repo root (with your venv active):
    streamlit run app.py
"""

import asyncio
import threading
import uuid

import streamlit as st

from investment_scanner import ask, build_agent

st.set_page_config(page_title="Investment Trends Scanner", page_icon="📈")


def _start_background_loop() -> asyncio.AbstractEventLoop:
    """Run one event loop forever in a background thread.

    Streamlit reruns this script on every interaction. asyncio.run() opens
    a fresh event loop each time and closes it when done -- but the agent
    is cached across reruns (st.cache_resource) and its Ollama/MCP clients
    lazily create async httpx connections tied to whichever loop was
    running on first use. Once that loop is closed, reusing the agent on
    the next question raises "RuntimeError: Event loop is closed". Running
    every async call against one loop that never closes avoids that.
    """
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return loop


@st.cache_resource(show_spinner=False)
def get_loop():
    return _start_background_loop()


def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, get_loop()).result()


st.markdown(
    """
    <style>
    textarea:disabled {
        color: #fafafa !important;
        -webkit-text-fill-color: #fafafa !important;
        opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Starting agent (Ollama + MCP server)...")
def get_agent():
    return run_async(build_agent())


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "answer" not in st.session_state:
    st.session_state.answer = ""
if "used_cache" not in st.session_state:
    st.session_state.used_cache = False

st.title("Investment Trends Scanner")
st.caption(
    "Educational demo -- news/trend summarizer for equities and ETFs only. "
    "Not financial advice."
)

question = st.text_area(
    "Question",
    placeholder="e.g. What's the latest news on Nvidia and semiconductor ETFs?",
    height=100,
)

if st.button("Ask", type="primary") and question.strip():
    agent = get_agent()
    with st.spinner("Thinking..."):
        st.session_state.answer, st.session_state.used_cache = run_async(
            ask(agent, question, thread_id=st.session_state.thread_id)
        )

if st.session_state.used_cache:
    st.info("Some news came from the local cache (market_log.md), not a live search.")

st.text_area("Answer", value=st.session_state.answer, height=300, disabled=True)
