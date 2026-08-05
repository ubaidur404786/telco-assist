"""
app.py - the TelcoAssist chat UI (Streamlit).

The bot answers in two steps the user can see:
  1. a collapsible "Thinking" panel shows the process - which engine it routed to,
     which tool it called, the SQL it ran (and the raw result) or the documents it
     retrieved;
  2. the final answer is a clean, natural sentence in the chat bubble.

Run:  streamlit run app.py
With no API key it runs in preview mode (routing + retrieval only).
"""

import random

import streamlit as st
from dotenv import load_dotenv

import models
from router import route, keyword_route
import sql_agent
import rag_agent

load_dotenv()

st.set_page_config(page_title="TelcoAssist", page_icon="📱", layout="centered")

USER_AVATAR = "🧑"
BOT_AVATAR = "📱"

DATA_EXAMPLES = [
    "How many customers churned in Ile-de-France?",
    "Which plan generates the most revenue?",
    "What is the average bill on the Unlimited Max plan?",
    "How many support tickets are still open?",
    "Which region has the highest churn rate?",
    "What is the total revenue from paid bills?",
]
RAG_EXAMPLES = [
    "What does roaming cost outside the EU?",
    "How do I cancel my contract?",
    "How do I activate my eSIM?",
    "My phone has no signal, what should I do?",
    "What travel passes do you offer?",
    "How much is the data overage charge?",
]


def pick_examples():
    """Two data + two policy questions, shuffled - so the suggestions stay mixed."""
    picks = random.sample(DATA_EXAMPLES, 2) + random.sample(RAG_EXAMPLES, 2)
    random.shuffle(picks)
    return picks


def gen_for(spec):
    return lambda system, user: models.generate(spec, system, user)


# ---------- session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "examples" not in st.session_state:
    st.session_state.examples = pick_examples()
if "pending" not in st.session_state:
    st.session_state.pending = None


# ---------- sidebar ----------
with st.sidebar:
    st.header("TelcoAssist")
    st.caption("HexaMobile support bot. Data questions use text-to-SQL; "
               "policy questions use document search (RAG).")

    available = models.available_models()
    if available:
        names = [s.name for s in available]
        chosen = st.selectbox("Model (free)", names, index=0)
        active_spec = next(s for s in available if s.name == chosen)
        st.caption(f"Provider: {active_spec.provider}")
    else:
        active_spec = None
        st.warning("No API key in .env — preview mode (routing + retrieval only).")

    st.divider()
    st.caption("Try one:")
    for q in st.session_state.examples:
        if st.button(q, key="ex_" + q, use_container_width=True):
            st.session_state.pending = q
    if st.button("↻ New examples", use_container_width=True):
        st.session_state.examples = pick_examples()
        st.rerun()

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


st.title("TelcoAssist")
st.caption("Ask about your account data or about HexaMobile policies.")


# ---------- render the stored "thinking" trace (history replay) ----------
def render_trace(msg):
    with st.expander("Show how this was answered"):
        st.markdown(f"**Routed to:** {msg['kind']} engine")
        if msg["kind"] == "data":
            st.markdown("**Tool:** text-to-SQL on `telco.db`")
            if msg.get("sql"):
                st.code(msg["sql"], language="sql")
            if msg.get("table") is not None and not msg["table"].empty:
                st.dataframe(msg["table"], use_container_width=True, hide_index=True)
        elif msg["kind"] == "knowledge":
            st.markdown("**Tool:** document search (RAG)")
            if msg.get("sources"):
                st.markdown("**Retrieved from:** " + ", ".join(msg["sources"]))


# ---------- replay history ----------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            if msg.get("kind") in ("data", "knowledge"):
                render_trace(msg)
            st.markdown(msg["content"])


# ---------- new input: typed, or from an example button ----------
prompt = st.chat_input("Ask a question…") or st.session_state.pending
st.session_state.pending = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        # ----- preview mode (no key): routing + retrieval only -----
        if active_spec is None:
            label, conf, d, k = keyword_route(prompt)
            note = f"I would route this to the **{label}** engine (data score {d}, knowledge score {k})."
            if label == "knowledge":
                from rag_common import retrieve
                srcs = ", ".join(sorted({h["source"] for h in retrieve(prompt, k=3)}))
                note += f" It would answer from: {srcs}."
            note += "\n\n_Add an API key to `.env` for the full answer._"
            st.markdown(note)
            st.session_state.messages.append(
                {"role": "assistant", "content": note, "kind": "preview"})

        # ----- full answer: show the thinking trace, then the clean answer -----
        else:
            gen = gen_for(active_spec)
            entry = {"role": "assistant"}
            with st.status("Thinking…", expanded=True) as status:
                st.write("Routing the question…")
                kind = route(prompt, gen=gen)
                st.write(f"→ **{kind}** engine")

                if kind == "data":
                    st.write("Calling the text-to-SQL tool…")
                    res = sql_agent.answer(gen, prompt)
                    st.write("Generated SQL:")
                    st.code(res["sql"], language="sql")
                    st.write("Ran it on `telco.db`. Result:")
                    if res["table"] is not None and not res["table"].empty:
                        st.dataframe(res["table"], use_container_width=True, hide_index=True)
                    entry.update(kind="data", content=res["answer"],
                                 sql=res["sql"], table=res["table"])
                else:
                    st.write("Searching the knowledge base…")
                    res = rag_agent.answer(gen, prompt)
                    st.write("Retrieved from: " + ", ".join(res["sources"]))
                    entry.update(kind="knowledge", content=res["answer"],
                                 sources=res["sources"])

                status.update(label=f"Answered via the {kind} engine",
                              state="complete", expanded=False)

            st.markdown(entry["content"])
            st.session_state.messages.append(entry)
