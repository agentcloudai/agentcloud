"""AgentCloud interactive CLI: `agtcld`.

Shows the logo, asks whether to work in the terminal or open the web app, then
runs an interactive agent in the terminal (ask how-tos, 'design ...' for
architectures) — or launches the immersive localhost UI.
"""
from rag_app.branding import agent_say, enable_utf8, print_banner, PROMPT

enable_utf8()  # ensure unicode-safe output even if helpers run before the banner

_C = "\033[96m"; _O = "\033[38;5;214m"; _G = "\033[90m"; _B = "\033[1m"; _R = "\033[0m"


def _save_openai_key(key: str) -> None:
    import pathlib

    p = pathlib.Path(".env")
    lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
    out, found = [], False
    for ln in lines:
        if ln.strip().lstrip("#").strip().startswith("OPENAI_API_KEY"):
            out.append(f"OPENAI_API_KEY={key}"); found = True
        else:
            out.append(ln)
    if not found:
        out.append(f"OPENAI_API_KEY={key}")
    p.write_text("\n".join(out) + "\n", encoding="utf-8")


def _ensure_api_key() -> None:
    """Make sure the agent has an LLM to answer with."""
    import os

    from rag_app.config import get_settings

    # A key saved from the web UI lives next to the index; reuse it so the two
    # modes don't each ask for one.
    try:
        from rag_app.web.server import load_saved_key
        load_saved_key()
        get_settings.cache_clear()
    except Exception:  # noqa: BLE001 - never block the CLI on this
        pass

    s = get_settings()
    if s.openai_api_key or os.environ.get("OPENAI_API_KEY") or s.llm_provider == "vllm":
        return
    print(agent_say(f"I need an LLM to answer. Paste your {_B}OpenAI API key{_R} "
                    f"(starts with sk-…), or press Enter to skip (e.g. if you run a local vLLM server)."))
    try:
        key = input(f"   {_O}OpenAI key >{_R} ").strip()
    except (EOFError, KeyboardInterrupt):
        key = ""
    if key:
        _save_openai_key(key)
        os.environ["OPENAI_API_KEY"] = key
        get_settings.cache_clear()
        print(agent_say(f"{_C}saved to .env ✓{_R}  (rotate it anytime at platform.openai.com)\n"))
    else:
        print(agent_say(f"{_G}skipped — set OPENAI_API_KEY in .env, or RAG_LLM_PROVIDER=vllm, before asking.{_R}\n"))


def _explain_setup() -> None:
    """Tell the user how storage works — no Docker needed by default."""
    import pathlib

    from rag_app.config import get_settings

    s = get_settings()
    if s.vector_backend.lower() == "chroma":
        print(agent_say(f"Storage: {_B}embedded{_R} vector store (Chroma) under {_C}data/chroma{_R} — "
                        f"{_B}no Docker, no database to install.{_R}"))
        print(f"   {_G}(For large / multi-tenant / SaaS use: set RAG_VECTOR_BACKEND=pgvector and run "
              f"Postgres via `docker compose --profile pgvector up -d`.){_R}")
        idx = pathlib.Path(s.chroma_path) / "chroma.sqlite3"
        if not idx.exists():
            print(agent_say(f"{_O}No index yet.{_R} Put PDFs/TXT/MD in {_C}data/raw{_R} and run "
                            f"{_C}rag-app index{_R} first."))
    else:
        print(agent_say(f"Storage: {_B}{s.vector_backend}{_R} (external) — make sure it is running "
                        f"(e.g. `docker compose --profile pgvector up -d`)."))
    print()


def _open_web() -> None:
    import threading
    import webbrowser

    from rag_app.web.server import serve

    url = "http://127.0.0.1:8000"
    print(agent_say(f"Launching the immersive web app at {_C}{url}{_R} …\n"))
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    serve()


def _print_answer(p: dict) -> None:
    if p.get("insufficient_context"):
        print(agent_say(f"{_G}I couldn't find enough in the indexed cloud docs to answer that.{_R}\n"))
        return
    print("\n" + agent_say(f"{_B}{p['answer']}{_R}"))
    for i, s in enumerate(p.get("steps", []), 1):
        print(f"   {_O}{i}.{_R} {s['text']}  {_G}[{', '.join(s.get('sources', []))}]{_R}")
    if p.get("sources"):
        print(f"   {_G}sources: " + "; ".join(
            f"{s['id']} {s.get('service') or s.get('file')}" for s in p["sources"][:6]) + _R)
    print()


def _print_design(d: dict) -> None:
    if d.get("insufficient_context"):
        print(agent_say(f"{_G}Not enough in the docs to design that — try naming the workload.{_R}\n"))
        return
    print("\n" + agent_say(f"{_B}{_O}Architecture design{_R}  {d['overview']}"))
    if d.get("clarifying_questions"):
        print(f"\n   {_O}To tailor this, tell me:{_R}")
        for q in d["clarifying_questions"]:
            print(f"     {_G}?{_R} {q}")
    if d.get("components"):
        print(f"\n   {_C}Components{_R}")
        for c in d["components"]:
            print(f"     {_O}•{_R} {_B}{c['name']}{_R} ({c.get('service','')}) — {c['purpose']}")
    if d.get("data_flow"):
        print(f"\n   {_C}Data flow{_R}")
        for i, f in enumerate(d["data_flow"], 1):
            why = f"  {_G}↳ {f['why']}{_R}" if f.get("why") else ""
            print(f"     {_O}{i}.{_R} {f['step']}{why}")
    if d.get("considerations"):
        print(f"\n   {_C}Considerations{_R}")
        for c in d["considerations"]:
            print(f"     {_O}•{_R} {c}")
    print(f"\n   {_G}tip: run the web app for an interactive 3D view of this architecture.{_R}\n")


def _repl() -> None:
    from rag_app.generation.design import is_design_question
    from rag_app.pipelines.query_pipeline import RAGService

    print(agent_say(f"{_G}warming up models…{_R}"))
    svc = RAGService()
    print(agent_say("Ready. Ask a how-to, or 'design …' for an architecture. Type 'exit' to quit.\n"))
    while True:
        try:
            q = input(PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "q"):
            break
        try:
            if is_design_question(q):
                _print_design(svc.design(q))
            else:
                _print_answer(svc.ask(q).to_payload())
        except Exception as exc:  # noqa: BLE001
            print(agent_say(f"\033[91merror:\033[0m {exc}\n"))
    print(agent_say("bye.\n"))


def main() -> None:
    print_banner()
    _explain_setup()
    _ensure_api_key()
    try:
        choice = input(f"Work in the {_B}terminal{_R}, or open the {_B}web app{_R}?  [{_O}T{_R}/w] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if choice.startswith("w"):
        _open_web()
    else:
        _repl()


if __name__ == "__main__":
    main()
