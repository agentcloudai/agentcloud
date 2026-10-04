"""Command line entry point.

  rag-app crawl https://...         # download PDFs from a JS-rendered site
  rag-app index                     # index files in RAG_DATA_DIR
  rag-app index --url https://...   # index web pages
  rag-app ask "your question" [--filter file_name=x.pdf]
  rag-app eval [--k 5]
"""
import argparse
import json

from rag_app.config import get_settings
from rag_app.logging_utils import setup_logging


def _parse_filters(pairs: list[str] | None) -> dict[str, str] | None:
    if not pairs:
        return None
    return dict(p.split("=", 1) for p in pairs)


def cmd_crawl(args) -> None:
    from rag_app.ingest.crawler import crawl_links
    from rag_app.ingest.downloader import download_files
    from rag_app.ingest.sites import pick_adapter

    adapter = pick_adapter(args.url)
    print(f"Using site adapter: {adapter.NAME}")

    links = crawl_links(
        args.url,
        adapter.is_relevant,
        max_pages=args.max_pages,
        headless=not args.show_browser,
    )
    print(f"Found {len(links)} relevant links.")

    # Map to direct download URLs and dedupe by the adapter's key (e.g. slug).
    download_urls: list[str] = []
    seen: set[str] = set()
    for link in links:
        url = adapter.to_download_url(link)
        if url is None:
            continue
        key = adapter.dedupe_key(url)
        if key in seen:
            continue
        seen.add(key)
        download_urls.append(url)
    print(f"{len(download_urls)} unique files to download.")

    if not download_urls:
        print("Nothing to download. If the site rendered no links, inspect crawl_debug.png.")
        return

    data_dir = get_settings().data_dir
    saved = download_files(download_urls, data_dir, limit=args.limit)
    print(f"\nDownloaded/kept {len(saved)} file(s) in {data_dir}.")
    print("Next: rag-app index")


def cmd_download(args) -> None:
    import csv

    from rag_app.ingest.downloader import download_files, filename_for
    from rag_app.ingest.manifest import update_manifest

    with open(args.csv, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    wanted = {s.lower() for s in (args.service or [])}
    if wanted:
        rows = [r for r in rows if r.get("title", "").lower() in wanted]
    if not rows:
        print("No matching rows in CSV (check --service names against the 'title' column).")
        return

    data_dir = get_settings().data_dir
    urls = [r["pdf_url"] for r in rows]
    saved = download_files(urls, data_dir, limit=args.limit)

    # Record where each saved file came from so chunks can be tagged by service.
    saved_names = {p.name for p in saved}
    records = [
        {
            "file_name": filename_for(r["pdf_url"]),
            "service": r.get("title", ""),
            "guide": r.get("guide", ""),
            "source_url": r["pdf_url"],
        }
        for r in rows
        if filename_for(r["pdf_url"]) in saved_names
    ]
    update_manifest(data_dir, records)

    services = sorted({r["service"] for r in records})
    print(f"\nDownloaded/kept {len(saved)} file(s) across {len(services)} service(s) in {data_dir}.")
    print(f"Manifest updated: chunks will be tagged with 'service' on index.")
    print("Next: rag-app index")


def cmd_serve(args) -> None:
    from rag_app.web.server import serve

    serve(host=args.host, port=args.port)


def cmd_index(args) -> None:
    import shutil
    from pathlib import Path

    from rag_app.pipelines.index_pipeline import run_indexing

    s = get_settings()
    if args.rebuild and s.vector_backend.lower() == "chroma":
        p = Path(s.chroma_path)
        if p.exists():
            shutil.rmtree(p)
            print(f"Rebuild: cleared {p}")
    n = run_indexing(s, urls=args.url)
    print(f"Indexed {n} chunks.")


def cmd_ask(args) -> None:
    from rag_app.pipelines.query_pipeline import RAGService

    result = RAGService().ask(args.question, _parse_filters(args.filter))

    if args.json:  # exactly what a UI would consume
        print(json.dumps(result.to_payload(), indent=2))
        return

    payload = result.to_payload()
    if payload["insufficient_context"]:
        print("\nNot enough information in the indexed documents to answer this.\n")
        return

    print(f"\n{payload['answer']}\n")          # crisp one-line answer
    if payload["steps"]:
        print("Steps:")
        for i, step in enumerate(payload["steps"], 1):
            print(f"  {i}. {step['text']} [{', '.join(step['sources'])}]")
    print("\nSources:")
    for src in payload["sources"]:
        page = f", p.{src['page']}" if src.get("page") else ""
        print(f"  [{src['id']}] {src['service']} — {src['guide']}{page}  (score {src['score']})")


def cmd_eval(args) -> None:
    from rag_app.evaluation.runner import run_eval
    from rag_app.pipelines.query_pipeline import RAGService

    s = get_settings()
    if args.file:
        s.eval_file = args.file
    summary = run_eval(RAGService(s), s, k=args.k)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


def cmd_eval_answers(args) -> None:
    from rag_app.evaluation.answer_eval import run_answer_eval
    from rag_app.pipelines.query_pipeline import RAGService

    s = get_settings()
    summary = run_answer_eval(RAGService(s), s, path=args.file, k=args.k, judge=args.judge)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    print("\nPer-question:")
    for r in summary["rows"]:
        if args.judge:
            flag = "OK " if r.get("judge_correct") else "!! "
            print(f"  {flag}judge={r['judge_coverage']:.0%} correct={r.get('judge_correct')} "
                  f"hit={r['retrieval_hit']} svc={r['service_match']}  {r['question'][:55]}")
            if r.get("judge_missing"):
                print(f"       missing: {', '.join(r['judge_missing'])}")
        else:
            flag = "OK " if r["step_coverage"] >= 0.6 and r["retrieval_hit"] else "!! "
            print(f"  {flag}cov={r['step_coverage']:.0%} hit={r['retrieval_hit']} "
                  f"svc={r['service_match']}  {r['question'][:60]}")
            if r["missing_terms"]:
                print(f"       missing: {', '.join(r['missing_terms'])}")


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(prog="rag-app", description="Modular RAG with LlamaIndex + pgvector")
    sub = parser.add_subparsers(dest="command", required=True)

    p_crawl = sub.add_parser("crawl", help="Download PDFs from a JavaScript-rendered site")
    p_crawl.add_argument("url", help="Listing page to crawl, e.g. https://aws.amazon.com/whitepapers/")
    p_crawl.add_argument("--max-pages", type=int, default=50, help="Max listing pages to visit")
    p_crawl.add_argument("--limit", type=int, default=None, help="Max files to download")
    p_crawl.add_argument("--show-browser", action="store_true", help="Run Chromium with a visible window")
    p_crawl.set_defaults(func=cmd_crawl)

    p_dl = sub.add_parser("download", help="Download PDFs listed in a CSV (title,guide,pdf_url)")
    p_dl.add_argument("--csv", required=True, help="CSV with columns: title, guide, pdf_url")
    p_dl.add_argument("--service", action="append", help="Only this service title (repeatable)")
    p_dl.add_argument("--limit", type=int, default=None, help="Max files to download")
    p_dl.set_defaults(func=cmd_download)

    p_serve = sub.add_parser("serve", help="Launch the local web UI")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    p_index = sub.add_parser("index", help="Load, chunk, embed and store documents")
    p_index.add_argument("--url", action="append", help="Index a web page instead of the data folder (repeatable)")
    p_index.add_argument("--rebuild", action="store_true", help="Clear the vector store first (clean full rebuild)")
    p_index.set_defaults(func=cmd_index)

    p_ask = sub.add_parser("ask", help="Ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--filter", action="append", help="Metadata filter key=value (repeatable)")
    p_ask.add_argument("--json", action="store_true", help="Emit the UI-ready JSON payload")
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("eval", help="Measure retrieval quality (recall@k, precision@k, MRR)")
    p_eval.add_argument("--k", type=int, default=5)
    p_eval.add_argument("--file", help="Eval JSONL file (default: RAG_EVAL_FILE)")
    p_eval.set_defaults(func=cmd_eval)

    p_evala = sub.add_parser("eval-answers", help="Check generated answers contain the correct steps (needs LLM key)")
    p_evala.add_argument("--file", default="eval/architecture_questions.jsonl", help="Eval JSONL with key_terms")
    p_evala.add_argument("--k", type=int, default=5)
    p_evala.add_argument("--judge", action="store_true", help="Use an LLM to grade step correctness")
    p_evala.set_defaults(func=cmd_eval_answers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
