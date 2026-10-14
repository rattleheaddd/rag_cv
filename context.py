def build_context(results):
    context = []
    for source, result in enumerate(results, start=1):
        context.append(f"[Source {source} | Page {result['page']}]\n"
                        f"{result['text']}")

    return "\n\n".join(context)