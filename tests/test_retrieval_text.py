from cogdoc.tools.retriever.retrieval_text import retrieval_text


def test_retrieval_text_indexes_source_section_context_and_child_body():
    indexed = retrieval_text(
        {
            "text": "正文证据",
            "meta": {
                "source": "paper.pdf",
                "section_path": "Methods > Training",
                "context": "前文：实验设置",
            },
        }
    )

    assert indexed == (
        "来源：paper.pdf\n章节：Methods > Training\n\n前文：实验设置\n\n正文证据"
    )


def test_retrieval_text_keeps_sparse_legacy_docs_compatible():
    assert retrieval_text({"text": "正文", "meta": {}}) == "正文"


def test_retrieval_text_does_not_add_internal_source_id_to_derived_knowledge():
    assert (
        retrieval_text(
            {
                "text": "审核通过的补充规则",
                "meta": {
                    "source": "knowledge:K1",
                    "source_type": "derived_knowledge",
                    "context": "来源说明",
                },
            }
        )
        == "来源说明\n\n审核通过的补充规则"
    )
