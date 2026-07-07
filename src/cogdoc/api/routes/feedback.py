from fastapi import APIRouter, Request
from cogdoc.api.schemas import FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/v1", tags=["feedback"])


# 构建纠错派生知识草稿。
def _knowledge_payload(body: FeedbackRequest) -> dict | None:
    correction = body.correction_text or body.correction
    if not body.save_as_knowledge or not correction or not body.kb_id:
        return None
    related_chunk_ids = body.related_chunk_ids or [
        item.chunk_id for item in body.citations if item.chunk_id
    ]
    related_source = body.related_source or next(
        (item.source for item in body.citations if item.source), None
    )
    return {
        "kb_id": body.kb_id,
        "text": correction,
        "related_document_id": body.related_document_id,
        "related_source": related_source,
        "related_source_sha256": body.related_source_sha256,
        "related_chunk_ids": related_chunk_ids,
        "source_note": body.source_note or body.feedback_text or body.comment,
        "certainty": body.certainty,
        "status": "pending",
        "origin": "correction",
        "created_from_trace_id": body.trace_id,
        "created_by": body.created_by,
    }


# 提交反馈。
@router.post("/feedback", status_code=201)
async def submit_feedback(body: FeedbackRequest, request: Request):
    # 控制层只落盘，不做评判；坏样本归集逻辑在存储层里。
    payload = body.model_dump(exclude_none=True)
    if body.feedback_text and not payload.get("comment"):
        payload["comment"] = body.feedback_text
    if body.correction_text and not payload.get("correction"):
        payload["correction"] = body.correction_text
    result = request.app.state.feedback_store.record(payload)
    knowledge_id = None
    knowledge_status = None
    knowledge_deduplicated = False
    knowledge_payload = _knowledge_payload(body)
    if knowledge_payload is not None:
        knowledge, knowledge_deduplicated = request.app.state.knowledge_store.create(
            knowledge_payload
        )
        knowledge_id = knowledge["knowledge_id"]
        knowledge_status = knowledge["status"]
    return FeedbackResponse(
        feedback_id=result["feedback_id"],
        is_bad_case=result["is_bad_case"],
        knowledge_id=knowledge_id,
        knowledge_status=knowledge_status,
        knowledge_deduplicated=knowledge_deduplicated,
    )
