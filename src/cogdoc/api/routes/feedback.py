from fastapi import APIRouter, Request
from cogdoc.api.schemas import FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/v1", tags=["feedback"])


@router.post("/feedback", status_code=201)
async def submit_feedback(body: FeedbackRequest, request: Request):
    # controller 只落盘，不做评判；坏样本归集逻辑在 store 里。
    result = request.app.state.feedback_store.record(body.model_dump(exclude_none=True))
    return FeedbackResponse(
        feedback_id=result["feedback_id"], is_bad_case=result["is_bad_case"]
    )
