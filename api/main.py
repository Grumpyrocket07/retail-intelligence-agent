"""The prediction API.  Start it with:  uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs to try every endpoint from the browser."""
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request

from api import settings
from api.schemas import (ChurnResult, ChurnTopResponse, ComponentStatus, HealthResponse, SentimentBatchRequest,
                         SentimentBatchResponse, SentimentRequest, SentimentResult)
from api.services import SentimentService, build_churn_service, sentiment_label

log = logging.getLogger("api")


def create_app(churn=None, sentiment=None, load_real: bool = True) -> FastAPI:
    """`churn` / `sentiment` let tests plug in stand-ins. With load_real=False nothing is loaded at start-up."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.churn, app.state.sentiment, app.state.errors = churn, sentiment, {}
        if load_real:
            if churn is None:
                try:
                    app.state.churn = build_churn_service()
                except Exception as exc:  # keep the API up so /health can explain what is wrong
                    log.exception("Churn service failed to load")
                    app.state.errors["churn"] = f"{type(exc).__name__}: {exc}"
            if sentiment is None:
                try:
                    app.state.sentiment = SentimentService(settings.SENTIMENT_MODEL_PATH)
                except Exception as exc:
                    log.exception("Sentiment service failed to load")
                    app.state.errors["sentiment"] = f"{type(exc).__name__}: {exc}"
        yield

    app = FastAPI(
        title="Retail Intelligence Agent: prediction API",
        description="Ask the churn and sentiment models questions. Churn scores use 'today' = the day after the last order in the data.",
        version="0.1.0", lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        log.info("%s %s -> %d in %.0f ms", request.method, request.url.path, response.status_code,
                 (time.perf_counter() - start) * 1000)
        return response

    def _need(request: Request, name: str):
        service = getattr(request.app.state, name)
        if service is None:
            why = request.app.state.errors.get(name, "not loaded")
            raise HTTPException(status_code=503, detail=f"The {name} model is not available ({why}). See /health.")
        return service

    @app.get("/health", response_model=HealthResponse, tags=["status"])
    def health(request: Request):
        st = request.app.state
        churn_ok, sent_ok = st.churn is not None, st.sentiment is not None
        churn_detail = (f"{st.churn.model_name}, as of {st.churn.as_of.date()}, {st.churn.customers_scored} customers scored"
                        if churn_ok else st.errors.get("churn", "not loaded"))
        sent_detail = "loaded" if sent_ok else st.errors.get("sentiment", "not loaded")
        return HealthResponse(status="ok" if churn_ok and sent_ok else "degraded",
                              churn=ComponentStatus(loaded=churn_ok, detail=churn_detail),
                              sentiment=ComponentStatus(loaded=sent_ok, detail=sent_detail))

    def _sentiment_results(service, texts: list[str]) -> list[SentimentResult]:
        try:
            probs = service.positive_probabilities(texts)
        except Exception as exc:  # show the real reason in the response while the project is in development
            log.exception("Sentiment scoring failed")
            raise HTTPException(status_code=500, detail=f"Sentiment model failed: {type(exc).__name__}: {exc}")
        return [SentimentResult(text=t, label=sentiment_label(p), positive_probability=round(float(p), 4))
                for t, p in zip(texts, probs)]

    @app.post("/sentiment", response_model=SentimentResult, tags=["sentiment"])
    def sentiment_one(body: SentimentRequest, request: Request):
        """Is this review positive or negative? Scores near the middle come back as 'uncertain'."""
        return _sentiment_results(_need(request, "sentiment"), [body.text])[0]

    @app.post("/sentiment/batch", response_model=SentimentBatchResponse, tags=["sentiment"])
    def sentiment_many(body: SentimentBatchRequest, request: Request):
        """Same as /sentiment for up to 64 texts at once."""
        return SentimentBatchResponse(results=_sentiment_results(_need(request, "sentiment"), body.texts))

    # NOTE: /churn/top must be declared before /churn/{customer_id}, or "top" would be read as a customer id.
    @app.get("/churn/top", response_model=ChurnTopResponse, tags=["churn"])
    def churn_top(request: Request, limit: int = Query(10, ge=1, le=100)):
        """The customers most likely to leave, highest score first."""
        svc = _need(request, "churn")
        return ChurnTopResponse(as_of=str(svc.as_of.date()), model=svc.model_name, threshold=round(svc.threshold, 4),
                                customers_scored=svc.customers_scored, customers=svc.top(limit))

    @app.get("/churn/{customer_id}", response_model=ChurnResult, tags=["churn"])
    def churn_one(customer_id: int, request: Request):
        """How likely is this customer to place no order in the next 90 days?"""
        result: Optional[dict] = _need(request, "churn").lookup(customer_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Customer {customer_id} has no order history in the data.")
        return result

    return app


app = create_app()
