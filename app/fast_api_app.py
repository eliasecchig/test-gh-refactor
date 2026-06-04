# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import json
import os
from collections.abc import AsyncGenerator, AsyncIterator, Callable

import google.auth
from fastapi import FastAPI, encoders, responses
from google.adk.cli.fast_api import get_fast_api_app
from google.cloud import logging as google_cloud_logging
from pydantic import BaseModel

from app.app_utils.telemetry import setup_telemetry
from app.app_utils.typing import Feedback

setup_telemetry()
_, project_id = google.auth.default()
logging_client = google_cloud_logging.Client()
logger = logging_client.logger(__name__)
allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None
)

# Artifact bucket for ADK (created by Terraform, passed via env var)
logs_bucket_name = os.environ.get("LOGS_BUCKET_NAME")

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# In-memory session configuration - no persistent storage
session_service_uri = None

artifact_service_uri = f"gs://{logs_bucket_name}" if logs_bucket_name else None

app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    artifact_service_uri=artifact_service_uri,
    allow_origins=allow_origins,
    session_service_uri=session_service_uri,
    otel_to_cloud=True,
)
app.title = "te3st122aassa"
app.description = "API for interacting with the Agent te3st122aassa"


# =============================================================================
# Agent Runtime Application Entrypoint Logic
# =============================================================================
# These imports are deliberately below the FastAPI app setup above. ruff
# E402 is silenced because the unified template serves two roles in one
# module (web server + Agent Engine runtime) and keeping the runtime
# imports near their usage aids readability.
import logging  # noqa: E402
from typing import Any  # noqa: E402

import vertexai  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from google.adk.artifacts import (  # noqa: E402
    GcsArtifactService,
    InMemoryArtifactService,
)

from app.agent import app as adk_app  # noqa: E402

load_dotenv()
from vertexai.agent_engines.templates.adk import AdkApp  # noqa: E402


class AgentEngineApp(AdkApp):
    def set_up(self) -> None:
        """Initialize the agent engine app with logging and telemetry."""
        vertexai.init()
        setup_telemetry()
        super().set_up()
        logging.basicConfig(level=logging.INFO)
        logging_client = google_cloud_logging.Client()
        self.logger = logging_client.logger(__name__)
        if gemini_location:
            os.environ["GOOGLE_CLOUD_LOCATION"] = gemini_location

    def register_feedback(self, feedback: dict[str, Any]) -> None:
        """Collect and log feedback."""
        feedback_obj = Feedback.model_validate(feedback)
        self.logger.log_struct(feedback_obj.model_dump(), severity="INFO")

    def register_operations(self) -> dict[str, list[str]]:
        """Registers the operations of the Agent."""
        operations = super().register_operations()
        operations[""] = [*operations.get("", []), "register_feedback"]
        return operations


gemini_location = os.environ.get("GOOGLE_CLOUD_LOCATION")
logs_bucket_name = os.environ.get("LOGS_BUCKET_NAME")
agent_runtime = AgentEngineApp(
    app=adk_app,
    artifact_service_builder=lambda: (
        GcsArtifactService(bucket_name=logs_bucket_name)
        if logs_bucket_name
        else InMemoryArtifactService()
    ),
)


class QueryRequest(BaseModel):
    input: dict | None = None
    class_method: str | None = None


def _encode_chunk_to_json(chunk: Any) -> str x| None:
    """Encodes a chunk to a JSON string with a newline."""
    try:
        json_chunk = encoders.jsonable_encoder(chunk)
        return json.dumps(json_chunk) + "\n"
    except Exception:
        logging.exception("Failed to encode chunk")
        return None


async def json_generator(output: AsyncIterator[Any]) -> AsyncGenerator[str, None]:
    async for chunk in output:
        encoded_chunk = _encode_chunk_to_json(chunk)
        if encoded_chunk is None:
            break
        yield encoded_chunk


async def _invoke_callable_or_raise(
    invocation_callable: Callable[..., Any], invocation_payload: dict[str, Any]
) -> Any:
    if inspect.iscoroutinefunction(invocation_callable):
        return await invocation_callable(**invocation_payload)
    else:
        return invocation_callable(**invocation_payload)


# Intercept legacy SDK proxy requests for Vertex AI Console Playground support
@app.post("/api/reasoning_engine")
async def query(request: QueryRequest) -> responses.JSONResponse:
    method = getattr(agent_runtime, request.class_method)
    output = await _invoke_callable_or_raise(method, request.input or {})

    try:
        json_serialized_content = encoders.jsonable_encoder({"output": output})
    except ValueError as encoding_error:
        logging.exception(
            "FastAPI could not JSON-encode the response from invocation method"
            " %s. Error: %s. Invocation method's original response: %r",
            request.class_method,
            encoding_error,
            output,
        )
        raise encoding_error
    return responses.JSONResponse(content=json_serialized_content)


@app.post("/api/stream_reasoning_engine")
async def stream_query(request: QueryRequest) -> responses.StreamingResponse:
    method = getattr(agent_runtime, request.class_method)
    output = await _invoke_callable_or_raise(method, request.input or {})
    return responses.StreamingResponse(
        content=json_generator(output),
        media_type="application/json",
    )


@app.post("/feedback")
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    """Collect and log feedback.

    Args:
        feedback: The feedback data to log

    Returns:
        Success message
    """
    logger.log_struct(feedback.model_dump(), severity="INFO")
    return {"status": "success"}


# Main execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
